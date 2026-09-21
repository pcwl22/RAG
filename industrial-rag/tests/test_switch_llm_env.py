from pathlib import Path

import pytest

from app.utils.strict_dotenv import load_release_env_file
from scripts.switch_llm_env import (
    activate_llm_profile,
    discover_llm_profiles,
    main,
    normalize_api_url,
    switch_llm_env,
)


def _write_env(path: Path) -> bytes:
    content = (
        b"# retained comment\r\n"
        b"RAG_ENV=laptop\r\n"
        b"DEEPSEEK_API_KEY=old-secret-key-1234567890\r\n"
        b"DEEPSEEK_API_URL=https://old.example.test/v1\r\n"
        b"DEEPSEEK_MODEL=old-model\r\n"
    )
    path.write_bytes(content)
    return content


def _write_profile_env(path: Path) -> bytes:
    content = (
        b"#DEEPSEEK_MODEL=profile-one-pro\n"
        b"#DEEPSEEK_MODEL=profile-one-flash\n"
        b"#DEEPSEEK_API_KEY=profile-one-secret-123456\n"
        b"#DEEPSEEK_API_URL=https://one.example.test/v1\n"
        b"\n"
        b"DEEPSEEK_MODEL=current-model\n"
        b"DEEPSEEK_API_KEY=current-secret-123456789\n"
        b"DEEPSEEK_API_URL=https://current.example.test/v1\n"
        b"\n"
        b"#DEEPSEEK_MODEL=profile-three-model\n"
        b"#DEEPSEEK_API_KEY=profile-three-secret-123456\n"
        b"#DEEPSEEK_API_URL=https://three.example.test/v1\n"
    )
    path.write_bytes(content)
    return content


def test_switch_updates_all_llm_values_atomically_without_returning_secret(tmp_path: Path):
    env_file = tmp_path / ".env"
    _write_env(env_file)

    summary = switch_llm_env(
        env_file,
        api_url="https://new.example.test/compatible-mode/v1/",
        model="vendor/new-model:latest",
        api_key="new-secret-key-1234567890",
    )

    values = load_release_env_file(env_file)
    assert values["RAG_ENV"] == "laptop"
    assert values["DEEPSEEK_API_KEY"] == "new-secret-key-1234567890"
    assert values["DEEPSEEK_API_URL"] == "https://new.example.test/compatible-mode/v1"
    assert values["DEEPSEEK_MODEL"] == "vendor/new-model:latest"
    assert env_file.read_bytes().startswith(b"# retained comment\r\n")
    assert summary.api_key_changed is True
    assert "new-secret-key" not in repr(summary)


def test_switch_can_preserve_api_key_and_append_missing_values(tmp_path: Path):
    env_file = tmp_path / ".env"
    env_file.write_text("DEEPSEEK_API_KEY=existing-secret-123456\n", encoding="utf-8")

    summary = switch_llm_env(
        env_file,
        api_url="https://api.example.test",
        model="model-2",
    )

    values = load_release_env_file(env_file)
    assert values == {
        "DEEPSEEK_API_KEY": "existing-secret-123456",
        "DEEPSEEK_API_URL": "https://api.example.test/v1",
        "DEEPSEEK_MODEL": "model-2",
    }
    assert summary.api_key_changed is False


def test_cli_reads_api_key_from_environment_without_printing_it(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
):
    env_file = tmp_path / ".env"
    _write_env(env_file)
    secret = "next-secret-key-1234567890"
    monkeypatch.setenv("NEXT_LLM_API_KEY", secret)
    monkeypatch.setattr(
        "sys.argv",
        [
            "switch_llm_env.py",
            "--env-file",
            str(env_file),
            "--api-key-env",
            "NEXT_LLM_API_KEY",
        ],
    )

    main()

    captured = capsys.readouterr()
    assert secret not in captured.out
    assert secret not in captured.err
    assert "updated (value hidden)" in captured.out
    assert load_release_env_file(env_file)["DEEPSEEK_API_KEY"] == secret


def test_profile_discovery_hides_keys_and_preserves_multiple_model_choices(
    tmp_path: Path,
):
    env_file = tmp_path / ".env"
    content = _write_profile_env(env_file).decode()

    profiles = discover_llm_profiles(content)

    assert len(profiles) == 3
    assert profiles[0].commented is True
    assert profiles[0].models == ("profile-one-pro", "profile-one-flash")
    assert profiles[1].commented is False
    assert profiles[1].active_model == "current-model"
    assert "profile-one-secret" not in repr(profiles[0])


def test_activate_profile_switches_comments_and_preserves_previous_profile(tmp_path: Path):
    env_file = tmp_path / ".env"
    original = _write_profile_env(env_file)

    summary = activate_llm_profile(env_file, profile_index=3)

    values = load_release_env_file(env_file)
    assert values["DEEPSEEK_MODEL"] == "profile-three-model"
    assert values["DEEPSEEK_API_KEY"] == "profile-three-secret-123456"
    assert values["DEEPSEEK_API_URL"] == "https://three.example.test/v1"
    switched = env_file.read_bytes()
    assert b"DEEPSEEK_API_KEY=profile-three-secret-123456" in switched
    assert b"#DEEPSEEK_API_KEY=current-secret-123456789" in switched
    assert env_file.read_bytes() != original
    assert summary.api_key_changed is True
    assert "profile-three-secret" not in repr(summary)

    profiles = discover_llm_profiles(switched.decode())
    assert profiles[1].commented is True
    assert profiles[2].commented is False
    assert profiles[2].active_model == "profile-three-model"


def test_activate_multi_model_profile_requires_explicit_declared_model(tmp_path: Path):
    env_file = tmp_path / ".env"
    original = _write_profile_env(env_file)

    with pytest.raises(ValueError, match="multiple models"):
        activate_llm_profile(env_file, profile_index=1)
    with pytest.raises(ValueError, match="is not declared"):
        activate_llm_profile(
            env_file,
            profile_index=1,
            profile_model="unknown-model",
        )
    assert env_file.read_bytes() == original

    activate_llm_profile(
        env_file,
        profile_index=1,
        profile_model="profile-one-flash",
    )
    assert load_release_env_file(env_file)["DEEPSEEK_MODEL"] == "profile-one-flash"


def test_cli_lists_profiles_without_printing_keys(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
):
    env_file = tmp_path / ".env"
    _write_profile_env(env_file)
    monkeypatch.setattr(
        "sys.argv",
        ["switch_llm_env.py", "--env-file", str(env_file), "--list-profiles"],
    )

    main()

    captured = capsys.readouterr()
    assert "profile=1 state=commented" in captured.out
    assert "models=profile-one-pro,profile-one-flash" in captured.out
    assert "url_host=one.example.test" in captured.out
    assert "api_key=present" in captured.out
    assert "secret" not in captured.out


def test_switch_dry_run_validates_without_mutating_file(tmp_path: Path):
    env_file = tmp_path / ".env"
    original = _write_env(env_file)

    summary = switch_llm_env(
        env_file,
        api_url="https://api.example.test/v1",
        model="model-2",
        dry_run=True,
    )

    assert summary.dry_run is True
    assert env_file.read_bytes() == original


@pytest.mark.parametrize(
    "duplicate",
    [
        "export DEEPSEEK_MODEL=second",
        "'DEEPSEEK_MODEL'=second",
        '\"DEEPSEEK_MODEL\"=second',
    ],
)
def test_switch_rejects_duplicate_target_key_without_mutating_file(
    tmp_path: Path, duplicate: str
):
    env_file = tmp_path / ".env"
    original = f"DEEPSEEK_MODEL=first\n{duplicate}\n".encode()
    env_file.write_bytes(original)

    with pytest.raises(ValueError, match="duplicate dotenv key"):
        switch_llm_env(env_file, model="replacement")

    assert env_file.read_bytes() == original


@pytest.mark.parametrize(
    "api_url",
    [
        "not-a-url",
        "https://user:password@example.test/v1",
        "https://example.test/v1?token=secret",
        "http://example.test/v1",
    ],
)
def test_switch_rejects_unsafe_endpoint_without_mutating_file(
    tmp_path: Path, api_url: str
):
    env_file = tmp_path / ".env"
    original = _write_env(env_file)

    with pytest.raises(ValueError):
        switch_llm_env(env_file, api_url=api_url)

    assert env_file.read_bytes() == original


def test_only_explicit_local_http_override_is_allowed():
    assert (
        normalize_api_url("http://127.0.0.1:11434", allow_local_http=True)
        == "http://127.0.0.1:11434/v1"
    )
    with pytest.raises(ValueError, match="must use https"):
        normalize_api_url("http://127.0.0.1:11434")


@pytest.mark.parametrize(
    "api_key",
    [
        "short",
        "change-me-to-a-real-key",
        "contains whitespace 123456789",
        "contains${INTERPOLATION}123456789",
        r"contains\backslash123456789",
    ],
)
def test_switch_rejects_invalid_api_key(tmp_path: Path, api_key: str):
    env_file = tmp_path / ".env"
    original = _write_env(env_file)

    with pytest.raises(ValueError):
        switch_llm_env(env_file, api_key=api_key)

    assert env_file.read_bytes() == original
