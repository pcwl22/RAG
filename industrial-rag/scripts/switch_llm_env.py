"""Atomically switch the OpenAI-compatible LLM settings in a dotenv file.

The API key is accepted only through a hidden prompt or an existing process
environment variable, so it never has to appear in shell history or command
arguments.  The command never prints the key.
"""

from __future__ import annotations

import argparse
import codecs
import getpass
import os
import re
import stat
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

LLM_ENV_KEYS = (
    "DEEPSEEK_API_KEY",
    "DEEPSEEK_API_URL",
    "DEEPSEEK_MODEL",
)
_ASSIGNMENT_PATTERN = re.compile(
    r"^\s*(?:export\s+)?(?P<key>"
    r"DEEPSEEK_API_KEY|DEEPSEEK_API_URL|DEEPSEEK_MODEL|"
    r"'DEEPSEEK_API_KEY'|'DEEPSEEK_API_URL'|'DEEPSEEK_MODEL'|"
    r"\"DEEPSEEK_API_KEY\"|\"DEEPSEEK_API_URL\"|\"DEEPSEEK_MODEL\""
    r")\s*=",
)
_PROFILE_ASSIGNMENT_PATTERN = re.compile(
    r"^\s*(?P<comment>#\s*)?(?:export\s+)?(?P<key>"
    r"DEEPSEEK_API_KEY|DEEPSEEK_API_URL|DEEPSEEK_MODEL|"
    r"'DEEPSEEK_API_KEY'|'DEEPSEEK_API_URL'|'DEEPSEEK_MODEL'|"
    r"\"DEEPSEEK_API_KEY\"|\"DEEPSEEK_API_URL\"|\"DEEPSEEK_MODEL\""
    r")\s*=\s*(?P<value>.*)$",
)
_ENV_NAME_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_MODEL_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/-]{0,199}$")
_SAFE_VALUE_PATTERN = re.compile(r"^[^\s#'\"`$\\\x00-\x1f\x7f]+$")
_PLACEHOLDER_PATTERN = re.compile(
    r"change[-_ ]?me|replace[-_ ]?with|xxxxx|example|your[-_ ]|<[^>]+>",
    re.IGNORECASE,
)
_LOCAL_HOSTS = frozenset({"localhost", "127.0.0.1", "::1"})


@dataclass(frozen=True)
class SwitchSummary:
    """Non-secret description of a planned or completed switch."""

    env_file: Path
    api_url: str | None
    model: str | None
    api_key_changed: bool
    dry_run: bool


@dataclass(frozen=True)
class LLMProfile:
    """One complete dotenv LLM block with a secret-safe representation."""

    index: int
    commented: bool
    models: tuple[str, ...]
    active_model: str | None
    api_url: str
    api_key: str = field(repr=False)
    entries: tuple[tuple[int, bool, str, str], ...] = field(repr=False)


def normalize_api_url(value: str, *, allow_local_http: bool = False) -> str:
    """Validate and normalize an OpenAI-compatible base URL."""
    raw = value.strip().rstrip("/")
    parsed = urlsplit(raw)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("API URL must be an absolute http:// or https:// URL")
    if parsed.username or parsed.password:
        raise ValueError("API URL must not contain credentials")
    if parsed.query or parsed.fragment:
        raise ValueError("API URL must not contain a query string or fragment")
    if parsed.scheme == "http" and not (
        allow_local_http and parsed.hostname in _LOCAL_HOSTS
    ):
        raise ValueError(
            "API URL must use https:// (or local HTTP with --allow-local-http)"
        )
    path = parsed.path.rstrip("/") or "/v1"
    normalized = urlunsplit((parsed.scheme, parsed.netloc, path, "", ""))
    _validate_dotenv_value("DEEPSEEK_API_URL", normalized)
    return normalized


def normalize_model(value: str) -> str:
    """Validate a provider model identifier without assuming one vendor."""
    model = value.strip()
    if not _MODEL_PATTERN.fullmatch(model):
        raise ValueError(
            "model must be 1-200 characters using letters, digits, '.', '_', ':', '/', or '-'"
        )
    return model


def validate_api_key(value: str) -> str:
    """Reject empty, placeholder, or dotenv-ambiguous API keys."""
    api_key = value.strip()
    if len(api_key) < 16:
        raise ValueError("API key must contain at least 16 characters")
    if _PLACEHOLDER_PATTERN.search(api_key):
        raise ValueError("API key must not be a placeholder value")
    _validate_dotenv_value("DEEPSEEK_API_KEY", api_key)
    return api_key


def _validate_dotenv_value(name: str, value: str) -> None:
    if not _SAFE_VALUE_PATTERN.fullmatch(value):
        raise ValueError(
            f"{name} contains whitespace or dotenv-interpreted characters"
        )


def _detect_newline(text: str) -> str:
    if "\r\n" in text:
        return "\r\n"
    return "\n"


def _dotenv_value(raw_value: str, *, line_number: int) -> str:
    value = raw_value.strip()
    starts_quoted = bool(value) and value[0] in {"'", '"'}
    ends_quoted = bool(value) and value[-1] in {"'", '"'}
    if starts_quoted or ends_quoted:
        if len(value) < 2 or value[0] != value[-1]:
            raise ValueError(f"unbalanced dotenv quotes at line {line_number}")
        value = value[1:-1]
    return value


def discover_llm_profiles(text: str) -> list[LLMProfile]:
    """Discover complete LLM blocks separated by blank dotenv lines."""
    raw_groups: list[list[tuple[int, bool, str, str]]] = []
    current: list[tuple[int, bool, str, str]] = []

    def finish_group() -> None:
        nonlocal current
        if current:
            raw_groups.append(current)
            current = []

    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            finish_group()
            continue
        match = _PROFILE_ASSIGNMENT_PATTERN.match(line)
        if not match:
            continue
        key = match.group("key").strip("'\"")
        value = _dotenv_value(match.group("value"), line_number=line_number)
        current.append((line_number - 1, bool(match.group("comment")), key, value))
    finish_group()

    profiles: list[LLMProfile] = []
    for index, group in enumerate(raw_groups, start=1):
        values: dict[str, list[tuple[str, bool, int]]] = {
            key: [] for key in LLM_ENV_KEYS
        }
        for line_index, commented, key, value in group:
            values[key].append((value, commented, line_index))
        for key in ("DEEPSEEK_API_KEY", "DEEPSEEK_API_URL"):
            if len(values[key]) != 1:
                raise ValueError(f"LLM profile {index} must contain exactly one {key}")
        if not values["DEEPSEEK_MODEL"]:
            raise ValueError(f"LLM profile {index} must contain at least one DEEPSEEK_MODEL")
        models = tuple(value for value, _, _ in values["DEEPSEEK_MODEL"])
        if len(set(models)) != len(models):
            raise ValueError(f"LLM profile {index} contains duplicate model names")
        key_commented = values["DEEPSEEK_API_KEY"][0][1]
        url_commented = values["DEEPSEEK_API_URL"][0][1]
        active_models = tuple(
            value for value, commented, _ in values["DEEPSEEK_MODEL"] if not commented
        )
        if key_commented and url_commented and not active_models:
            active_model = None
        elif not key_commented and not url_commented and len(active_models) == 1:
            active_model = active_models[0]
        else:
            raise ValueError(
                f"LLM profile {index} must be fully commented or expose exactly one active model"
            )
        profiles.append(
            LLMProfile(
                index=index,
                commented=active_model is None,
                models=models,
                active_model=active_model,
                api_url=values["DEEPSEEK_API_URL"][0][0],
                api_key=values["DEEPSEEK_API_KEY"][0][0],
                entries=tuple(group),
            )
        )
    return profiles


def _set_assignment_commented(line: str, *, commented: bool) -> str:
    line_ending = "\r\n" if line.endswith("\r\n") else "\n" if line.endswith("\n") else ""
    body = line[: -len(line_ending)] if line_ending else line
    if commented:
        match = re.match(r"^(?P<indent>\s*)(?P<body>.*)$", body)
        if match is None:
            raise ValueError("unable to comment dotenv assignment")
        return f"{match.group('indent')}#{match.group('body')}{line_ending}"
    match = re.match(r"^(?P<indent>\s*)#\s?(?P<body>.*)$", body)
    if match is None:
        raise ValueError("unable to uncomment dotenv assignment")
    return f"{match.group('indent')}{match.group('body')}{line_ending}"


def _replace_values(text: str, updates: dict[str, str]) -> str:
    lines = text.splitlines(keepends=True)
    seen: dict[str, int] = {}
    rendered: list[str] = []
    newline = _detect_newline(text)

    for line_number, line in enumerate(lines, start=1):
        match = _ASSIGNMENT_PATTERN.match(line)
        if not match:
            rendered.append(line)
            continue
        key = match.group("key").strip("'\"")
        if key in seen:
            raise ValueError(
                f"duplicate dotenv key {key!r} at lines {seen[key]} and {line_number}"
            )
        seen[key] = line_number
        if key not in updates:
            rendered.append(line)
            continue
        line_ending = "\r\n" if line.endswith("\r\n") else "\n" if line.endswith("\n") else ""
        rendered.append(f"{key}={updates[key]}{line_ending}")

    missing = [key for key in LLM_ENV_KEYS if key in updates and key not in seen]
    if missing:
        if rendered and not rendered[-1].endswith(("\n", "\r")):
            rendered[-1] += newline
        rendered.extend(f"{key}={updates[key]}{newline}" for key in missing)
    return "".join(rendered)


def _write_atomic(path: Path, *, original: bytes, replacement: bytes) -> None:
    file_mode = stat.S_IMODE(path.stat().st_mode)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f"{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(replacement)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary_path, file_mode)
        if path.read_bytes() != original:
            raise RuntimeError("dotenv file changed during the switch; no update was applied")
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def switch_llm_env(
    env_file: Path,
    *,
    api_url: str | None = None,
    model: str | None = None,
    api_key: str | None = None,
    allow_local_http: bool = False,
    dry_run: bool = False,
) -> SwitchSummary:
    """Validate and atomically apply one or more LLM dotenv settings."""
    path = env_file.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"dotenv file does not exist: {path}")
    if api_url is None and model is None and api_key is None:
        raise ValueError("at least one of API URL, model, or API key must be supplied")

    updates: dict[str, str] = {}
    normalized_url: str | None = None
    normalized_model: str | None = None
    if api_url is not None:
        normalized_url = normalize_api_url(api_url, allow_local_http=allow_local_http)
        updates["DEEPSEEK_API_URL"] = normalized_url
    if model is not None:
        normalized_model = normalize_model(model)
        updates["DEEPSEEK_MODEL"] = normalized_model
    if api_key is not None:
        updates["DEEPSEEK_API_KEY"] = validate_api_key(api_key)

    original = path.read_bytes()
    has_bom = original.startswith(codecs.BOM_UTF8)
    text = original.decode("utf-8-sig")
    replacement_text = _replace_values(text, updates)
    replacement = (codecs.BOM_UTF8 if has_bom else b"") + replacement_text.encode("utf-8")
    if not dry_run and replacement != original:
        _write_atomic(path, original=original, replacement=replacement)

    return SwitchSummary(
        env_file=path,
        api_url=normalized_url,
        model=normalized_model,
        api_key_changed=api_key is not None,
        dry_run=dry_run,
    )


def load_llm_profiles(env_file: Path) -> list[LLMProfile]:
    """Load profile metadata without exposing API keys in logs or repr output."""
    path = env_file.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"dotenv file does not exist: {path}")
    profiles = discover_llm_profiles(path.read_text(encoding="utf-8-sig"))
    if not profiles:
        raise ValueError("dotenv file does not contain any complete LLM profiles")
    return profiles


def activate_llm_profile(
    env_file: Path,
    *,
    profile_index: int,
    profile_model: str | None = None,
    allow_local_http: bool = False,
    dry_run: bool = False,
) -> SwitchSummary:
    """Atomically comment the old block and activate one declared profile."""
    path = env_file.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"dotenv file does not exist: {path}")
    original = path.read_bytes()
    has_bom = original.startswith(codecs.BOM_UTF8)
    text = original.decode("utf-8-sig")
    profiles = discover_llm_profiles(text)
    if not profiles:
        raise ValueError("dotenv file does not contain any complete LLM profiles")
    if profile_index < 1 or profile_index > len(profiles):
        raise ValueError(f"profile index must be between 1 and {len(profiles)}")
    active_profiles = [profile for profile in profiles if not profile.commented]
    if len(active_profiles) != 1:
        raise ValueError("dotenv file must contain exactly one active LLM profile")
    profile = profiles[profile_index - 1]
    if profile_model is None:
        if len(profile.models) != 1:
            raise ValueError(
                f"LLM profile {profile_index} has multiple models; use --profile-model"
            )
        selected_model = profile.models[0]
    else:
        selected_model = normalize_model(profile_model)
        if selected_model not in profile.models:
            raise ValueError(
                f"model {selected_model!r} is not declared in LLM profile {profile_index}"
            )
    normalized_url = normalize_api_url(
        profile.api_url,
        allow_local_http=allow_local_http,
    )
    normalized_model = normalize_model(selected_model)
    validate_api_key(profile.api_key)

    lines = text.splitlines(keepends=True)
    for candidate in profiles:
        for line_index, is_commented, key, value in candidate.entries:
            should_be_active = candidate.index == profile_index and (
                key != "DEEPSEEK_MODEL" or value == selected_model
            )
            if should_be_active == (not is_commented):
                continue
            lines[line_index] = _set_assignment_commented(
                lines[line_index],
                commented=not should_be_active,
            )
    replacement_text = "".join(lines)
    replacement_profiles = discover_llm_profiles(replacement_text)
    replacement_active = [
        candidate for candidate in replacement_profiles if not candidate.commented
    ]
    if (
        len(replacement_active) != 1
        or replacement_active[0].index != profile_index
        or replacement_active[0].active_model != selected_model
    ):
        raise RuntimeError("profile switch did not produce one exact active LLM profile")
    replacement = (codecs.BOM_UTF8 if has_bom else b"") + replacement_text.encode("utf-8")
    if not dry_run and replacement != original:
        _write_atomic(path, original=original, replacement=replacement)

    previous = active_profiles[0]
    return SwitchSummary(
        env_file=path,
        api_url=normalized_url,
        model=normalized_model,
        api_key_changed=(
            previous.index != profile_index or previous.api_key != profile.api_key
        ),
        dry_run=dry_run,
    )


def _read_api_key(args: argparse.Namespace) -> str | None:
    if args.prompt_api_key:
        return getpass.getpass("New DEEPSEEK_API_KEY (input hidden): ")
    if args.api_key_env:
        if not _ENV_NAME_PATTERN.fullmatch(args.api_key_env):
            raise ValueError("--api-key-env must be a valid environment variable name")
        value = os.environ.get(args.api_key_env)
        if value is None:
            raise ValueError(f"environment variable {args.api_key_env!r} is not set")
        return value
    return None


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument("--api-url")
    parser.add_argument("--model")
    parser.add_argument(
        "--list-profiles",
        action="store_true",
        help="list non-secret metadata for LLM blocks in the dotenv file",
    )
    parser.add_argument(
        "--activate-profile",
        type=int,
        metavar="N",
        help="copy dotenv LLM block N into the active configuration",
    )
    parser.add_argument(
        "--profile-model",
        help="select one model when the chosen profile declares multiple models",
    )
    key_source = parser.add_mutually_exclusive_group()
    key_source.add_argument(
        "--prompt-api-key",
        action="store_true",
        help="read the replacement API key from a hidden interactive prompt",
    )
    key_source.add_argument(
        "--api-key-env",
        metavar="NAME",
        help="read the replacement API key from process environment variable NAME",
    )
    parser.add_argument(
        "--allow-local-http",
        action="store_true",
        help="allow http:// only for localhost, 127.0.0.1, or ::1",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    try:
        profile_mode = args.list_profiles or args.activate_profile is not None
        manual_mode = bool(
            args.api_url or args.model or args.prompt_api_key or args.api_key_env
        )
        if profile_mode and manual_mode:
            raise ValueError("profile selection cannot be combined with manual LLM values")
        if args.list_profiles and args.activate_profile is not None:
            raise ValueError("--list-profiles cannot be combined with --activate-profile")
        if args.profile_model and args.activate_profile is None:
            raise ValueError("--profile-model requires --activate-profile")
        if args.list_profiles:
            profiles = load_llm_profiles(args.env_file)
            for profile in profiles:
                normalized_url = normalize_api_url(
                    profile.api_url,
                    allow_local_http=args.allow_local_http,
                )
                state = "commented" if profile.commented else "active"
                host = urlsplit(normalized_url).hostname or "<invalid>"
                print(
                    f"profile={profile.index} state={state} "
                    f"models={','.join(profile.models)} url_host={host} api_key=present"
                )
            return
        if args.activate_profile is not None:
            summary = activate_llm_profile(
                args.env_file,
                profile_index=args.activate_profile,
                profile_model=args.profile_model,
                allow_local_http=args.allow_local_http,
                dry_run=args.dry_run,
            )
        else:
            api_key = _read_api_key(args)
            summary = switch_llm_env(
                args.env_file,
                api_url=args.api_url,
                model=args.model,
                api_key=api_key,
                allow_local_http=args.allow_local_http,
                dry_run=args.dry_run,
            )
    except (FileNotFoundError, RuntimeError, UnicodeError, ValueError) as exc:
        parser.error(str(exc))

    action = "Validated" if summary.dry_run else "Updated"
    print(f"{action} LLM configuration: {summary.env_file}")
    if summary.api_url is not None:
        print(f"DEEPSEEK_API_URL={summary.api_url}")
    if summary.model is not None:
        print(f"DEEPSEEK_MODEL={summary.model}")
    print(
        "DEEPSEEK_API_KEY="
        + ("updated (value hidden)" if summary.api_key_changed else "preserved")
    )
    if not summary.dry_run:
        print("Restart the application/evaluation process to load the new configuration.")


if __name__ == "__main__":
    main()
