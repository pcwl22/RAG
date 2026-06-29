"""Import smoke tests for the app package."""
import importlib
import pkgutil


def test_app_modules_import_without_src_dependency():
    groups = [
        "app.utils",
        "app.parser",
        "app.embedding",
        "app.vectorstore",
        "app.retrieval",
        "app.llm",
        "app.service",
        "app.workers",
        "app.api",
    ]

    imported = []
    for group in groups:
        package = importlib.import_module(group)
        imported.append(group)
        if hasattr(package, "__path__"):
            for module in pkgutil.walk_packages(package.__path__, group + "."):
                importlib.import_module(module.name)
                imported.append(module.name)

    assert "app.service.chat_service" in imported
    assert "app.api.chat" in imported
