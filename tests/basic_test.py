import importlib
import importlib.metadata as metadata
import builtins
import io

def test_version_fallback(monkeypatch):
    # Force PackageNotFoundError
    monkeypatch.setattr(metadata, "version", lambda _: (_ for _ in ()).throw(metadata.PackageNotFoundError))

    # Fake pyproject.toml content
    fake_toml = b"[project]\nversion = '0.1.0'\n"
    monkeypatch.setattr(builtins, "open", lambda *_: io.BytesIO(fake_toml))

    # Reload the module so the fallback branch executes
    import quantium
    importlib.reload(quantium)

    assert quantium.__version__ == "0.1.0"
