"""Where the library looks for residues once it is packaged.

On the household shelf catalog.py sits beside the workdirs, so LIBRARY_DIR.parent
IS the shelf. Packaging moved the module into src/video_perceive/library/, where
the same expression names a directory containing no residues — `rebuild` would
then catalogue nothing and report success. Silent empty coverage reads as
coverage, so this pins the resolution order.
"""

import importlib

import pytest


def _reload(monkeypatch, **env):
    import video_perceive.library.catalog as c

    for k in ("VIDEO_PERCEIVE_SHELF", "VIDEO_PERCEIVE_LIBRARY"):
        monkeypatch.delenv(k, raising=False)
    for k, v in env.items():
        monkeypatch.setenv(k, str(v))
    return importlib.reload(c)


def test_env_var_wins(monkeypatch, tmp_path):
    c = _reload(monkeypatch, VIDEO_PERCEIVE_SHELF=tmp_path)
    assert c.DEFAULT_SHELF == tmp_path
    assert c.CATALOG_JSON.parent == tmp_path / "library"


def test_falls_back_to_cwd_when_package_parent_holds_no_residues(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    c = _reload(monkeypatch)
    assert c.DEFAULT_SHELF == tmp_path
    assert "video_perceive" not in str(c.DEFAULT_SHELF)


def test_package_parent_is_used_only_when_it_actually_holds_residues(monkeypatch, tmp_path):
    import video_perceive.library.catalog as c

    parent = tmp_path / "shelf"
    (parent / "demo-x" / "out").mkdir(parents=True)
    (parent / "demo-x" / "out" / "summary.json").write_text("{}")
    monkeypatch.setattr(c, "LIBRARY_DIR", parent / "library")
    monkeypatch.delenv("VIDEO_PERCEIVE_SHELF", raising=False)
    assert c._default_shelf() == parent


def test_library_store_is_overridable(monkeypatch, tmp_path):
    store = tmp_path / "elsewhere"
    c = _reload(monkeypatch, VIDEO_PERCEIVE_SHELF=tmp_path, VIDEO_PERCEIVE_LIBRARY=store)
    assert c.CATALOG_JSON.parent == store


@pytest.fixture(autouse=True)
def _restore(monkeypatch):
    yield
    import video_perceive.library.catalog as c

    for k in ("VIDEO_PERCEIVE_SHELF", "VIDEO_PERCEIVE_LIBRARY"):
        monkeypatch.delenv(k, raising=False)
    importlib.reload(c)
