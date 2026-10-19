from pathlib import Path

import pytest

from app.env_file import DEFAULT_ENV_FILES, load_env_files

SECRET = "AIza-FAKE-secret-value"


def env_file(tmp_path: Path, text: str, name: str = ".env", bom: bool = False) -> Path:
    path = tmp_path / name
    path.write_text(text, encoding="utf-8-sig" if bom else "utf-8")
    return path


def test_loads_variables_and_returns_names_only(tmp_path: Path) -> None:
    path = env_file(tmp_path, f"GEMINI_API_KEY={SECRET}\nGEMINI_MODEL=gemini-test\n")
    environ: dict[str, str] = {}

    loaded = load_env_files([path], environ)

    assert loaded == ("GEMINI_API_KEY", "GEMINI_MODEL")
    assert environ == {"GEMINI_API_KEY": SECRET, "GEMINI_MODEL": "gemini-test"}


def test_existing_environment_wins(tmp_path: Path) -> None:
    path = env_file(tmp_path, "GEMINI_API_KEY=from-file\n")
    environ = {"GEMINI_API_KEY": "from-environment"}

    assert load_env_files([path], environ) == ()
    assert environ["GEMINI_API_KEY"] == "from-environment"


def test_earlier_file_wins(tmp_path: Path) -> None:
    backend = env_file(tmp_path, "GEMINI_MODEL=backend\n", "backend.env")
    root = env_file(tmp_path, "GEMINI_MODEL=root\nGEMINI_API_KEY=root-key\n", "root.env")
    environ: dict[str, str] = {}

    load_env_files([backend, root], environ)

    assert environ == {"GEMINI_MODEL": "backend", "GEMINI_API_KEY": "root-key"}


def test_missing_files_are_skipped(tmp_path: Path) -> None:
    environ: dict[str, str] = {}

    assert load_env_files([tmp_path / "nope.env"], environ) == ()
    assert environ == {}


@pytest.mark.parametrize(
    "line, value",
    [
        ('KEY="quoted value"', "quoted value"),
        ("KEY='single # not a comment'", "single # not a comment"),
        ("KEY=plain # trailing comment", "plain"),
        ("export KEY=exported", "exported"),
        ("  KEY  =  spaced  ", "spaced"),
        ("KEY=", ""),
        ("KEY=a=b", "a=b"),
    ],
)
def test_value_forms(tmp_path: Path, line: str, value: str) -> None:
    environ: dict[str, str] = {}

    load_env_files([env_file(tmp_path, f"# comment\n\n{line}\n")], environ)

    assert environ == {"KEY": value}


def test_notepad_byte_order_mark_is_ignored(tmp_path: Path) -> None:
    environ: dict[str, str] = {}

    load_env_files([env_file(tmp_path, f"GEMINI_API_KEY={SECRET}\n", bom=True)], environ)

    assert environ == {"GEMINI_API_KEY": SECRET}


def test_malformed_line_error_does_not_show_the_line(tmp_path: Path) -> None:
    path = env_file(tmp_path, f"GOOD=1\n{SECRET} without a name\n")

    with pytest.raises(ValueError) as info:
        load_env_files([path], {})

    assert "line 2" in str(info.value)
    assert SECRET not in str(info.value)


def test_default_files_are_backend_then_repository_root() -> None:
    backend_env, root_env = DEFAULT_ENV_FILES

    assert backend_env.parent.name == "backend"
    assert root_env.parent == backend_env.parent.parent
