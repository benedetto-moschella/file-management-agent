"""
Tests for the individual functions in the FileTools class.
"""
import sys
from pathlib import Path

import pytest

# Adds the project root to the path to allow importing 'tools'
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# pylint: disable=wrong-import-position
from tools.tools import FileTools


@pytest.fixture
def tmp_workspace(tmp_path: Path) -> str:
    """Creates a temporary and empty workspace for each test."""
    ws = tmp_path / "workspace"
    ws.mkdir()
    return str(ws)


def test_list_on_empty(tmp_workspace: str):
    """Tests that list_files returns an empty list for an empty directory."""
    # pylint: disable=redefined-outer-name
    ft = FileTools(tmp_workspace)
    assert ft.list_files() == []


def test_write_and_read(tmp_workspace: str):
    """Tests writing to and then reading from a file."""
    # pylint: disable=redefined-outer-name
    ft = FileTools(tmp_workspace)
    msg = ft.write_file("foo.txt", "hello")
    assert "Wrote" in msg and "foo.txt" in msg
    content = ft.read_file("foo.txt")
    assert content == "hello"


def test_delete(tmp_workspace: str):
    """Tests deleting a file."""
    # pylint: disable=redefined-outer-name
    ft = FileTools(tmp_workspace)
    ft.write_file("bar.txt", "data")
    assert "bar.txt" in ft.list_files()
    msg = ft.delete_file("bar.txt")
    assert "Deleted" in msg
    assert "bar.txt" not in ft.list_files()


def test_read_nonexistent(tmp_workspace: str):
    """Tests that reading a non-existent file raises FileNotFoundError."""
    # pylint: disable=redefined-outer-name
    ft = FileTools(tmp_workspace)
    with pytest.raises(FileNotFoundError):
        ft.read_file("nonexistent.txt")


def test_path_traversal(tmp_workspace: str):
    """Tests that path traversal attempts raise a ValueError."""
    # pylint: disable=redefined-outer-name
    ft = FileTools(tmp_workspace)
    with pytest.raises(ValueError, match="Access denied"):
        ft.read_file("../outside.txt")


def test_answer_no_files(tmp_workspace: str):
    """Tests the response of the analysis tool when no files are present."""
    # pylint: disable=redefined-outer-name
    ft = FileTools(tmp_workspace)
    response = ft.answer_question_about_files("Any files?")
    assert "There are no files" in response
