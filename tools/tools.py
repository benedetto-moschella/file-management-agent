"""
A collection of tools for performing file system operations within a secure workspace.
"""
from pathlib import Path
from typing import List


class FileTools:
    """Encapsulates all file system operations."""

    def __init__(self, base_path: str):
        """Initializes the tools with a safe base directory."""
        self.base_path = Path(base_path).resolve()
        self.base_path.mkdir(parents=True, exist_ok=True)

    def _resolve(self, filename: str) -> Path:
        """Resolves a filename to a secure absolute path, preventing directory traversal."""
        safe_filename = filename.strip().strip("'\" ")
        candidate = (self.base_path / safe_filename).resolve()
        if not str(candidate).startswith(str(self.base_path)):
            raise ValueError(f"Access denied: '{filename}' is outside of the workspace.")
        return candidate

    def list_files(self) -> List[str]:
        """Lists all files in the workspace."""
        results = [
            str(p.relative_to(self.base_path)) for p in self.base_path.rglob("*") if p.is_file()
        ]
        return results

    def read_file(self, filename: str) -> str:
        """Reads the content of a single file."""
        p = self._resolve(filename)
        if not p.exists():
            raise FileNotFoundError(f"The file '{filename}' does not exist.")
        return p.read_text(encoding="utf-8")

    def write_file(self, filename: str, content: str) -> str:
        """Writes (or overwrites) content to a file."""
        p = self._resolve(filename)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
        return f"Wrote {len(content)} characters to '{filename}'."

    def delete_file(self, filename: str) -> str:
        """Deletes a file."""
        p = self._resolve(filename)
        if not p.exists():
            raise FileNotFoundError(f"The file '{filename}' does not exist.")
        p.unlink()
        return f"Deleted '{filename}'."

    def answer_question_about_files(self, query: str) -> str:
        """Gathers content from all files to provide context to the agent."""
        files = self.list_files()
        if not files:
            return "There are no files in the workspace to analyze."

        full_context = f"Context extracted from files to answer the query: '{query}'\n\n"
        for rel_path in files:
            content = self.read_file(rel_path)
            full_context += (
                f"--- START OF FILE: {rel_path} ---\n{content}\n"
                f"--- END OF FILE: {rel_path} ---\n\n"
            )
        return full_context
    