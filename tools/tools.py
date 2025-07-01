"""
A collection of tools for performing file system operations and RAG.
"""
from pathlib import Path
from typing import List

from rag.vector_store_manager import VectorStoreManager


class FileTools:
    """Encapsulates all file system and RAG operations."""

    def __init__(self, base_path: str):
        """Initializes the tools, workspace, and vector store."""
        self.base_path = Path(base_path).resolve()
        self.base_path.mkdir(parents=True, exist_ok=True)

        index_path = self.base_path / "faiss_index"
        self.vector_store_manager = VectorStoreManager(index_path=str(index_path))

        # --- NUOVA LOGICA: Ricostruisci l'indice all'avvio ---
        self.vector_store_manager.rebuild_index_from_workspace(self.base_path)
        # ---------------------------------------------------

    # ... (il resto del file rimane identico) ...
    def _resolve(self, filename: str) -> Path:
        """Resolves a filename to a secure absolute path."""
        safe_filename = filename.strip().strip("'\" ")
        candidate = (self.base_path / safe_filename).resolve()
        if not str(candidate).startswith(str(self.base_path)):
            raise ValueError(f"Access denied: '{filename}' is outside of the workspace.")
        return candidate

    def list_files(self) -> List[str]:
        """Lists all files in the workspace (excluding the index folder)."""
        all_files = [
            str(p.relative_to(self.base_path)) for p in self.base_path.rglob("*") if p.is_file()
        ]
        if self.vector_store_manager and self.vector_store_manager.index_path.exists():
            index_dir_name = self.vector_store_manager.index_path.name
            return [f for f in all_files if not f.startswith(index_dir_name)]
        return all_files

    def read_file(self, filename: str) -> str:
        """Reads the content of a single file."""
        p = self._resolve(filename)
        if not p.exists():
            raise FileNotFoundError(f"The file '{filename}' does not exist.")
        return p.read_text(encoding="utf-8")

    def write_file(self, filename: str, content: str) -> str:
        """Writes content to a file and updates the vector index."""
        p = self._resolve(filename)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
        self.vector_store_manager.add_file(p)
        return f"Wrote {len(content)} characters to '{filename}' and updated the index."

    def delete_file(self, filename: str) -> str:
        """Deletes a file and removes it from the vector index."""
        p = self._resolve(filename)
        if not p.exists():
            raise FileNotFoundError(f"The file '{filename}' does not exist.")
        self.vector_store_manager.remove_file(filename)
        p.unlink()
        return f"Deleted '{filename}' and removed it from the index."

    def answer_question_about_files(self, query: str) -> str:
        """
        Esegue una ricerca semantica nell'indice vettoriale per trovare
        informazioni pertinenti alla domanda dell'utente.
        """
        return self.vector_store_manager.search(query)
    