"""
Manages the vector store for the RAG pipeline.
"""
import shutil
from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


class VectorStoreManager:
    """Manages the creation and searching of the vector index."""

    def __init__(self, index_path: str):
        """Initializes the manager, text splitter, and embedding model."""
        self.index_path = Path(index_path)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200, length_function=len
        )
        self.embedding_model = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2", model_kwargs={'device': 'cpu'}
        )
        self.vector_store = self._load_or_create_index()

    def _load_or_create_index(self) -> FAISS:
        """Loads an existing FAISS index or creates a new one."""
        if self.index_path.exists():
            print("Loading existing FAISS index...")
            return FAISS.load_local(
                str(self.index_path),
                self.embedding_model,
                allow_dangerous_deserialization=True
            )
        print("Creating new FAISS index...")
        # Crea un indice vuoto se non ne esiste uno.
        empty_index = FAISS.from_texts([" "], self.embedding_model)
        empty_index.delete(list(empty_index.index_to_docstore_id.values()))
        return empty_index

    def rebuild_index_from_workspace(self, workspace_path: Path):
        """
        Deletes the old index and rebuilds it from all files in the workspace.
        """
        print(f"Rebuilding index from workspace: {workspace_path}")
        if self.index_path.exists():
            shutil.rmtree(self.index_path)

        self.vector_store = self._load_or_create_index()

        all_files = [p for p in workspace_path.rglob("*") if p.is_file()]

        for file_path in all_files:
            # Assicurati di non indicizzare l'indice stesso
            if "faiss_index" not in str(file_path):
                self.add_file(file_path, save=False)  # Aggiunge senza salvare ogni volta

        self.vector_store.save_local(str(self.index_path))
        print("Index rebuilt successfully.")


    def add_file(self, file_path: Path, save: bool = True):
        """Processes a file, chunks it, and adds it to the vector store."""
        print(f"Indexing file: {file_path.name}...")
        content = file_path.read_text(encoding="utf-8")
        doc = Document(page_content=content, metadata={"source": file_path.name})
        docs_to_index = self.text_splitter.split_documents([doc])
        self.vector_store.add_documents(docs_to_index)

        if save:
            self.vector_store.save_local(str(self.index_path))
        print(f"File {file_path.name} indexed.")

    def remove_file(self, filename: str):
        """Removes all vectors associated with a specific file from the store."""
        if not self.vector_store.index_to_docstore_id:
            return
        # pylint: disable=protected-access
        ids_to_delete = [
            doc_id for doc_id, doc in self.vector_store.docstore._dict.items()
            if doc.metadata.get("source") == filename
        ]
        if ids_to_delete:
            self.vector_store.delete(ids_to_delete)
            self.vector_store.save_local(str(self.index_path))
            print(f"Removed file {filename} from index.")

    def search(self, query: str) -> str:
        """Performs a similarity search and returns the formatted context."""
        print(f"Performing similarity search for query: '{query}'")
        results = self.vector_store.similarity_search(query, k=3)
        if not results:
            return "No relevant information found in the documents."
        context = "Relevant excerpts from documents found:\n\n"
        for doc in results:
            context += (
                f"--- Excerpt from: {doc.metadata.get('source', 'Unknown')} ---\n"
                f"{doc.page_content}\n\n"
            )
        return context
    