"""Embedding step.

This module provides the EmbeddingStep which generates vector embeddings
for extracted text chunks and stores them in ChromaDB.
"""

import re
from pathlib import Path

from course_builder.domain.books import ExtractedChunk
from course_builder.pipeline.base import PipelineContext, PipelineStep


def _sanitize_collection_name(name: str) -> str:
    """Ensure a ChromaDB-compatible collection name.

    Rules: 3-63 chars, alphanumeric start/end, only alphanumeric,
    underscores, or hyphens in between.
    """
    sanitized = re.sub(r"[^a-zA-Z0-9_-]", "_", name)
    sanitized = sanitized.strip("_-")
    if not sanitized or not sanitized[0].isalnum():
        sanitized = "a" + sanitized
    if not sanitized[-1].isalnum():
        sanitized = sanitized + "a"
    if len(sanitized) < 3:
        sanitized = sanitized.ljust(3, "a")
    if len(sanitized) > 63:
        sanitized = sanitized[:63]
        if not sanitized[-1].isalnum():
            sanitized = sanitized.rstrip("_-") or "aaa"
    return sanitized


class EmbeddingStep(PipelineStep):
    """Pipeline step that generates embeddings and stores them in ChromaDB."""

    def __init__(
        self,
        vectorstore_dir: str = "vectorstore",
        model_name: str = "all-MiniLM-L6-v2",
        batch_size: int = 64,
        chroma_client=None,
    ):
        """Initialize the embedding step.

        Args:
            vectorstore_dir: Directory for ChromaDB persistence.
            model_name: Sentence transformer model name.
            batch_size: Batch size for embedding generation.
            chroma_client: Optional pre-configured ChromaDB client.
        """
        self.vectorstore_dir = vectorstore_dir
        self.model_name = model_name
        self.batch_size = batch_size
        self._chroma_client = chroma_client

    def _get_chroma_client(self):
        """Get or create the ChromaDB client."""
        if self._chroma_client is not None:
            return self._chroma_client
        import chromadb

        path = str(Path(self.vectorstore_dir))
        Path(path).mkdir(parents=True, exist_ok=True)
        return chromadb.PersistentClient(path=path)

    def _get_embedding_model(self):
        """Get the sentence transformer model."""
        from sentence_transformers import SentenceTransformer

        return SentenceTransformer(self.model_name)

    def run(self, context: PipelineContext) -> PipelineContext:
        """Generate embeddings and store in ChromaDB.

        Expects context["extracted_chunks"] and context["certification_name"].
        Populates context["collection_name"], context["vectorstore_path"],
        and context["chunks_stored"].
        """
        chunks: list[ExtractedChunk] = context["extracted_chunks"]
        certification_name: str = context["certification_name"]

        print(f"\nEmbedding {len(chunks)} chunks into vectorstore...")

        if not chunks:
            print("  No chunks to embed, skipping")
            context["collection_name"] = ""
            context["vectorstore_path"] = self.vectorstore_dir
            context["chunks_stored"] = 0
            return context

        collection_name = _sanitize_collection_name(certification_name)
        print(f"  Collection: {collection_name}")
        print(f"  Vectorstore: {self.vectorstore_dir}")

        client = self._get_chroma_client()
        collection = client.get_or_create_collection(name=collection_name)

        print(f"  Loading embedding model: {self.model_name}...")
        model = self._get_embedding_model()

        texts = [chunk.text for chunk in chunks]
        print(f"  Generating embeddings for {len(texts)} chunks...")
        embeddings = model.encode(texts, batch_size=self.batch_size, show_progress_bar=True)

        ids = [f"chunk_{i}" for i in range(len(chunks))]
        metadatas = [
            {
                "book_title": chunk.book_title,
                "book_author": chunk.book_author,
                "section_heading": chunk.section_heading or "",
                "page_numbers": ",".join(str(p) for p in chunk.page_numbers),
                "image_paths": "|||".join(chunk.image_paths),
                "source_file": chunk.source_file,
            }
            for chunk in chunks
        ]

        # Upsert in batches
        print(f"  Storing embeddings in ChromaDB...")
        for i in range(0, len(chunks), self.batch_size):
            end = min(i + self.batch_size, len(chunks))
            collection.upsert(
                ids=ids[i:end],
                documents=texts[i:end],
                embeddings=[emb.tolist() for emb in embeddings[i:end]],
                metadatas=metadatas[i:end],
            )

        print(f"  Stored {len(chunks)} chunks in collection '{collection_name}'")
        context["collection_name"] = collection_name
        context["vectorstore_path"] = self.vectorstore_dir
        context["chunks_stored"] = len(chunks)
        return context
