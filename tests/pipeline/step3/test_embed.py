from unittest.mock import patch, MagicMock

import chromadb
import numpy as np
import pytest

from pipeline.base import PipelineContext
from pipeline.models import ExtractedChunk
from pipeline.step3.embed import EmbeddingStep, _sanitize_collection_name


class TestSanitizeCollectionName:
    def test_basic(self):
        assert _sanitize_collection_name("AWS SAA-C03") == "AWS_SAA-C03"

    def test_short_name_padded(self):
        result = _sanitize_collection_name("ab")
        assert len(result) >= 3

    def test_long_name_truncated(self):
        long_name = "a" * 100
        result = _sanitize_collection_name(long_name)
        assert len(result) <= 63

    def test_alphanumeric_start_end(self):
        result = _sanitize_collection_name("__test__")
        assert result[0].isalnum()
        assert result[-1].isalnum()

    def test_special_chars_replaced(self):
        result = _sanitize_collection_name("hello world! @#$")
        assert all(c.isalnum() or c in "_-" for c in result)


class TestEmbeddingStep:
    def _make_chunks(self, n=3):
        return [
            ExtractedChunk(
                text=f"Sample text chunk {i}",
                section_heading=f"Section {i}",
                image_paths=[f"img{i}.png"],
                page_numbers=[i],
                book_title="Test Book",
                book_author="Test Author",
                source_file="test.pdf",
            )
            for i in range(n)
        ]

    def _make_mock_model(self, n):
        model = MagicMock()
        model.encode.return_value = np.random.rand(n, 384)
        return model

    def test_stores_chunks_in_chromadb(self):
        chunks = self._make_chunks(3)
        client = chromadb.Client()

        ctx = PipelineContext(
            certification_name="Test Cert",
            extracted_chunks=chunks,
        )

        step = EmbeddingStep(chroma_client=client)
        with patch.object(step, "_get_embedding_model", return_value=self._make_mock_model(3)):
            result = step.run(ctx)

        assert result["chunks_stored"] == 3
        assert result["collection_name"] == "Test_Cert"

        collection = client.get_collection("Test_Cert")
        assert collection.count() == 3

    def test_empty_chunks_noop(self):
        client = chromadb.Client()
        ctx = PipelineContext(
            certification_name="Test Cert",
            extracted_chunks=[],
        )

        step = EmbeddingStep(chroma_client=client)
        result = step.run(ctx)

        assert result["chunks_stored"] == 0
        assert result["collection_name"] == ""

    def test_metadata_stored_correctly(self):
        chunks = self._make_chunks(1)
        client = chromadb.Client()

        ctx = PipelineContext(
            certification_name="Meta Test",
            extracted_chunks=chunks,
        )

        step = EmbeddingStep(chroma_client=client)
        with patch.object(step, "_get_embedding_model", return_value=self._make_mock_model(1)):
            step.run(ctx)

        collection = client.get_collection("Meta_Test")
        results = collection.get(ids=["chunk_0"], include=["metadatas"])
        meta = results["metadatas"][0]

        assert meta["book_title"] == "Test Book"
        assert meta["book_author"] == "Test Author"
        assert meta["section_heading"] == "Section 0"
        assert meta["page_numbers"] == "0"
        assert meta["image_paths"] == "img0.png"
        assert meta["source_file"] == "test.pdf"

    def test_batch_upsert(self):
        chunks = self._make_chunks(10)
        client = chromadb.Client()

        ctx = PipelineContext(
            certification_name="Batch Test",
            extracted_chunks=chunks,
        )

        step = EmbeddingStep(chroma_client=client, batch_size=3)
        with patch.object(step, "_get_embedding_model", return_value=self._make_mock_model(10)):
            result = step.run(ctx)

        assert result["chunks_stored"] == 10
        collection = client.get_collection("Batch_Test")
        assert collection.count() == 10
