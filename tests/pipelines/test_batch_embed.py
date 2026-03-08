"""Tests for batch_embed pipeline module."""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from course_builder.pipelines.batch_embed import (
    build_embedding_plan,
    find_mineru_outputs,
    rechunk_from_mineru_output,
    run_batch_embed,
)


class TestFindMineruOutputs:
    def test_finds_direct_structure(self, tmp_path):
        """Test finding outputs in direct structure: folder/book_name/book_name.md"""
        book_dir = tmp_path / "My Book"
        book_dir.mkdir()
        md_file = book_dir / "My Book.md"
        md_file.write_text("# Chapter 1\nContent here")
        content_list = book_dir / "My Book_content_list.json"
        content_list.write_text("[]")

        results = find_mineru_outputs(tmp_path)

        assert len(results) == 1
        assert results[0][0] == md_file
        assert results[0][1] == content_list
        assert results[0][2] == "My Book"

    def test_finds_nested_structure(self, tmp_path):
        """Test finding outputs in nested structure: folder/cert/book/book.md"""
        cert_dir = tmp_path / "NREMT_EMT"
        cert_dir.mkdir()
        book_dir = cert_dir / "Emergency Care"
        book_dir.mkdir()
        md_file = book_dir / "Emergency Care.md"
        md_file.write_text("# Chapter 1")
        content_list = book_dir / "Emergency Care_content_list.json"
        content_list.write_text("[]")

        results = find_mineru_outputs(tmp_path)

        assert len(results) == 1
        assert results[0][0] == md_file
        assert results[0][2] == "Emergency Care"

    def test_finds_multiple_books(self, tmp_path):
        """Test finding multiple books in one folder."""
        for i in range(3):
            book_dir = tmp_path / f"Book {i}"
            book_dir.mkdir()
            (book_dir / f"Book {i}.md").write_text(f"# Book {i}")
            (book_dir / f"Book {i}_content_list.json").write_text("[]")

        results = find_mineru_outputs(tmp_path)

        assert len(results) == 3

    def test_handles_missing_content_list(self, tmp_path):
        """Test handling when content_list.json is missing."""
        book_dir = tmp_path / "No Content List"
        book_dir.mkdir()
        md_file = book_dir / "No Content List.md"
        md_file.write_text("# Chapter 1")

        results = find_mineru_outputs(tmp_path)

        assert len(results) == 1
        assert results[0][0] == md_file
        assert results[0][1] is None

    def test_empty_directory(self, tmp_path):
        """Test empty directory returns empty list."""
        results = find_mineru_outputs(tmp_path)
        assert results == []


class TestBuildEmbeddingPlan:
    def test_regular_folders_use_folder_name(self, tmp_path):
        """Test that regular folders map to sanitized folder name."""
        # Create folder with MinerU output
        folder = tmp_path / "NREMT EMT"
        folder.mkdir()
        book_dir = folder / "My Book"
        book_dir.mkdir()
        (book_dir / "My Book.md").write_text("# Content")
        (book_dir / "My Book_content_list.json").write_text("[]")

        plan = build_embedding_plan(tmp_path)

        assert "NREMT_EMT" in plan
        assert len(plan["NREMT_EMT"]) == 1

    def test_shared_folder_maps_to_multiple_collections(self, tmp_path):
        """Test shared folders map to multiple collections."""
        # Create shared folder
        shared = tmp_path / "paramedic and emt"
        shared.mkdir()
        book_dir = shared / "Shared Book"
        book_dir.mkdir()
        (book_dir / "Shared Book.md").write_text("# Content")
        (book_dir / "Shared Book_content_list.json").write_text("[]")

        shared_mappings = {"paramedic and emt": ["NREMT_EMT", "NREMT_Paramedic"]}
        plan = build_embedding_plan(tmp_path, shared_mappings)

        assert "NREMT_EMT" in plan
        assert "NREMT_Paramedic" in plan
        assert len(plan["NREMT_EMT"]) == 1
        assert len(plan["NREMT_Paramedic"]) == 1

    def test_mixed_regular_and_shared_folders(self, tmp_path):
        """Test mix of regular and shared folders."""
        # Regular folder
        regular = tmp_path / "NREMT_AEMT"
        regular.mkdir()
        book1_dir = regular / "AEMT Book"
        book1_dir.mkdir()
        (book1_dir / "AEMT Book.md").write_text("# AEMT")
        (book1_dir / "AEMT Book_content_list.json").write_text("[]")

        # Shared folder
        shared = tmp_path / "shared content"
        shared.mkdir()
        book2_dir = shared / "Shared Book"
        book2_dir.mkdir()
        (book2_dir / "Shared Book.md").write_text("# Shared")
        (book2_dir / "Shared Book_content_list.json").write_text("[]")

        shared_mappings = {"shared content": ["NREMT_EMT", "NREMT_Paramedic"]}
        plan = build_embedding_plan(tmp_path, shared_mappings)

        assert "NREMT_AEMT" in plan
        assert "NREMT_EMT" in plan
        assert "NREMT_Paramedic" in plan
        assert len(plan) == 3

    def test_empty_folders_skipped(self, tmp_path):
        """Test folders without MinerU outputs are skipped."""
        empty_folder = tmp_path / "empty"
        empty_folder.mkdir()

        plan = build_embedding_plan(tmp_path)

        assert plan == {}


class TestRechunkFromMineruOutput:
    def test_basic_chunking(self, tmp_path):
        """Test basic markdown chunking."""
        md_file = tmp_path / "test.md"
        md_file.write_text("# Introduction\nSome intro text\n\n# Chapter 1\nChapter content")
        content_list = tmp_path / "test_content_list.json"
        content_list.write_text("[]")

        chunks = rechunk_from_mineru_output(md_file, content_list, "Test Book")

        assert len(chunks) == 2
        assert chunks[0].section_heading == "Introduction"
        assert chunks[1].section_heading == "Chapter 1"
        assert all(c.book_title == "Test Book" for c in chunks)

    def test_handles_missing_content_list(self, tmp_path):
        """Test chunking works without content_list.json."""
        md_file = tmp_path / "test.md"
        md_file.write_text("# Only Chapter\nText content")

        chunks = rechunk_from_mineru_output(md_file, None, "Test Book")

        assert len(chunks) == 1
        assert chunks[0].section_heading == "Only Chapter"


class TestRunBatchEmbed:
    def test_dry_run_does_not_embed(self, tmp_path, capsys):
        """Test dry run shows plan but doesn't embed."""
        # Create folder with MinerU output
        folder = tmp_path / "NREMT_EMR"
        folder.mkdir()
        book_dir = folder / "EMR Book"
        book_dir.mkdir()
        (book_dir / "EMR Book.md").write_text("# EMR Content")
        (book_dir / "EMR Book_content_list.json").write_text("[]")

        with patch("course_builder.pipelines.batch_embed.EmbeddingStep") as mock_embed:
            run_batch_embed(
                extracted_dir=str(tmp_path),
                output_dir=str(tmp_path / "vectorstore"),
                dry_run=True,
            )

            mock_embed.assert_not_called()

        captured = capsys.readouterr()
        assert "DRY RUN" in captured.out
        assert "NREMT_EMR" in captured.out

    def test_delete_existing_removes_collections(self, tmp_path):
        """Test delete_existing removes collections before embedding."""
        # Create folder with MinerU output
        folder = tmp_path / "NREMT_EMR"
        folder.mkdir()
        book_dir = folder / "EMR Book"
        book_dir.mkdir()
        (book_dir / "EMR Book.md").write_text("# EMR Content")
        (book_dir / "EMR Book_content_list.json").write_text("[]")

        mock_client = MagicMock()

        with patch("chromadb.PersistentClient", return_value=mock_client):
            with patch("course_builder.pipelines.batch_embed.EmbeddingStep") as mock_embed:
                mock_embed_instance = MagicMock()
                mock_embed_instance.run.return_value = {"chunks_stored": 1}
                mock_embed.return_value = mock_embed_instance

                run_batch_embed(
                    extracted_dir=str(tmp_path),
                    output_dir=str(tmp_path / "vectorstore"),
                    delete_existing=True,
                )

                mock_client.delete_collection.assert_called_with("NREMT_EMR")

    def test_shared_mapping_embeds_to_multiple_collections(self, tmp_path, capsys):
        """Test shared folder embeds to multiple collections."""
        # Create shared folder
        shared = tmp_path / "paramedic and emt"
        shared.mkdir()
        book_dir = shared / "Shared Book"
        book_dir.mkdir()
        (book_dir / "Shared Book.md").write_text("# Shared Content\nSome actual text content here")
        (book_dir / "Shared Book_content_list.json").write_text("[]")

        with patch("course_builder.pipelines.batch_embed.EmbeddingStep") as mock_embed:
            mock_embed_instance = MagicMock()
            mock_embed_instance.run.return_value = {"chunks_stored": 1}
            mock_embed.return_value = mock_embed_instance

            run_batch_embed(
                extracted_dir=str(tmp_path),
                output_dir=str(tmp_path / "vectorstore"),
                shared_mappings={"paramedic and emt": ["NREMT_EMT", "NREMT_Paramedic"]},
            )

            # Should be called twice - once for each collection
            assert mock_embed_instance.run.call_count == 2

        captured = capsys.readouterr()
        assert "NREMT_EMT" in captured.out
        assert "NREMT_Paramedic" in captured.out

    def test_nonexistent_directory_shows_error(self, tmp_path, capsys):
        """Test error message for nonexistent directory."""
        run_batch_embed(
            extracted_dir=str(tmp_path / "nonexistent"),
            output_dir=str(tmp_path / "vectorstore"),
        )

        captured = capsys.readouterr()
        assert "ERROR" in captured.out
        assert "not found" in captured.out
