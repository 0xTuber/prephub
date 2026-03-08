"""Tests for checkpoint utilities."""

import tempfile
from pathlib import Path

import pytest

from course_builder.domain.course import CourseSkeleton, CourseOverview
from course_builder.pipeline.checkpoint import (
    CHECKPOINT_STAGES,
    get_checkpoint_filename,
    get_stage_index,
    get_stages_after,
    get_stages_up_to,
    save_checkpoint,
    load_checkpoint,
    list_checkpoints,
)


@pytest.fixture
def sample_skeleton():
    """Create a minimal skeleton for testing."""
    return CourseSkeleton(
        certification_name="Test Certification",
        exam_code="TEST-001",
        overview=CourseOverview(
            title="Test Course",
            description="A test course",
            target_audience="Testers",
            total_domains=1,
            estimated_study_hours=10,
        ),
        domain_modules=[],
        version=1,
    )


class TestCheckpointStages:
    """Tests for checkpoint stage utilities."""

    def test_stages_in_order(self):
        assert CHECKPOINT_STAGES == [
            "exam", "course", "labs", "capsules", "items", "content", "validated"
        ]

    def test_get_stage_index(self):
        assert get_stage_index("exam") == 0
        assert get_stage_index("labs") == 2
        assert get_stage_index("validated") == 6

    def test_get_stage_index_invalid(self):
        with pytest.raises(ValueError) as exc_info:
            get_stage_index("invalid")
        assert "Invalid stage" in str(exc_info.value)

    def test_get_stages_after(self):
        assert get_stages_after("labs") == ["capsules", "items", "content", "validated"]
        assert get_stages_after("validated") == []

    def test_get_stages_up_to(self):
        assert get_stages_up_to("labs") == ["exam", "course", "labs"]
        assert get_stages_up_to("exam") == ["exam"]


class TestCheckpointFilename:
    """Tests for checkpoint filename generation."""

    def test_basic_filename(self):
        filename = get_checkpoint_filename("Test Cert", "labs", 1)
        assert filename == "Test_Cert_v1_labs.json"

    def test_filename_with_special_chars(self):
        filename = get_checkpoint_filename("NREMT EMR (Emergency)", "course", 2)
        assert filename == "NREMT_EMR_Emergency_v2_course.json"


class TestSaveLoadCheckpoint:
    """Tests for save/load functionality."""

    def test_save_and_load(self, sample_skeleton):
        with tempfile.TemporaryDirectory() as tmpdir:
            # Save checkpoint
            filepath = save_checkpoint(
                sample_skeleton,
                "labs",
                tmpdir,
                engine="gemini",
                model="gemini-2.0-flash",
            )

            assert filepath.exists()
            assert "labs" in filepath.name

            # Load checkpoint
            loaded = load_checkpoint(filepath)

            assert loaded.certification_name == "Test Certification"
            assert loaded.checkpoint_stage == "labs"
            assert loaded.checkpoint_engine == "gemini"
            assert loaded.checkpoint_model == "gemini-2.0-flash"

    def test_save_invalid_stage(self, sample_skeleton):
        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(ValueError) as exc_info:
                save_checkpoint(sample_skeleton, "invalid", tmpdir)
            assert "Invalid stage" in str(exc_info.value)

    def test_load_nonexistent(self):
        with pytest.raises(FileNotFoundError):
            load_checkpoint("/nonexistent/path.json")


class TestListCheckpoints:
    """Tests for listing checkpoints."""

    def test_list_empty_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoints = list_checkpoints(tmpdir)
            assert checkpoints == []

    def test_list_nonexistent_dir(self):
        checkpoints = list_checkpoints("/nonexistent/path")
        assert checkpoints == []

    def test_list_with_checkpoints(self, sample_skeleton):
        with tempfile.TemporaryDirectory() as tmpdir:
            # Save multiple checkpoints
            save_checkpoint(sample_skeleton, "exam", tmpdir)
            sample_skeleton.version = 2
            save_checkpoint(sample_skeleton, "labs", tmpdir)

            checkpoints = list_checkpoints(tmpdir)

            assert len(checkpoints) == 2
            # Check that both stages are present
            stages = {cp["stage"] for cp in checkpoints}
            assert stages == {"exam", "labs"}
