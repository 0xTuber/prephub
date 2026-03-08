from unittest.mock import patch

import pytest

from pipeline.base import PipelineContext
from pipeline.step1.input import InputStep


class TestInputStep:
    def test_stores_certification_name(self):
        step = InputStep()
        with patch("builtins.input", return_value="AWS SAA-C03"):
            ctx = step.run(PipelineContext())
        assert ctx["certification_name"] == "AWS SAA-C03"

    def test_strips_whitespace(self):
        step = InputStep()
        with patch("builtins.input", return_value="  AWS SAA-C03  "):
            ctx = step.run(PipelineContext())
        assert ctx["certification_name"] == "AWS SAA-C03"

    def test_empty_input_raises(self):
        step = InputStep()
        with patch("builtins.input", return_value=""):
            with pytest.raises(ValueError, match="cannot be empty"):
                step.run(PipelineContext())

    def test_whitespace_only_raises(self):
        step = InputStep()
        with patch("builtins.input", return_value="   "):
            with pytest.raises(ValueError, match="cannot be empty"):
                step.run(PipelineContext())
