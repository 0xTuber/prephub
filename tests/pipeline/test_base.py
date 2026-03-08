from course_builder.pipeline.base import PipelineContext, PipelineStep, Pipeline


class TestPipelineContext:
    def test_is_dict(self):
        ctx = PipelineContext()
        ctx["key"] = "value"
        assert ctx["key"] == "value"
        assert isinstance(ctx, dict)

    def test_initial_values(self):
        ctx = PipelineContext(a=1, b=2)
        assert ctx["a"] == 1
        assert ctx["b"] == 2


class AddStep(PipelineStep):
    def __init__(self, key: str, value):
        self.key = key
        self.value = value

    def run(self, context: PipelineContext) -> PipelineContext:
        context[self.key] = self.value
        return context


class DoubleStep(PipelineStep):
    """Reads 'number' from context, doubles it, stores as 'result'."""

    def run(self, context: PipelineContext) -> PipelineContext:
        context["result"] = context["number"] * 2
        return context


class TestPipeline:
    def test_empty_pipeline(self):
        p = Pipeline(steps=[])
        ctx = p.run()
        assert ctx == {}

    def test_single_step(self):
        p = Pipeline(steps=[AddStep("name", "AWS")])
        ctx = p.run()
        assert ctx["name"] == "AWS"

    def test_multiple_steps_accumulate(self):
        p = Pipeline(
            steps=[
                AddStep("a", 1),
                AddStep("b", 2),
                AddStep("c", 3),
            ]
        )
        ctx = p.run()
        assert ctx == {"a": 1, "b": 2, "c": 3}

    def test_steps_see_previous_context(self):
        p = Pipeline(
            steps=[
                AddStep("number", 5),
                DoubleStep(),
            ]
        )
        ctx = p.run()
        assert ctx["result"] == 10

    def test_run_with_existing_context(self):
        ctx = PipelineContext(existing="data")
        p = Pipeline(steps=[AddStep("new", "value")])
        result = p.run(ctx)
        assert result["existing"] == "data"
        assert result["new"] == "value"
