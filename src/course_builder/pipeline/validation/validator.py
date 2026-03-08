"""Hierarchical Validation Step.

Validates the skeleton top-down through the hierarchy using a funnel approach:
if a parent fails critically, skip validating its children.
"""

import json
import re
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

from course_builder.domain.course import CourseSkeleton
from course_builder.domain.validation import (
    ValidationIssue,
    ValidationReport,
    ValidationResult,
)
from course_builder.pipeline.base import EngineAwareStep, PipelineContext
from course_builder.pipeline.validation.rules import (
    GROUNDING_CHECK_PROMPT,
    QUALITY_REVIEW_PROMPT,
    Severity,
    get_structural_rules,
    get_worst_severity,
)

if TYPE_CHECKING:
    from course_builder.engine import GenerationEngine


def _sanitize_collection_name(name: str) -> str:
    """Ensure a ChromaDB-compatible collection name."""
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


def _strip_code_fences(text: str) -> str:
    """Remove markdown code fences from text."""
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines)
    return text


def _call_with_retry(fn, max_retries=3, base_delay=2.0):
    """Call function with exponential backoff retry."""
    last_error = None
    for attempt in range(max_retries):
        try:
            return fn()
        except Exception as e:
            last_error = e
            if attempt < max_retries - 1:
                time.sleep(base_delay * (2**attempt))
    raise last_error


def _generate_issue_id() -> str:
    """Generate a unique issue ID."""
    return f"issue_{uuid.uuid4().hex[:8]}"


def _validate_entity_structural(
    entity,
    entity_type: str,
    entity_id: str,
    entity_path: list[str],
) -> ValidationResult:
    """Run structural validation rules on an entity."""
    rules = get_structural_rules(entity_type)
    issues = []
    severities = [Severity.PASSED]

    for rule in rules:
        passed, description, suggested_fix = rule.check_fn(entity)
        if not passed:
            issues.append(
                ValidationIssue(
                    issue_id=_generate_issue_id(),
                    severity=rule.severity_if_failed.value,
                    rule_name=rule.name,
                    description=description or rule.description,
                    field_path=rule.name.split("_")[0] if "_" in rule.name else "",
                    suggested_fix=suggested_fix,
                )
            )
            severities.append(rule.severity_if_failed)

    return ValidationResult(
        entity_type=entity_type,
        entity_id=entity_id,
        entity_path=entity_path,
        overall_status=get_worst_severity(severities).value,
        issues=issues,
        validated_at=datetime.now(),
    )


def _validate_item_grounding(
    item,
    collection,
    embedding_model,
    engine: "GenerationEngine",
    cert_name: str,
    domain_name: str,
    topic_name: str,
    entity_path: list[str],
) -> list[ValidationIssue]:
    """Validate item grounding against source chunks using RAG."""
    from course_builder.engine import GenerationConfig

    issues = []

    if not item.source_reference or not item.source_reference.chunk_ids:
        return issues  # No source to validate against

    if not item.content or not item.options:
        return issues  # No content to validate

    # Retrieve the source chunks
    try:
        chunk_ids = item.source_reference.chunk_ids
        results = collection.get(ids=chunk_ids, include=["documents"])
        source_chunks = "\n\n---\n\n".join(results["documents"]) if results["documents"] else ""
    except Exception:
        source_chunks = ""

    if not source_chunks:
        issues.append(
            ValidationIssue(
                issue_id=_generate_issue_id(),
                severity=Severity.MINOR.value,
                rule_name="source_chunks_retrievable",
                description="Could not retrieve source chunks for validation",
                field_path="source_reference.chunk_ids",
            )
        )
        return issues

    # Get correct answer text
    correct_answer = item.options[item.correct_answer_index] if item.correct_answer_index is not None else ""

    prompt = GROUNDING_CHECK_PROMPT.format(
        content=item.content,
        correct_answer=correct_answer,
        explanation=item.explanation or "",
        source_chunks=source_chunks,
    )

    config = GenerationConfig(
        system_prompt="You are a fact-checker verifying educational content against source material.",
    )

    def _call():
        return engine.generate(prompt, config=config)

    try:
        result = _call_with_retry(_call)
        raw_json = _strip_code_fences(result.text)
        data = json.loads(raw_json)

        if not data.get("is_grounded", True):
            issues.append(
                ValidationIssue(
                    issue_id=_generate_issue_id(),
                    severity=Severity.MAJOR.value,
                    rule_name="answer_grounded",
                    description=data.get("issues", "Answer not found in source material"),
                    field_path="correct_answer_index",
                    source_evidence=data.get("evidence"),
                    suggested_fix="Regenerate question with better source alignment",
                )
            )
    except Exception as e:
        # Grounding check failed, log but don't block
        issues.append(
            ValidationIssue(
                issue_id=_generate_issue_id(),
                severity=Severity.MINOR.value,
                rule_name="grounding_check_error",
                description=f"Grounding check failed: {str(e)}",
                field_path="content",
            )
        )

    return issues


def _validate_item_quality(
    item,
    collection,
    engine: "GenerationEngine",
    cert_name: str,
    domain_name: str,
    topic_name: str,
    entity_path: list[str],
) -> list[ValidationIssue]:
    """Validate item quality using LLM review."""
    from course_builder.engine import GenerationConfig

    issues = []

    if not item.content or not item.options:
        return issues  # No content to validate

    # Format options for prompt
    options_formatted = "\n".join(
        f"{chr(65 + i)}. {opt}" for i, opt in enumerate(item.options)
    )
    correct_answer = item.options[item.correct_answer_index] if item.correct_answer_index is not None else ""

    # Retrieve source chunks if available
    source_chunks = ""
    if item.source_reference and item.source_reference.chunk_ids:
        try:
            chunk_ids = item.source_reference.chunk_ids
            results = collection.get(ids=chunk_ids, include=["documents"])
            source_chunks = "\n\n---\n\n".join(results["documents"]) if results["documents"] else ""
        except Exception:
            source_chunks = "(Source material not available)"

    prompt = QUALITY_REVIEW_PROMPT.format(
        certification=cert_name,
        domain_name=domain_name,
        topic_name=topic_name,
        learning_target=item.learning_target or "",
        content=item.content,
        options_formatted=options_formatted,
        correct_answer=correct_answer,
        explanation=item.explanation or "",
        source_chunks=source_chunks or "(No source material provided)",
    )

    config = GenerationConfig(
        system_prompt="You are an expert exam question reviewer.",
    )

    def _call():
        return engine.generate(prompt, config=config)

    try:
        result = _call_with_retry(_call)
        raw_json = _strip_code_fences(result.text)
        data = json.loads(raw_json)

        for issue_data in data.get("issues", []):
            severity_str = issue_data.get("severity", "minor").lower()
            if severity_str not in ["passed", "minor", "major", "critical"]:
                severity_str = "minor"

            issues.append(
                ValidationIssue(
                    issue_id=_generate_issue_id(),
                    severity=severity_str,
                    rule_name=issue_data.get("rule_name", "quality_review"),
                    description=issue_data.get("description", "Quality issue found"),
                    field_path="content",
                    suggested_fix=issue_data.get("suggested_fix"),
                )
            )
    except Exception as e:
        # Quality check failed, log but don't block
        issues.append(
            ValidationIssue(
                issue_id=_generate_issue_id(),
                severity=Severity.MINOR.value,
                rule_name="quality_check_error",
                description=f"Quality check failed: {str(e)}",
                field_path="content",
            )
        )

    return issues


class HierarchicalValidationStep(EngineAwareStep):
    """Validates the skeleton top-down through the hierarchy.

    Uses a funnel approach: if parent fails critically, skip children.
    Runs structural validation first (fast), then grounding/quality (LLM).

    Standard pattern for initialization (keyword-only parameters):
        step = HierarchicalValidationStep(
            engine=provider.validation_engine,
            max_workers=10,
        )

    For backward compatibility, you can still use the legacy model parameter:
        step = HierarchicalValidationStep(model="gemini-pro")
    """

    def __init__(
        self,
        *,  # Force keyword-only arguments
        engine: "GenerationEngine | None" = None,
        model: str | None = None,  # Legacy parameter for backward compatibility
        max_workers: int = 8,
        skip_llm_review: bool = False,  # For fast structural-only validation
        vectorstore_dir: str = "vectorstore",
        embedding_model_name: str = "all-MiniLM-L6-v2",
        output_dir: str = "corrections",
    ):
        """Initialize the validation step.

        Args:
            engine: Generation engine to use (preferred).
            model: Legacy parameter - model name for Gemini API.
                   Ignored if engine is provided.
            max_workers: Maximum concurrent validation workers.
            skip_llm_review: If True, skip LLM-based validation (structural only).
            vectorstore_dir: Directory containing the vectorstore.
            embedding_model_name: Sentence transformer model for embeddings.
            output_dir: Directory for validation reports.
        """
        super().__init__(engine=engine)
        self._legacy_model = model or "gemini-pro"
        self.max_workers = max_workers
        self.skip_llm_review = skip_llm_review
        self.vectorstore_dir = vectorstore_dir
        self.embedding_model_name = embedding_model_name
        self.output_dir = output_dir

    def run(self, context: PipelineContext) -> PipelineContext:
        skeleton: CourseSkeleton = context["course_skeleton"]
        cert_name = skeleton.certification_name

        print(f"Validating skeleton for '{cert_name}'...")

        # Get or create engine
        engine = self.get_engine()
        if engine is None and not self.skip_llm_review:
            # Legacy fallback: create Gemini engine
            from course_builder.engine import create_engine

            engine = create_engine("gemini", model=self._legacy_model)
            print(f"  Using model: {engine.model_name} (legacy mode)")
        elif engine is not None:
            print(f"  Using engine: {engine.engine_type}, model: {engine.model_name}")

        # Initialize vectorstore and embedding model if needed for grounding
        collection = None
        embedding_model = None

        if not self.skip_llm_review:
            collection_name = context.get("collection_name")
            if not collection_name:
                collection_name = _sanitize_collection_name(cert_name)

            vectorstore_path = context.get("vectorstore_path", self.vectorstore_dir)

            try:
                import chromadb
                from sentence_transformers import SentenceTransformer

                chroma_client = chromadb.PersistentClient(path=str(Path(vectorstore_path)))
                collection = chroma_client.get_collection(name=collection_name)
                embedding_model = SentenceTransformer(self.embedding_model_name)
                print(f"  Loaded vectorstore for grounding validation")
            except Exception as e:
                print(f"  WARNING: Could not load vectorstore: {e}")
                print(f"  Proceeding with structural validation only")

        # Initialize counters
        results: list[ValidationResult] = []
        total_entities = 0
        counts = {
            "passed": 0,
            "minor": 0,
            "major": 0,
            "critical": 0,
        }

        # Level 1: Validate skeleton
        print(f"  Validating skeleton level...")
        skeleton_result = _validate_entity_structural(
            skeleton, "skeleton", "skeleton", []
        )
        results.append(skeleton_result)
        total_entities += 1
        counts[skeleton_result.overall_status] += 1

        if skeleton_result.overall_status == "critical":
            print(f"  CRITICAL: Skeleton validation failed, skipping children")
            return self._build_report_and_return(
                context, skeleton, results, total_entities, counts
            )

        # Level 2: Validate modules
        print(f"  Validating {len(skeleton.domain_modules)} modules...")
        modules_to_process = []

        for module in skeleton.domain_modules:
            module_id = module.domain_name.replace(" ", "_").lower()
            module_path = [module_id]

            module_result = _validate_entity_structural(
                module, "module", module_id, module_path
            )
            results.append(module_result)
            total_entities += 1
            counts[module_result.overall_status] += 1

            if module_result.overall_status != "critical":
                modules_to_process.append((module, module_path))
            else:
                print(f"    CRITICAL: Module '{module.domain_name}' failed, skipping children")

        # Level 3-7: Validate topics → subtopics → labs → capsules → items
        items_to_validate = []

        for module, module_path in modules_to_process:
            for topic in module.topics:
                topic_id = topic.name.replace(" ", "_").lower()
                topic_path = module_path + [topic_id]

                topic_result = _validate_entity_structural(
                    topic, "topic", topic_id, topic_path
                )
                results.append(topic_result)
                total_entities += 1
                counts[topic_result.overall_status] += 1

                if topic_result.overall_status == "critical":
                    continue

                for subtopic in topic.subtopics:
                    subtopic_id = subtopic.name.replace(" ", "_").lower()
                    subtopic_path = topic_path + [subtopic_id]

                    subtopic_result = _validate_entity_structural(
                        subtopic, "subtopic", subtopic_id, subtopic_path
                    )
                    results.append(subtopic_result)
                    total_entities += 1
                    counts[subtopic_result.overall_status] += 1

                    if subtopic_result.overall_status == "critical":
                        continue

                    for lab in subtopic.labs:
                        lab_path = subtopic_path + [lab.lab_id]

                        lab_result = _validate_entity_structural(
                            lab, "lab", lab.lab_id, lab_path
                        )
                        results.append(lab_result)
                        total_entities += 1
                        counts[lab_result.overall_status] += 1

                        if lab_result.overall_status == "critical":
                            continue

                        for capsule in lab.capsules:
                            capsule_path = lab_path + [capsule.capsule_id]

                            capsule_result = _validate_entity_structural(
                                capsule, "capsule", capsule.capsule_id, capsule_path
                            )
                            results.append(capsule_result)
                            total_entities += 1
                            counts[capsule_result.overall_status] += 1

                            if capsule_result.overall_status == "critical":
                                continue

                            for item in capsule.items:
                                item_path = capsule_path + [item.item_id]

                                # Structural validation for item
                                item_result = _validate_entity_structural(
                                    item, "item", item.item_id, item_path
                                )

                                # Queue for grounding/quality validation if not critical
                                if item_result.overall_status != "critical" and not self.skip_llm_review:
                                    items_to_validate.append({
                                        "item": item,
                                        "item_result": item_result,
                                        "domain_name": module.domain_name,
                                        "topic_name": topic.name,
                                        "item_path": item_path,
                                    })
                                else:
                                    results.append(item_result)
                                    total_entities += 1
                                    counts[item_result.overall_status] += 1

        # Run LLM-based validation on items in parallel
        if items_to_validate and collection and engine:
            print(f"  Running grounding/quality validation on {len(items_to_validate)} items...")

            def validate_item_full(task):
                item = task["item"]
                item_result = task["item_result"]

                # Grounding validation
                grounding_issues = _validate_item_grounding(
                    item,
                    collection,
                    embedding_model,
                    engine,
                    cert_name,
                    task["domain_name"],
                    task["topic_name"],
                    task["item_path"],
                )
                item_result.issues.extend(grounding_issues)

                # Quality validation
                quality_issues = _validate_item_quality(
                    item,
                    collection,
                    engine,
                    cert_name,
                    task["domain_name"],
                    task["topic_name"],
                    task["item_path"],
                )
                item_result.issues.extend(quality_issues)

                # Update overall status
                all_severities = [Severity.PASSED]
                for issue in item_result.issues:
                    all_severities.append(Severity(issue.severity))
                item_result.overall_status = get_worst_severity(all_severities).value
                item_result.validator_model = engine.model_name

                return item_result

            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                future_to_task = {
                    executor.submit(validate_item_full, task): task
                    for task in items_to_validate
                }

                for future in as_completed(future_to_task):
                    try:
                        item_result = future.result()
                        results.append(item_result)
                        total_entities += 1
                        counts[item_result.overall_status] += 1
                    except Exception as e:
                        task = future_to_task[future]
                        # Create a failed result
                        item_result = task["item_result"]
                        item_result.issues.append(
                            ValidationIssue(
                                issue_id=_generate_issue_id(),
                                severity=Severity.MINOR.value,
                                rule_name="validation_error",
                                description=f"Validation failed: {str(e)}",
                                field_path="",
                            )
                        )
                        results.append(item_result)
                        total_entities += 1
                        counts[item_result.overall_status] += 1
        elif items_to_validate:
            # Add structural-only results
            for task in items_to_validate:
                results.append(task["item_result"])
                total_entities += 1
                counts[task["item_result"].overall_status] += 1

        return self._build_report_and_return(
            context, skeleton, results, total_entities, counts
        )

    def _build_report_and_return(
        self,
        context: PipelineContext,
        skeleton: CourseSkeleton,
        results: list[ValidationResult],
        total_entities: int,
        counts: dict[str, int],
    ) -> PipelineContext:
        """Build validation report, save it, and return updated context."""
        report = ValidationReport(
            certification_name=skeleton.certification_name,
            skeleton_version=skeleton.version,
            validated_at=datetime.now(),
            total_entities=total_entities,
            passed_count=counts["passed"],
            minor_count=counts["minor"],
            major_count=counts["major"],
            critical_count=counts["critical"],
            results=results,
        )

        # Save report to file
        output_dir = Path(self.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        cert_slug = skeleton.certification_name.replace(" ", "_").replace("/", "_")
        report_file = output_dir / f"{cert_slug}_validation_v{skeleton.version}.json"

        with open(report_file, "w", encoding="utf-8") as f:
            f.write(report.model_dump_json(indent=2))

        print(f"  Validation complete:")
        print(f"    - Total entities: {total_entities}")
        print(f"    - Passed: {counts['passed']}")
        print(f"    - Minor issues: {counts['minor']}")
        print(f"    - Major issues: {counts['major']}")
        print(f"    - Critical issues: {counts['critical']}")
        print(f"  Report saved to: {report_file}")

        context["validation_report"] = report
        return context
