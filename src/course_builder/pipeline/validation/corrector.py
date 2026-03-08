"""Correction Application Step.

Applies corrections from queue to skeleton.
Handles auto_fix, regeneration, and version bumping.
"""

import json
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

from course_builder.domain.content import CapsuleItem
from course_builder.domain.course import CourseSkeleton
from course_builder.domain.validation import (
    CorrectionAction,
    CorrectionQueue,
    Step6Output,
)
from course_builder.pipeline.base import EngineAwareStep, PipelineContext
from course_builder.pipeline.validation.rules import Severity, get_structural_rules

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


def _find_entity_in_skeleton(
    skeleton: CourseSkeleton,
    entity_type: str,
    entity_path: list[str],
):
    """Find an entity in the skeleton by its path.

    Returns (entity, parent_list, index) where:
    - entity is the found object
    - parent_list is the list containing the entity (for replacement)
    - index is the position in parent_list
    """
    if entity_type == "skeleton":
        return skeleton, None, None

    if not entity_path:
        return None, None, None

    # Navigate through the hierarchy
    current = skeleton

    # Path format: [module_id, topic_id, subtopic_id, lab_id, capsule_id, item_id]
    # We need to match against the actual names/IDs in the structure

    for module in skeleton.domain_modules:
        module_id = module.domain_name.replace(" ", "_").lower()

        if entity_type == "module" and len(entity_path) == 1:
            if module_id == entity_path[0]:
                idx = skeleton.domain_modules.index(module)
                return module, skeleton.domain_modules, idx

        if len(entity_path) < 1 or module_id != entity_path[0]:
            continue

        for topic in module.topics:
            topic_id = topic.name.replace(" ", "_").lower()

            if entity_type == "topic" and len(entity_path) == 2:
                if topic_id == entity_path[1]:
                    idx = module.topics.index(topic)
                    return topic, module.topics, idx

            if len(entity_path) < 2 or topic_id != entity_path[1]:
                continue

            for subtopic in topic.subtopics:
                subtopic_id = subtopic.name.replace(" ", "_").lower()

                if entity_type == "subtopic" and len(entity_path) == 3:
                    if subtopic_id == entity_path[2]:
                        idx = topic.subtopics.index(subtopic)
                        return subtopic, topic.subtopics, idx

                if len(entity_path) < 3 or subtopic_id != entity_path[2]:
                    continue

                for lab in subtopic.labs:
                    if entity_type == "lab" and len(entity_path) == 4:
                        if lab.lab_id == entity_path[3]:
                            idx = subtopic.labs.index(lab)
                            return lab, subtopic.labs, idx

                    if len(entity_path) < 4 or lab.lab_id != entity_path[3]:
                        continue

                    for capsule in lab.capsules:
                        if entity_type == "capsule" and len(entity_path) == 5:
                            if capsule.capsule_id == entity_path[4]:
                                idx = lab.capsules.index(capsule)
                                return capsule, lab.capsules, idx

                        if len(entity_path) < 5 or capsule.capsule_id != entity_path[4]:
                            continue

                        for item in capsule.items:
                            if entity_type == "item" and len(entity_path) == 6:
                                if item.item_id == entity_path[5]:
                                    idx = capsule.items.index(item)
                                    return item, capsule.items, idx

    return None, None, None


def _apply_auto_fix(
    skeleton: CourseSkeleton,
    action: CorrectionAction,
) -> bool:
    """Apply auto-fix corrections to an entity.

    Returns True if successful, False otherwise.
    """
    entity, parent_list, idx = _find_entity_in_skeleton(
        skeleton, action.entity_type, action.entity_path
    )

    if entity is None:
        return False

    if not action.field_corrections:
        return True  # Nothing to fix

    # Apply field corrections
    for field_path, new_value in action.field_corrections.items():
        try:
            # Handle nested paths like "source_reference.summary"
            parts = field_path.split(".")
            target = entity

            for part in parts[:-1]:
                if hasattr(target, part):
                    target = getattr(target, part)
                else:
                    return False

            final_field = parts[-1]
            if hasattr(target, final_field):
                setattr(target, final_field, new_value)
            else:
                return False
        except Exception:
            return False

    return True


REGENERATION_PROMPT = """Regenerate content for this practice question.

{regenerate_prompt}

CERTIFICATION: {certification}
QUESTION TYPE: {item_type}
LEARNING TARGET: {learning_target}

SOURCE MATERIAL:
{source_chunks}

Generate a new question that:
1. Addresses the identified issues
2. Is clearly worded and unambiguous
3. Has exactly one correct answer that is definitively correct
4. Has plausible distractors
5. Includes a thorough explanation

Return a JSON object:
{{
  "stem": "The question text",
  "options": ["Option A", "Option B", "Option C", "Option D"],
  "correct_index": 0,
  "explanation": "Why the correct answer is correct"
}}
"""


def _regenerate_item(
    item: CapsuleItem,
    action: CorrectionAction,
    engine: "GenerationEngine",
    collection,
    cert_name: str,
) -> bool:
    """Regenerate an item's content using LLM.

    Returns True if successful, False otherwise.
    """
    from course_builder.engine import GenerationConfig

    if not action.regenerate_prompt:
        return False

    # Get source chunks if available
    source_chunks = ""
    if item.source_reference and item.source_reference.chunk_ids and collection:
        try:
            chunk_ids = item.source_reference.chunk_ids
            results = collection.get(ids=chunk_ids, include=["documents"])
            source_chunks = "\n\n---\n\n".join(results["documents"]) if results["documents"] else ""
        except Exception:
            source_chunks = "(Source material not available)"

    prompt = REGENERATION_PROMPT.format(
        regenerate_prompt=action.regenerate_prompt,
        certification=cert_name,
        item_type=item.item_type or "Multiple Choice",
        learning_target=item.learning_target or "",
        source_chunks=source_chunks or "(No source material provided)",
    )

    config = GenerationConfig(
        system_prompt="You are an expert exam question writer. Generate high-quality practice questions.",
    )

    def _call():
        return engine.generate(prompt, config=config)

    try:
        result = _call_with_retry(_call)
        raw_json = _strip_code_fences(result.text)
        data = json.loads(raw_json)

        # Update item with regenerated content
        item.content = data.get("stem", item.content)
        item.options = data.get("options", item.options)
        item.correct_answer_index = data.get("correct_index", item.correct_answer_index)
        item.explanation = data.get("explanation", item.explanation)

        return True
    except Exception:
        return False


def _quick_validate_item(item: CapsuleItem) -> bool:
    """Quick structural validation of a regenerated item.

    Returns True if item passes basic validation.
    """
    rules = get_structural_rules("item")

    for rule in rules:
        if rule.severity_if_failed == Severity.CRITICAL:
            passed, _, _ = rule.check_fn(item)
            if not passed:
                return False

    return True


class CorrectionApplicationStep(EngineAwareStep):
    """Applies corrections from queue to skeleton.

    Handles:
    - auto_fix actions: Direct field updates
    - regenerate actions: LLM-based content regeneration
    - manual_review actions: Skip for now, log for human review

    After applying corrections:
    - Re-validates regenerated items (quick structural check)
    - Updates queue status (applied/failed)
    - Bumps skeleton version
    - Saves corrected skeleton

    Standard pattern for initialization (keyword-only parameters):
        step = CorrectionApplicationStep(
            engine=provider.generation_engine,
            max_workers=10,
        )

    For backward compatibility, you can still use the legacy model parameter:
        step = CorrectionApplicationStep(model="gemini-flash-latest")
    """

    def __init__(
        self,
        *,  # Force keyword-only arguments
        engine: "GenerationEngine | None" = None,
        model: str | None = None,  # Legacy parameter for backward compatibility
        max_workers: int = 8,
        max_regeneration_attempts: int = 2,
        corrections_dir: str = "corrections",
        output_dir: str = "output",
        vectorstore_dir: str = "vectorstore",
    ):
        """Initialize the correction application step.

        Args:
            engine: Generation engine to use (preferred).
            model: Legacy parameter - model name for Gemini API.
                   Ignored if engine is provided.
            max_workers: Maximum concurrent regeneration workers.
            max_regeneration_attempts: Max attempts per regeneration.
            corrections_dir: Directory for correction queue files.
            output_dir: Directory for output skeleton files.
            vectorstore_dir: Directory containing the vectorstore.
        """
        super().__init__(engine=engine)
        self._legacy_model = model or "gemini-flash-latest"
        self.max_workers = max_workers
        self.max_regeneration_attempts = max_regeneration_attempts
        self.corrections_dir = corrections_dir
        self.output_dir = output_dir
        self.vectorstore_dir = vectorstore_dir

    def run(self, context: PipelineContext) -> PipelineContext:
        skeleton: CourseSkeleton = context["course_skeleton"]
        queue: CorrectionQueue | None = context.get("correction_queue")

        cert_name = skeleton.certification_name
        print(f"Applying corrections for '{cert_name}'...")

        # Try to load queue from file if not in context
        if not queue:
            queue_path = context.get("correction_queue_path")
            if queue_path and Path(queue_path).exists():
                queue = CorrectionQueue.load(Path(queue_path))
                print(f"  Loaded queue from: {queue_path}")
            else:
                print("  No correction queue found, skipping application")
                return context

        if not queue.actions:
            print("  No corrections to apply")
            skeleton.version = queue.target_version
            skeleton.validated_at = datetime.now()
            skeleton.validation_status = "passed"

            # Still save the skeleton
            output_dir = Path(self.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            cert_slug = cert_name.replace(" ", "_").replace("/", "_")
            skeleton_file = output_dir / f"{cert_slug}_skeleton_v{skeleton.version}.json"
            with open(skeleton_file, "w", encoding="utf-8") as f:
                f.write(skeleton.model_dump_json(indent=2))
            latest_file = output_dir / f"{cert_slug}_skeleton_latest.json"
            with open(latest_file, "w", encoding="utf-8") as f:
                f.write(skeleton.model_dump_json(indent=2))

            # Build output summary
            report = context.get("validation_report")
            output = Step6Output(
                certification_name=cert_name,
                input_version=queue.source_version,
                output_version=skeleton.version,
                total_entities_validated=report.total_entities if report else 0,
                passed_count=report.passed_count if report else 0,
                minor_count=report.minor_count if report else 0,
                major_count=report.major_count if report else 0,
                critical_count=report.critical_count if report else 0,
                corrections_applied=0,
                corrections_failed=0,
                validation_status=skeleton.validation_status,
            )

            context["course_skeleton"] = skeleton
            context["step6_output"] = output
            return context

        # Get or create engine
        engine = self.get_engine()
        if engine is None:
            # Legacy fallback: create Gemini engine
            from course_builder.engine import create_engine

            engine = create_engine("gemini", model=self._legacy_model)
            print(f"  Using model: {engine.model_name} (legacy mode)")
        else:
            print(f"  Using engine: {engine.engine_type}, model: {engine.model_name}")

        # Initialize vectorstore for regeneration
        collection = None

        collection_name = context.get("collection_name")
        if not collection_name:
            collection_name = _sanitize_collection_name(cert_name)

        vectorstore_path = context.get("vectorstore_path", self.vectorstore_dir)

        try:
            import chromadb

            chroma_client = chromadb.PersistentClient(path=str(Path(vectorstore_path)))
            collection = chroma_client.get_collection(name=collection_name)
        except Exception as e:
            print(f"  WARNING: Could not load vectorstore: {e}")
            print(f"  Regeneration will proceed without source material")

        # Group actions by type
        auto_fix_actions = [a for a in queue.actions if a.action_type == "auto_fix" and a.status == "pending"]
        regenerate_actions = [a for a in queue.actions if a.action_type == "regenerate" and a.status == "pending"]
        manual_review_actions = [a for a in queue.actions if a.action_type == "manual_review" and a.status == "pending"]

        print(f"  Actions to process:")
        print(f"    - Auto-fix: {len(auto_fix_actions)}")
        print(f"    - Regenerate: {len(regenerate_actions)}")
        print(f"    - Manual review: {len(manual_review_actions)} (skipped)")

        applied_count = 0
        failed_count = 0

        # Apply auto-fix actions (synchronous, fast)
        for action in auto_fix_actions:
            action.status = "in_progress"
            try:
                if _apply_auto_fix(skeleton, action):
                    action.status = "applied"
                    action.applied_at = datetime.now()
                    applied_count += 1
                else:
                    action.status = "failed"
                    action.error = "Could not find entity or apply fix"
                    failed_count += 1
            except Exception as e:
                action.status = "failed"
                action.error = str(e)
                failed_count += 1

        # Apply regeneration actions (parallel LLM calls)
        if regenerate_actions and engine:
            print(f"  Regenerating {len(regenerate_actions)} items...")

            def regenerate_item_action(action: CorrectionAction) -> tuple[CorrectionAction, bool]:
                if action.entity_type != "item":
                    # Currently only support item regeneration
                    action.status = "skipped"
                    return action, False

                entity, parent_list, idx = _find_entity_in_skeleton(
                    skeleton, action.entity_type, action.entity_path
                )

                if entity is None:
                    action.status = "failed"
                    action.error = "Could not find entity in skeleton"
                    return action, False

                action.status = "in_progress"

                # Try regeneration with retries
                for attempt in range(self.max_regeneration_attempts):
                    success = _regenerate_item(
                        entity,
                        action,
                        engine,
                        collection,
                        cert_name,
                    )

                    if success and _quick_validate_item(entity):
                        action.status = "applied"
                        action.applied_at = datetime.now()
                        return action, True

                action.status = "failed"
                action.error = f"Regeneration failed after {self.max_regeneration_attempts} attempts"
                return action, False

            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                futures = {
                    executor.submit(regenerate_item_action, action): action
                    for action in regenerate_actions
                }

                for future in as_completed(futures):
                    try:
                        action, success = future.result()
                        if success:
                            applied_count += 1
                        else:
                            failed_count += 1
                    except Exception as e:
                        action = futures[future]
                        action.status = "failed"
                        action.error = str(e)
                        failed_count += 1
        elif regenerate_actions:
            # No engine available, skip regeneration
            for action in regenerate_actions:
                action.status = "skipped"
                action.error = "No generation engine available for regeneration"

        # Skip manual review actions (they need human intervention)
        for action in manual_review_actions:
            action.status = "skipped"
            action.error = "Manual review required"

        # Update skeleton version
        skeleton.version = queue.target_version
        skeleton.validated_at = datetime.now()

        # Determine overall status
        if failed_count == 0 and len(manual_review_actions) == 0:
            skeleton.validation_status = "passed" if applied_count == 0 else "corrected"
        elif len(manual_review_actions) > 0:
            skeleton.validation_status = "needs_review"
        else:
            skeleton.validation_status = "corrected"

        # Save updated queue
        corrections_dir = Path(self.corrections_dir)
        corrections_dir.mkdir(parents=True, exist_ok=True)

        cert_slug = cert_name.replace(" ", "_").replace("/", "_")
        queue_file = corrections_dir / f"{cert_slug}_corrections_v{queue.source_version}_to_v{queue.target_version}.jsonl"
        queue.save(queue_file)

        # Update state file
        state_file = corrections_dir / f"{cert_slug}_state.json"
        state = {
            "certification_name": cert_name,
            "current_version": skeleton.version,
            "last_validation": skeleton.validated_at.isoformat() if skeleton.validated_at else None,
            "pending_corrections": queue.pending_count,
            "applied_corrections": queue.applied_count,
        }
        with open(state_file, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2)

        # Save corrected skeleton
        output_dir = Path(self.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        skeleton_file = output_dir / f"{cert_slug}_skeleton_v{skeleton.version}.json"
        with open(skeleton_file, "w", encoding="utf-8") as f:
            f.write(skeleton.model_dump_json(indent=2))

        # Also save as "latest"
        latest_file = output_dir / f"{cert_slug}_skeleton_latest.json"
        with open(latest_file, "w", encoding="utf-8") as f:
            f.write(skeleton.model_dump_json(indent=2))

        print(f"  Corrections applied:")
        print(f"    - Applied: {applied_count}")
        print(f"    - Failed: {failed_count}")
        print(f"    - Skipped (manual): {len(manual_review_actions)}")
        print(f"  Skeleton version: {skeleton.version}")
        print(f"  Status: {skeleton.validation_status}")
        print(f"  Saved to: {skeleton_file}")

        # Build output summary
        report = context.get("validation_report")
        output = Step6Output(
            certification_name=cert_name,
            input_version=queue.source_version,
            output_version=skeleton.version,
            total_entities_validated=report.total_entities if report else 0,
            passed_count=report.passed_count if report else 0,
            minor_count=report.minor_count if report else 0,
            major_count=report.major_count if report else 0,
            critical_count=report.critical_count if report else 0,
            corrections_applied=applied_count,
            corrections_failed=failed_count,
            validation_status=skeleton.validation_status or "unknown",
        )

        context["course_skeleton"] = skeleton
        context["correction_queue"] = queue
        context["step6_output"] = output
        return context
