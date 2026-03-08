"""Generate item content using RAG from source materials."""

import json
import random
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import TYPE_CHECKING

from course_builder.domain.content import (
    CapsuleItem,
    ItemSourceReference,
    SourceChunkSnapshot,
    SourceCitation,
)
from course_builder.domain.course import CourseSkeleton
from course_builder.pipeline.base import EngineAwareStep, PipelineContext

if TYPE_CHECKING:
    from course_builder.engine import GenerationEngine


def _load_image_annotations(annotations_path: str | Path | None) -> dict[str, str]:
    """Load image annotations and return path->description mapping."""
    if not annotations_path:
        return {}
    path = Path(annotations_path)
    if not path.exists():
        return {}
    with open(path) as f:
        data = json.load(f)
    return {
        a["path"]: a["description"]
        for a in data.get("annotations", [])
        if a.get("description")
    }


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


CONTENT_SYSTEM_PROMPT = """You are an expert exam question writer for certification preparation.
Generate high-quality practice questions based ONLY on the provided source material.
Your questions must:
1. Be directly grounded in the source content
2. Test the specific learning target accurately
3. Have plausible distractors (wrong answers) that address common misconceptions
4. Include clear explanations for why the correct answer is correct"""

CONTENT_USER_PROMPT_TEMPLATE = """Generate a practice question for this certification exam.

CERTIFICATION: {certification} ({exam_code})
DOMAIN: {domain_name}
TOPIC: {topic_name}

QUESTION SKELETON:
- Type: {item_type}
- Title: {title}
- Learning Target: {learning_target}
- Difficulty: {difficulty}

SOURCE MATERIAL (use ONLY this content):
{source_chunks}

AVAILABLE IMAGES (embed relevant ones in the summary):
{available_images}

Generate a {item_type} question that tests the learning target.
The question MUST be answerable from the source material provided.

IMPORTANT GUIDELINES:
- Do NOT use "NOT", "EXCEPT", or "Which is FALSE" style questions - they are error-prone and confusing
- Instead, ask what IS correct, what SHOULD be done, or which option BEST describes something
- All answer options must be clearly distinguishable (no A/B/C/D prefixes - just the text)
- The correct answer must be unambiguously supported by the source material
- Focus on the specific learning target - create a unique question that tests THIS specific concept

DIFFICULTY REQUIREMENTS ({difficulty}):

If BEGINNER:
- Bloom's Level: Remember/Understand - direct recall or comprehension
- Reasoning: Single-step logic (if X, then Y)
- Scenario: ONE clinical finding only, controlled environment
- Distractors: At least 2 must be CLEARLY wrong (violate basic protocol)
- Stem: Use "What is...", "Which describes...", "Identify the..."
- FORBIDDEN: Do NOT use "first", "priority", "most important", "initially"

If INTERMEDIATE:
- Bloom's Level: Apply/Analyze - use concept in context
- Reasoning: Two-step logic (Given A and B, therefore C)
- Scenario: 2-3 presenting findings, may include minor complication
- Distractors: ALL must be legitimate actions (wrong timing/context)
- Stem: Use "Based on this presentation...", "Which action addresses..."
- ALLOWED: "most appropriate" but NOT "first priority"

If ADVANCED:
- Bloom's Level: Evaluate/Synthesize - prioritization judgment
- Reasoning: Multi-step with competing priorities
- Scenario: Multiple problems OR active hazards OR time pressure
- Distractors: ALL 4 options must be valid actions (differ in SEQUENCE only)
- Stem: MUST use "first", "priority", or "initially"
- REQUIRED: Cite 2+ different pages in explanation

Return a JSON object with:
- "stem" (str): The question text
- "options" (list of str): Answer choices (4 options for MCQ, more for multiple response)
- "correct_index" (int): 0-based index of the correct answer
- "explanation" (str): Why the correct answer is correct
- "source_summary" (str): A 100-150 word learning capsule that teaches the key concepts. IMPORTANT: If relevant images are available, embed 1-2 of them naturally within the text using markdown format ![Figure: description](image_path). Place images where they best support the educational content, not at the end. Write as if teaching a student the essential information they need to know.

IMPORTANT: For multiple choice, provide exactly 4 options.

Return ONLY the JSON object, no other text."""


def _strip_code_fences(text: str) -> str:
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


def _shuffle_options(options: list[str], correct_index: int) -> tuple[list[str], int]:
    """Shuffle options and return new list with updated correct index."""
    indexed = list(enumerate(options))
    random.shuffle(indexed)
    new_options = [opt for _, opt in indexed]
    # Find where the correct answer ended up
    for new_idx, (orig_idx, _) in enumerate(indexed):
        if orig_idx == correct_index:
            return new_options, new_idx
    return new_options, 0


def _query_vectorstore(
    collection,
    embedding_model,
    query: str,
    n_results: int = 5,
) -> list[dict]:
    """Query the vectorstore and return matching chunks with metadata."""
    query_embedding = embedding_model.encode([query])[0].tolist()
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
        include=["documents", "metadatas", "distances"],
    )

    chunks = []
    if results and results["documents"] and results["documents"][0]:
        for i, doc in enumerate(results["documents"][0]):
            metadata = results["metadatas"][0][i] if results["metadatas"] else {}
            chunk_id = results["ids"][0][i] if results["ids"] else f"chunk_{i}"
            distance = results["distances"][0][i] if results["distances"] else None
            chunks.append({
                "chunk_id": chunk_id,
                "text": doc,
                "book_title": metadata.get("book_title", "Unknown"),
                "book_author": metadata.get("book_author", "Unknown"),
                "pages": [int(p) for p in metadata.get("page_numbers", "").split(",") if p],
                "section_heading": metadata.get("section_heading", ""),
                "image_paths": [p.strip() for p in metadata.get("image_paths", "").split("|||") if p.strip()],
                "distance": distance,
            })
    return chunks


def _get_image_description(
    img_path: str,
    image_annotations: dict[str, str] | None,
) -> str:
    """Get image description from annotations, trying multiple path formats."""
    if not image_annotations:
        return "No description"

    # Try exact path
    desc = image_annotations.get(img_path)
    if desc:
        return desc

    # Try with images/ prefix stripped
    if img_path.startswith("images/"):
        desc = image_annotations.get(img_path[7:])
        if desc:
            return desc

    # Try matching by filename only
    filename = img_path.split("/")[-1]
    for ann_path, ann_desc in image_annotations.items():
        if ann_path.endswith(filename):
            return ann_desc

    return "No description"


def _select_optimal_images(
    chunks: list[dict],
    max_images: int = 3,
    image_annotations: dict[str, str] | None = None,
    learning_target: str | None = None,
) -> list[str]:
    """Select most relevant images from retrieved chunks.

    If image_annotations and learning_target are provided, scores images
    based on keyword overlap between description and learning target.
    Otherwise, uses chunk distance (closer chunks = more relevant images).
    """
    # Collect all images with their chunk distances
    all_images: list[tuple[str, float]] = []
    seen = set()
    for chunk in chunks:
        dist = chunk.get("distance", 1.0)
        for img in chunk.get("image_paths", []):
            if img not in seen:
                all_images.append((img, dist))
                seen.add(img)

    if not all_images:
        return []

    # If we have annotations and a learning target, score by description relevance
    if image_annotations and learning_target:
        target_words = set(learning_target.lower().split())

        def score_image(img_path: str, chunk_dist: float) -> float:
            desc = _get_image_description(img_path, image_annotations)
            if desc == "No description":
                return chunk_dist  # Fall back to chunk distance
            desc_words = set(desc.lower().split())
            overlap = len(target_words & desc_words)
            # Lower score = better (inverted overlap + chunk distance as tiebreaker)
            return -overlap + chunk_dist * 0.1

        all_images.sort(key=lambda x: score_image(x[0], x[1]))
    else:
        # Sort by chunk distance (lower = more relevant)
        all_images.sort(key=lambda x: x[1])

    return [img for img, _ in all_images[:max_images]]


def _generate_item_content_with_engine(
    engine: "GenerationEngine",
    item: CapsuleItem,
    cert_name: str,
    exam_code: str | None,
    domain_name: str,
    topic_name: str,
    retrieved_chunks: list[dict],
    image_annotations: dict[str, str] | None = None,
) -> CapsuleItem:
    """Generate content for a single item using the generation engine."""
    from course_builder.engine import GenerationConfig

    if not retrieved_chunks:
        return item

    # Format chunks for prompt
    source_chunks_str = "\n\n---\n\n".join(
        f"[Source: {c['book_title']}, p.{','.join(str(p) for p in c['pages']) or 'N/A'}]\n{c['text']}"
        for c in retrieved_chunks[:5]
    )

    # Collect available images with descriptions
    available_images_list = []
    seen_images = set()
    for chunk in retrieved_chunks[:5]:
        for img_path in chunk.get("image_paths", []):
            if img_path not in seen_images:
                seen_images.add(img_path)
                desc = _get_image_description(img_path, image_annotations)
                available_images_list.append(f"- {img_path}: {desc}")

    available_images_str = "\n".join(available_images_list) if available_images_list else "No images available"

    prompt = CONTENT_USER_PROMPT_TEMPLATE.format(
        certification=cert_name,
        exam_code=exam_code or "N/A",
        domain_name=domain_name,
        topic_name=topic_name,
        item_type=item.item_type,
        title=item.title,
        learning_target=item.learning_target,
        difficulty=item.difficulty or "intermediate",
        source_chunks=source_chunks_str,
        available_images=available_images_str,
    )

    config = GenerationConfig(system_prompt=CONTENT_SYSTEM_PROMPT)
    result = engine.generate(prompt, config=config)

    raw_json = _strip_code_fences(result.text)
    data = json.loads(raw_json)

    # Extract generated content
    stem = data.get("stem", "")
    options = data.get("options", [])
    correct_index = data.get("correct_index", 0)
    explanation = data.get("explanation", "")
    source_summary = data.get("source_summary", "")

    # Validate summary length (100-150 words for learning capsule)
    words = source_summary.split()
    if len(words) > 150:
        source_summary = " ".join(words[:150]) + "..."

    # Shuffle options to avoid predictable answer patterns
    if options and len(options) > 1:
        shuffled_options, new_correct_index = _shuffle_options(options, correct_index)
    else:
        shuffled_options = options
        new_correct_index = correct_index

    # Build citations from chunks used
    citations = []
    seen_books = set()
    for chunk in retrieved_chunks[:5]:
        book_key = (chunk["book_title"], chunk["book_author"])
        if book_key not in seen_books:
            citations.append(SourceCitation(
                book_title=chunk["book_title"],
                book_author=chunk["book_author"],
                pages=chunk["pages"],
            ))
            seen_books.add(book_key)
        else:
            for c in citations:
                if c.book_title == chunk["book_title"]:
                    c.pages.extend(p for p in chunk["pages"] if p not in c.pages)
                    break

    # Build source chunk snapshots for "see summary" feature
    source_chunks = [
        SourceChunkSnapshot(
            chunk_id=c["chunk_id"],
            text=c["text"],
            book_title=c["book_title"],
            book_author=c["book_author"],
            pages=c["pages"],
            section_heading=c.get("section_heading") or None,
            image_paths=c.get("image_paths", []),
        )
        for c in retrieved_chunks[:5]
    ]

    # Select optimal images from chunks (use annotations if available)
    optimal_images = _select_optimal_images(
        retrieved_chunks,
        image_annotations=image_annotations,
        learning_target=item.learning_target,
    )

    # Build source reference with new fields
    source_ref = ItemSourceReference(
        summary=source_summary,
        citations=citations,
        chunk_ids=[c["chunk_id"] for c in retrieved_chunks[:5]],
        optimal_images=optimal_images,
        source_chunks=source_chunks,
    )

    # Update item with generated content
    item.content = stem
    item.options = shuffled_options
    item.correct_answer_index = new_correct_index
    item.explanation = explanation
    item.source_reference = source_ref

    return item


class ItemContentGenerationStep(EngineAwareStep):
    """Generate practice question content using RAG from source materials.

    This step:
    1. Queries the vectorstore for relevant chunks based on each item's learning_target
    2. Generates question content (stem, options, explanation) using the chunks
    3. Shuffles answer options to avoid predictable patterns
    4. Creates source references with summaries and chunk IDs for QA

    Standard pattern for initialization (keyword-only parameters):
        step = ItemContentGenerationStep(
            engine=provider.generation_engine,
            max_workers=30,
        )

    For backward compatibility, you can still use the legacy model parameter:
        step = ItemContentGenerationStep(model="gemini-flash-latest")
    """

    def __init__(
        self,
        *,  # Force keyword-only arguments
        engine: "GenerationEngine | None" = None,
        model: str | None = None,  # Legacy parameter for backward compatibility
        max_workers: int = 4,
        vectorstore_dir: str = "vectorstore",
        embedding_model_name: str = "all-MiniLM-L6-v2",
        chunks_per_query: int = 5,
        image_annotations_path: str | None = None,
    ):
        """Initialize the content generation step.

        Args:
            engine: Generation engine to use (preferred).
            model: Legacy parameter - model name for Gemini API.
                   Ignored if engine is provided.
            max_workers: Maximum concurrent generation workers.
            vectorstore_dir: Directory containing the vectorstore.
            embedding_model_name: Sentence transformer model for embeddings.
            chunks_per_query: Number of chunks to retrieve per query.
            image_annotations_path: Path to image_annotations.json for smarter image selection.
        """
        super().__init__(engine=engine)
        self._legacy_model = model or "gemini-flash-latest"
        self.max_workers = max_workers
        self.vectorstore_dir = vectorstore_dir
        self.embedding_model_name = embedding_model_name
        self.chunks_per_query = chunks_per_query
        self.image_annotations_path = image_annotations_path

    def run(self, context: PipelineContext) -> PipelineContext:
        skeleton: CourseSkeleton = context["course_skeleton"]
        cert_name = skeleton.certification_name

        # Check if vectorstore exists
        collection_name = context.get("collection_name")
        if not collection_name:
            collection_name = _sanitize_collection_name(cert_name)

        vectorstore_path = context.get("vectorstore_path", self.vectorstore_dir)

        print(f"Generating item content for '{cert_name}' using RAG...")
        print(f"  Vectorstore: {vectorstore_path}/{collection_name}")

        # Initialize ChromaDB and embedding model
        try:
            import chromadb
            from sentence_transformers import SentenceTransformer

            chroma_client = chromadb.PersistentClient(path=str(Path(vectorstore_path)))
            collection = chroma_client.get_collection(name=collection_name)
            embedding_model = SentenceTransformer(self.embedding_model_name)
            has_vectorstore = True
            chunk_count = collection.count()
            print(f"  Found {chunk_count} chunks in vectorstore")
        except Exception as e:
            print(f"  WARNING: Could not load vectorstore: {e}")
            print(f"  Skipping content generation - no source material available")
            has_vectorstore = False
            context["items_without_content"] = True
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

        # Load image annotations if available
        annotations_path = context.get("image_annotations_path", self.image_annotations_path)
        image_annotations = _load_image_annotations(annotations_path)
        if image_annotations:
            print(f"  Loaded {len(image_annotations)} image annotations for smart selection")

        # Collect all items with their parent info
        item_tasks = []
        for module in skeleton.domain_modules:
            for topic in module.topics:
                for subtopic in topic.subtopics:
                    for lab in subtopic.labs:
                        for capsule in lab.capsules:
                            for item in capsule.items:
                                # Skip items that already have content
                                if item.content:
                                    continue
                                item_tasks.append({
                                    "domain_name": module.domain_name,
                                    "topic_name": topic.name,
                                    "subtopic_name": subtopic.name,
                                    "lab_id": lab.lab_id,
                                    "capsule_id": capsule.capsule_id,
                                    "capsule_title": capsule.title,
                                    "item": item,
                                })

        if not item_tasks:
            print("  No items to generate content for")
            return context

        print(f"  Generating content for {len(item_tasks)} items...")

        # Process items
        succeeded = 0
        failed = 0
        failed_items = []

        def process_item(task):
            item = task["item"]
            # Build query from learning target and context
            query = f"{task['topic_name']} {task['subtopic_name']} {item.learning_target}"

            # Retrieve relevant chunks
            chunks = _query_vectorstore(
                collection,
                embedding_model,
                query,
                n_results=self.chunks_per_query,
            )

            if not chunks:
                return None, "No relevant chunks found"

            # Generate content
            updated_item = _generate_item_content_with_engine(
                engine,
                item,
                cert_name,
                skeleton.exam_code,
                task["domain_name"],
                task["topic_name"],
                chunks,
                image_annotations=image_annotations,
            )
            return updated_item, None

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_task = {
                executor.submit(process_item, task): task
                for task in item_tasks
            }

            for future in as_completed(future_to_task):
                task = future_to_task[future]
                try:
                    updated_item, error = future.result()
                    if error:
                        failed += 1
                        failed_items.append({
                            "item_id": task["item"].item_id,
                            "capsule_id": task["capsule_id"],
                            "error": error,
                        })
                    else:
                        succeeded += 1
                except Exception as e:
                    failed += 1
                    failed_items.append({
                        "item_id": task["item"].item_id,
                        "capsule_id": task["capsule_id"],
                        "error": str(e),
                    })

        print(f"  Content generation: {succeeded} succeeded, {failed} failed")

        if failed_items:
            context["failed_content_generation"] = failed_items

        context["course_skeleton"] = skeleton
        context["content_generation_complete"] = True
        print(f"Item content generation complete for '{cert_name}'.\n")
        return context
