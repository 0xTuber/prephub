"""Content generation domain models.

This module contains models for content generation:
- CapsuleItem: A practice question within a capsule
- SourceCitation, SourceReference: Source material references
- Practice question types (MultipleChoice, DragDrop, etc.)
"""

from typing import Any

from pydantic import BaseModel


class SourceCitation(BaseModel):
    """Citation to a specific book and pages."""

    book_title: str
    book_author: str
    pages: list[int] = []


class SourceChunkSnapshot(BaseModel):
    """Snapshot of a chunk used for generation (for 'see summary' feature)."""

    chunk_id: str
    text: str
    book_title: str
    book_author: str
    pages: list[int] = []
    section_heading: str | None = None
    image_paths: list[str] = []


class AnchorQuote(BaseModel):
    """A verbatim quote from source material used to ground an item."""

    text: str  # Verbatim quote from source
    chunk_id: str  # Reference to the source chunk
    chunk_index: int  # Index in the retrieved chunks list
    page_numbers: list[int] = []  # Page numbers where quote appears
    relevance_score: float = 0.0  # How relevant to learning target


class QuoteVerification(BaseModel):
    """Result of verifying a quote against source chunks."""

    quote_text: str  # The quote that was verified
    found: bool  # Whether the quote was found in chunks
    chunk_id: str | None = None  # Chunk ID where found
    match_type: str = "none"  # "exact", "partial", "paraphrase", "none"
    match_confidence: float = 0.0  # 0-1 confidence score


class ItemSourceReference(BaseModel):
    """Source reference for a capsule item, including QA chunk IDs."""

    summary: str  # 30-80 words summarizing the chunks used
    citations: list[SourceCitation] = []
    chunk_ids: list[str] = []  # For QA - references to vectorstore chunks
    optimal_images: list[str] = []  # Top 3 most relevant images
    source_chunks: list[SourceChunkSnapshot] = []  # Full chunk data for "see summary"

    # Quality tracking fields (Phase 1)
    evidence_indices: list[int] = []  # Indices of chunks actually cited in explanation
    quotes_verified: list[QuoteVerification] = []  # Verification results for each quote
    primary_evidence_chunk_id: str | None = None  # Main evidence chunk for correct answer
    anchor_quotes: list[AnchorQuote] = []  # Anchor quotes selected before drafting


class CapsuleItem(BaseModel):
    """A practice question within a capsule (exam prep item)."""

    # Skeleton fields (Step 4):
    item_id: str  # e.g., "item_01"
    item_type: str  # From exam_format.question_types (MCQ, multiple response, etc.)
    title: str
    learning_target: str  # What concept/skill this question assesses
    difficulty: str | None = None  # "beginner", "intermediate", "advanced"

    # Source grounding fields (Phase 1 quality):
    concept_tag: str | None = None  # Unique identifier for deduplication
    assigned_concepts: list[str] = []  # Concept keys from source map (e.g., ["CONCEPT 1"])

    # Content fields (Step 5 - RAG generation):
    content: str | None = None  # The actual question stem
    options: list[str] | None = None  # Answer choices (randomized order)
    correct_answer_index: int | None = None  # Index of correct answer after shuffle
    explanation: str | None = None  # Why the correct answer is correct
    source_reference: ItemSourceReference | None = None  # Source material used

    # Quality tracking fields
    generation_status: str = "pending"  # "pending", "success", "insufficient_source", "rejected"
    quality_flags: list[str] = []  # Quality issues detected (e.g., "weak_grounding", "ambiguous")


class CapsuleItemGenerationResult(BaseModel):
    """Result of generating items for a capsule."""

    domain_name: str
    topic_name: str
    subtopic_name: str
    lab_id: str
    capsule_id: str
    items: list[CapsuleItem] = []
    success: bool = True
    error: str | None = None


# Step 5: Practice Question Generation Models


class SourceHighlight(BaseModel):
    """A highlighted span of text from a source document."""

    text: str
    start_char: int | None = None
    end_char: int | None = None


class SourceReference(BaseModel):
    """Complete reference to source material for Layer 2 validation."""

    book_title: str
    book_author: str
    section_heading: str | None = None
    page_numbers: list[int] = []
    chunk_text: str
    highlights: list[SourceHighlight] = []
    image_paths: list[str] = []
    relevance_score: float | None = None


class AnswerChoice(BaseModel):
    """A single answer choice for multiple choice/response questions."""

    label: str
    text: str
    is_correct: bool = False


class PracticeQuestionBase(BaseModel):
    """Base class for all practice question types."""

    question_type: str
    stem: str
    correct_answer_explanation: str
    wrong_answer_explanations: dict[str, str] = {}
    difficulty: str | None = None
    bloom_level: str | None = None
    source_refs: list[SourceReference] = []


class MultipleChoiceQuestion(PracticeQuestionBase):
    """Multiple choice question with exactly 4 choices and one correct answer."""

    question_type: str = "multiple_choice"
    choices: list[AnswerChoice]
    correct_label: str


class MultipleResponseQuestion(PracticeQuestionBase):
    """Multiple response question with 5+ choices and multiple correct answers."""

    question_type: str = "multiple_response"
    num_correct: int
    choices: list[AnswerChoice]
    correct_labels: list[str]


class DragDropItem(BaseModel):
    """An item that can be dragged in a drag-and-drop question."""

    id: str
    text: str


class DropZone(BaseModel):
    """A zone that accepts dragged items."""

    id: str
    label: str
    accepts: list[str]  # IDs of correct items


class DragDropQuestion(PracticeQuestionBase):
    """Drag and drop question for matching items to zones."""

    question_type: str = "drag_drop"
    drag_items: list[DragDropItem]
    drop_zones: list[DropZone]


class HotspotRegion(BaseModel):
    """A clickable region in a hotspot question."""

    id: str
    shape: str  # "rect", "circle", "polygon"
    coords: list[float]
    is_correct: bool


class HotspotQuestion(PracticeQuestionBase):
    """Hotspot question where users click on image regions."""

    question_type: str = "hotspot"
    image_url: str
    regions: list[HotspotRegion]
    num_correct_regions: int


class BlankSlot(BaseModel):
    """A blank slot in a fill-in-the-blank question."""

    position: int
    correct_answers: list[str]


class FillBlankQuestion(PracticeQuestionBase):
    """Fill in the blank question."""

    question_type: str = "fill_blank"
    stem_with_blanks: str
    blanks: list[BlankSlot]


class OrderingQuestion(PracticeQuestionBase):
    """Ordering/sequence question."""

    question_type: str = "ordering"
    items: list[str]
    correct_order: list[int]


class SimulationTask(BaseModel):
    """A task within a simulation question."""

    task_id: str
    instruction: str
    expected_outcome: str
    validation_type: str  # "command_output", "file_content", "state_check"


class SimulationQuestion(PracticeQuestionBase):
    """Performance-based simulation question."""

    question_type: str = "simulation"
    environment_description: str
    initial_state: str
    tasks: list[SimulationTask]


class CaseStudyQuestion(PracticeQuestionBase):
    """Case study with scenario and sub-questions."""

    question_type: str = "case_study"
    scenario: str
    exhibits: list[str] = []
    sub_questions: list["PracticeQuestionBase"] = []


# Union type for any question
PracticeQuestion = (
    MultipleChoiceQuestion
    | MultipleResponseQuestion
    | DragDropQuestion
    | HotspotQuestion
    | FillBlankQuestion
    | OrderingQuestion
    | SimulationQuestion
    | CaseStudyQuestion
)


class TopicContent(BaseModel):
    """Generated content for a single topic."""

    domain_name: str
    topic_name: str
    questions: list[PracticeQuestionBase] = []


class TopicContentResult(BaseModel):
    """Result of generating content for a topic (success or failure)."""

    domain_name: str
    topic_name: str
    content: TopicContent | None = None
    success: bool = True
    error: str | None = None


class Course(BaseModel):
    """Complete generated course with all topics and questions."""

    certification_name: str
    exam_code: str | None = None
    topic_contents: list[TopicContent] = []
    failed_topics: list[TopicContentResult] = []
    total_questions: int = 0


class Step5Output(BaseModel):
    """Summary output for Step 5."""

    certification_name: str
    exam_code: str | None = None
    total_topics: int = 0
    topics_succeeded: int = 0
    topics_failed: int = 0
    total_questions: int = 0
    questions_by_type: dict[str, int] = {}
