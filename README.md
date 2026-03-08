# Course Builder

AI-powered certification exam preparation course generator. Automatically creates structured learning content from source materials using LLMs and RAG (Retrieval-Augmented Generation), with an interactive web frontend for learning and review.

## Screenshots

### Course Dashboard
![Course Dashboard](docs/images/homepage.png)
*Home page showing course grid with "Review Center" and "+ New Course" buttons*

### Learning Path
![Learning Path](docs/images/learning-path.png)
*Course detail page with interactive learning path and module progress*

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              COURSE BUILDER                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐                   │
│  │   PDF Book   │───▶│   MinerU     │───▶│  ChromaDB    │                   │
│  │   Upload     │    │  Extraction  │    │  Vectorstore │                   │
│  └──────────────┘    └──────────────┘    └──────┬───────┘                   │
│                                                  │                           │
│                                                  ▼                           │
│  ┌──────────────┐    ┌──────────────────────────────────────────────────┐   │
│  │   Google     │    │              LLM Pipeline (Gemini/vLLM)          │   │
│  │   Search     │───▶│  ┌────────┐ ┌────────┐ ┌────────┐ ┌───────────┐  │   │
│  │  (Exam Info) │    │  │  Exam  │▶│ Course │▶│  Labs  │▶│ Capsules  │  │   │
│  └──────────────┘    │  │ Format │ │Skeleton│ │        │ │ + Items   │  │   │
│                      │  └────────┘ └────────┘ └────────┘ └───────────┘  │   │
│                      └──────────────────────────────────────────────────┘   │
│                                                  │                           │
│                                                  ▼                           │
│                      ┌──────────────────────────────────────────────────┐   │
│                      │           course_skeleton.json                    │   │
│                      │  (Modules, Topics, Labs, Questions, Answers)     │   │
│                      └──────────────────────────────┬───────────────────┘   │
│                                                      │                       │
└──────────────────────────────────────────────────────┼───────────────────────┘
                                                       │
                                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           NEXT.JS FRONTEND                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │                           HOME PAGE                                     │ │
│  │  ┌─────────────────────────────────────────────────────────────────┐   │ │
│  │  │  Generation Banner (non-blocking progress)                       │   │ │
│  │  └─────────────────────────────────────────────────────────────────┘   │ │
│  │  ┌──────────────────┐  ┌─────────────┐  ┌─────────────┐                │ │
│  │  │ [+ New Course]   │  │ [Review]    │  │   [Theme]   │                │ │
│  │  └──────────────────┘  └─────────────┘  └─────────────┘                │ │
│  │  ┌─────────────────────────────────────────────────────────────────┐   │ │
│  │  │  Review Widget - "15 questions ready" [Start Review]             │   │ │
│  │  └─────────────────────────────────────────────────────────────────┘   │ │
│  │  ┌─────────┐  ┌─────────┐  ┌─────────┐                                 │ │
│  │  │ Course  │  │ Course  │  │ Course  │  ... Course Grid               │ │
│  │  │  Card   │  │  Card   │  │  Card   │                                 │ │
│  │  └─────────┘  └─────────┘  └─────────┘                                 │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                                     │                                        │
│            ┌────────────────────────┼────────────────────────┐              │
│            ▼                        ▼                        ▼              │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐         │
│  │  Course Detail  │    │  Lab Practice   │    │  Review Center  │         │
│  │  Learning Path  │───▶│  Interactive    │    │  Spaced         │         │
│  │  Navigator      │    │  Questions      │    │  Repetition     │         │
│  └─────────────────┘    └────────┬────────┘    └────────┬────────┘         │
│                                  │                       │                  │
│                                  └───────────────────────┘                  │
│                                              │                              │
│                                              ▼                              │
│                                  ┌─────────────────┐                        │
│                                  │   localStorage  │                        │
│                                  │   (Progress)    │                        │
│                                  └─────────────────┘                        │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

## Features

### Backend (Python)
- **Exam Format Discovery**: Uses Google Search grounding to find official exam specifications
- **Hierarchical Course Structure**: Generates domain modules, topics, subtopics, labs, and capsules
- **Source-Grounded Content**: All questions are backed by citations from source materials
- **Quality Gates**: Built-in verification, novelty detection, and ambiguity checking
- **Multiple LLM Backends**: Supports Gemini API and vLLM (local or server)
- **Checkpoint/Resume**: Long-running pipelines can be stopped and resumed
- **Parallel Processing**: Configurable worker pools for faster generation

### Frontend (Next.js)
- **Course Dashboard**: View and manage all generated courses
- **Create Course Modal**: Upload PDFs and generate courses with non-blocking progress
- **Learning Path Navigator**: Visual path through modules, topics, and labs
- **Interactive Labs**: Practice questions with immediate feedback
- **Review Center**: Spaced repetition system for long-term retention

## Review Center

The Review Center uses spaced repetition to help you retain knowledge long-term. It works by:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         REVIEW CENTER FLOW                               │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  1. QUESTION POOL                                                        │
│     ┌─────────────────────────────────────────────────────────────────┐ │
│     │  All questions from completed labs across all courses            │ │
│     │  [Q1] [Q2] [Q3] [Q4] [Q5] [Q6] [Q7] [Q8] [Q9] ...               │ │
│     └─────────────────────────────────────────────────────────────────┘ │
│                                    │                                     │
│                                    ▼                                     │
│  2. SELECT SESSION SIZE                                                  │
│     ┌─────────────────────────────────────────────────────────────────┐ │
│     │  "How many questions?"                                           │ │
│     │  [ - ]  10  [ + ]                                                │ │
│     │  ═══════●════════════  (1 to 50 available)                       │ │
│     │            [Begin Session]                                        │ │
│     └─────────────────────────────────────────────────────────────────┘ │
│                                    │                                     │
│                                    ▼                                     │
│  3. ANSWER QUESTIONS                                                     │
│     ┌─────────────────────────────────────────────────────────────────┐ │
│     │  ▓▓▓▓▓▓▓▓▓░░░░░░░░░░░░░░░  3/10                                 │ │
│     │                                                                   │ │
│     │  "What is the correct procedure for..."                          │ │
│     │                                                                   │ │
│     │  ○ Option A                                                       │ │
│     │  ● Option B  ← selected                                           │ │
│     │  ○ Option C                                                       │ │
│     │  ○ Option D                                                       │ │
│     │                                                                   │ │
│     │            [Check Answer]                                         │ │
│     └─────────────────────────────────────────────────────────────────┘ │
│                                    │                                     │
│                                    ▼                                     │
│  4. RATE CONFIDENCE                                                      │
│     ┌─────────────────────────────────────────────────────────────────┐ │
│     │  ✓ Correct!                                                      │ │
│     │                                                                   │ │
│     │  "How confident were you?"                                        │ │
│     │                                                                   │ │
│     │  [😰]  [😕]  [😐]  [😊]  [🤩]                                      │ │
│     │  Guess  Unsure Some  Confident Certain                            │ │
│     └─────────────────────────────────────────────────────────────────┘ │
│                                    │                                     │
│                                    ▼                                     │
│  5. SESSION SUMMARY                                                      │
│     ┌─────────────────────────────────────────────────────────────────┐ │
│     │           ╭──────────╮                                            │ │
│     │           │   80%    │  ← Accuracy ring                           │ │
│     │           ╰──────────╯                                            │ │
│     │                                                                   │ │
│     │  ●●●●●●●●○○  ← Per-question results                               │ │
│     │                                                                   │ │
│     │  ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐                     │ │
│     │  │   10   │ │  80%   │ │  3.5   │ │   5m   │                     │ │
│     │  │Questions│ │Accuracy│ │Confid. │ │ Time   │                     │ │
│     │  └────────┘ └────────┘ └────────┘ └────────┘                     │ │
│     │                                                                   │ │
│     │  [Review More]              [Done]                                │ │
│     └─────────────────────────────────────────────────────────────────┘ │
│                                                                          │
└──────────────────────────────────────────────────────────────────────────┘
```

### How Spaced Repetition Works

1. **Answer + Confidence**: After each question, rate how confident you were
2. **Smart Scheduling**: The algorithm uses your accuracy and confidence to prioritize questions
3. **Priority System**: Questions you got wrong (especially with high confidence) are flagged for more frequent review
4. **Long-term Retention**: Studies show spaced repetition leads to 2x better retention vs cramming

## Prerequisites

- Python 3.11+ (for backend/generation)
- Node.js 18+ (for frontend)
- A [Google AI API key](https://aistudio.google.com/apikey) set as `GOOGLE_API_KEY`
- **GPU (recommended)**: NVIDIA GPU with 24GB+ VRAM for local vLLM inference
- (Optional) vLLM server for local inference

### GPU Requirements

For local LLM inference with vLLM, you'll need a capable GPU. This project has been primarily developed and tested on:

- **Lambda Labs A10 instance** (24GB VRAM) - Recommended for running Qwen2.5-7B-Instruct and similar models
- Works with any NVIDIA GPU with 16GB+ VRAM (RTX 4090, A100, etc.)

You can also use cloud-based Gemini API without a GPU, but local vLLM provides faster iteration and lower costs for bulk content generation.

## Installation

### Option 1: Docker (Recommended)

```bash
# Clone the repository
git clone https://github.com/yourusername/course-builder.git
cd course-builder

# Create .env file with your API key
echo GOOGLE_API_KEY=your-key-here > .env

# Build and run
docker-compose build
docker-compose run course-builder generate --certification "NREMT EMT"
```

### Option 2: Local Development

**Backend (Python):**
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -e .

# Create a .env file with your API key
echo GOOGLE_API_KEY=your-key-here > .env
```

**Frontend (Next.js):**
```bash
cd frontend

# Install dependencies
pnpm install  # or npm install

# Run development server
pnpm dev  # or npm run dev
```

The frontend will be available at http://localhost:3000

## Quick Start

### Using the Frontend

1. **Start the frontend**: `cd frontend && pnpm dev`
2. **Click "+ New Course"** in the header
3. **Upload a PDF** book in the dialog
4. **Enter a course name** and click "Generate"
5. **Browse courses** while generation happens (non-blocking)
6. **Click a course** to view the learning path
7. **Practice labs** and review in the **Review Center**

### Using the CLI

```bash
# Prepare source materials
# Place PDFs in data/sources/downloads/YOUR_CERT/

# Extract and embed sources
course-builder batch-extract data/sources/downloads/

# Generate course
course-builder generate --certification "NREMT EMT"
```

## Frontend Structure

```
frontend/
├── app/
│   ├── page.tsx                    # Home page with course grid
│   ├── courses/[slug]/             # Course detail pages
│   │   ├── page.tsx                # Learning path navigator
│   │   └── labs/[labId]/page.tsx   # Interactive lab practice
│   ├── review/page.tsx             # Review Center
│   └── api/                        # API routes
│       ├── courses/                # Course data endpoints
│       ├── generate/               # Generation endpoints
│       ├── upload/                 # File upload endpoint
│       └── review/                 # Review questions endpoint
├── components/
│   ├── GenerationBanner.tsx        # Non-blocking progress banner
│   ├── CreateCourseDialog.tsx      # New course modal
│   ├── ReviewWidget.tsx            # Home page review widget
│   ├── courses/                    # Course components
│   ├── learning-path/              # Learning path components
│   ├── interactive-labs/           # Lab practice components
│   └── cbr/                        # Review center components
└── lib/
    └── types.ts                    # TypeScript type definitions
```

## Pipeline Stages

The generation pipeline has several stages that can be run independently:

| Stage | Description | Output |
|-------|-------------|--------|
| `exam` | Discover exam format from web | Exam structure (domains, question types) |
| `course` | Generate course outline | Domain modules with topics |
| `labs` | Create lab structure | Subtopics and labs |
| `capsules` | Generate learning capsules | Capsule skeletons with learning targets |
| `items` | Create assessment items | Question skeletons |
| `content` | Generate full content | Complete questions with explanations |
| `validated` | Quality validation | Verified, source-grounded content |

### Checkpoints

Use `--stop-after` and `--resume-from` to run stages separately:

```bash
# Run skeleton stages with Gemini
course-builder generate --certification "NREMT EMT" --stop-after capsules

# List available checkpoints
course-builder checkpoints

# Resume with vLLM for content generation
course-builder generate --certification "NREMT EMT" \
    --resume-from data/skeletons/NREMT_EMT_v1_capsules.json \
    --skip-sources
```

## LLM Backends

### Gemini (Cloud)

Default backend. Requires `GOOGLE_API_KEY` in environment.

```bash
course-builder generate --certification "Your Cert" \
    --skeleton-model gemini-2.0-flash \
    --validation-model gemini-2.0-flash-thinking-exp
```

### vLLM Server (Local)

Run a local vLLM server for cost-effective generation. Tested on **Lambda Labs A10 GPU instances**:

```bash
# On Lambda Labs A10 (or similar 24GB GPU)
# Install vLLM
pip install vllm

# Start vLLM server
vllm serve Qwen/Qwen2.5-7B-Instruct \
    --port 8000 \
    --tensor-parallel-size 1 \
    --max-model-len 32768

# Use for content generation
course-builder generate --certification "Your Cert" \
    --item-engine vllm-server \
    --base-url http://localhost:8000/v1 \
    --item-model Qwen/Qwen2.5-7B-Instruct
```

> **Tip**: Lambda Labs A10 instances (~$0.60/hr) provide excellent price/performance for running 7B parameter models. The A10's 24GB VRAM comfortably handles Qwen2.5-7B-Instruct with 32k context.

### Hybrid Setup (Recommended)

Use Gemini for planning/validation, vLLM for bulk generation:

```bash
course-builder generate --certification "Your Cert" \
    --skeleton-model gemini-2.0-flash \
    --capsule-engine vllm-server \
    --item-engine vllm-server \
    --item-model Qwen/Qwen2.5-7B-Instruct \
    --validation-model gemini-2.0-flash-thinking-exp \
    --base-url http://localhost:8000/v1
```

## Output Structure

Generated courses follow this hierarchy:

```
CourseSkeleton
├── certification_name: "NREMT EMT"
├── exam_format: ExamFormat
│   ├── total_questions: 120
│   ├── domains: [Domain, ...]
│   └── question_types: [QuestionType, ...]
└── domain_modules: [DomainModule, ...]
    └── topics: [Topic, ...]
        └── subtopics: [Subtopic, ...]
            └── labs: [Lab, ...]
                └── capsules: [Capsule, ...]
                    ├── title: "Scene Safety Assessment"
                    ├── learning_target: "Identify hazards..."
                    └── items: [CapsuleItem, ...]
                        ├── question_stem: "..."
                        ├── answer_options: [...]
                        ├── correct_answer: "B"
                        ├── explanation: "..."
                        └── source_citations: [...]
```

## Data Persistence

The `./data` directory stores all outputs:

```
data/
├── sources/
│   ├── downloads/    # PDF textbooks
│   └── extracted/    # MinerU output
├── vectorstore/      # ChromaDB collections
└── [course-slug]/    # Generated course files
    └── course_skeleton.json
```

## Running Tests

```bash
# Backend tests
pytest tests/ -v

# Frontend tests
cd frontend
pnpm test
```

## License

MIT License - see LICENSE file for details.
