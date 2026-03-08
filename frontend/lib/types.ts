// Course skeleton types matching the Python domain models

export interface Capsule {
  capsule_id: string
  title: string
  description?: string
  learning_goal: string
  capsule_type: string
  estimated_duration_minutes?: number
  items: CapsuleItem[]
  prerequisites_within_lab: string[]
  assessment_criteria: string[]
  common_errors: string[]
}

export interface CapsuleItem {
  item_id: string
  question_type: string
  stem: string
  options?: AnswerOption[]
  correct_answer?: string | string[]
  explanation?: string
  difficulty?: string
}

export interface AnswerOption {
  id: string
  text: string
  is_correct?: boolean
}

export interface Lab {
  lab_id: string
  title: string
  description?: string
  objective: string
  lab_type: string
  estimated_duration_minutes?: number
  tools_required: string[]
  capsules: Capsule[]
  prerequisites_within_subtopic: string[]
  success_criteria: string[]
  real_world_application?: string
}

export interface SubTopic {
  name: string
  description?: string
  key_concepts: string[]
  practical_skills: string[]
  common_misconceptions: string[]
  labs: Lab[]
}

export interface CourseTopic {
  name: string
  description?: string
  learning_objectives: LearningObjective[]
  subtopics: SubTopic[]
  estimated_study_hours?: number
}

export interface LearningObjective {
  objective: string
  bloom_level?: string
  relevant_question_types: string[]
}

export interface CourseModule {
  domain_name: string
  domain_weight_pct?: number
  overview?: string
  topics: CourseTopic[]
  prerequisites_for_domain: string[]
  recommended_study_order: string[]
  official_references: string[]
}

export interface CourseOverview {
  target_audience?: string
  course_description?: string
  total_estimated_study_hours?: number
  study_strategies: StudyStrategy[]
  exam_day_tips: string[]
  prerequisites_detail: string[]
}

export interface StudyStrategy {
  name: string
  description?: string
  when_to_use?: string
}

export interface CourseSkeleton {
  certification_name: string
  exam_code?: string
  overview: CourseOverview
  domain_modules: CourseModule[]
  version: number
  validated_at?: string
  validation_status?: string
}

// Frontend specific types

export interface Course {
  slug: string
  name: string
  status: "processing" | "ready" | "error"
  moduleCount: number
  labCount: number
  createdAt: string
  skeleton?: CourseSkeleton
}

export interface GenerationJob {
  jobId: string
  status: "pending" | "running" | "completed" | "error"
  progress: number
  currentStep?: string
  error?: string
  courseName?: string
}

// Learning path view types (adapted from nmet-prephub)

export interface PathSection {
  id: string
  title: string
  description?: string
  nodes: PathNode[]
}

export interface PathNode {
  id: string
  title: string
  description?: string
  status: "locked" | "available" | "in_progress" | "completed"
  labs: PathLab[]
  estimatedMinutes?: number
}

export interface PathLab {
  id: string
  title: string
  description?: string
  difficulty: "beginner" | "intermediate" | "advanced"
  estimatedMinutes?: number
  capsules: PathCapsule[]
}

export interface PathCapsule {
  id: string
  title: string
  type: string
  questionCount: number
}

// Interactive Lab types for capsule/item pages

export interface InteractiveLab {
  id: string
  title: string
  description?: string
  category?: string
  difficulty?: string
  estimatedTime?: number
  capsules: LabCapsule[]
}

export interface LabCapsule {
  id: string
  title: string
  description?: string
  type: string
  questions: Question[]
  clinicalSituation?: string | object
  isCompleted: boolean
  score?: number
  timeSpent?: number
  reasoningTimeSpent?: number
}

export interface Question {
  id: string
  text: string
  category?: string
  difficulty?: string
  itemType?: 'MCQ' | 'MR' | 'OPTIONS_TABLE' | 'BUILD_LIST' | 'DRAG_DROP'
  multipleCorrect?: boolean
  correctAnswersCount?: number
  answerOptions: QuestionAnswerOption[]
  reasoning?: {
    objective?: string
    analysis?: string
    conclusion?: string
    rationale?: string
  }
  rows?: { text: string; correctColumn: string }[]
  correctOrder?: string[]
}

export interface QuestionAnswerOption {
  id: string
  text: string
  isCorrect: boolean
  explanation?: string
}

// Progress tracking types (stored in localStorage)
export interface CapsuleProgress {
  capsuleId: string
  isCompleted: boolean
  score?: number
  timeSpent?: number
  reasoningTimeSpent?: number
  answers?: Record<string, string | string[]>
  completedAt?: string
}
