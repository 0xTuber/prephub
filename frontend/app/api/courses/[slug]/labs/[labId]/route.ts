import { NextRequest, NextResponse } from 'next/server'
import { readFileSync, existsSync } from 'fs'
import path from 'path'
import type { CourseSkeleton, Lab, InteractiveLab, LabCapsule, Question, QuestionAnswerOption } from '@/lib/types'

const DATA_DIR = path.join(process.cwd(), '..', 'data')

function transformLabToInteractive(lab: Lab, labIndex: number): InteractiveLab {
  return {
    id: lab.lab_id || `lab_${labIndex}`,
    title: lab.title,
    description: lab.description || lab.objective,
    category: lab.lab_type,
    difficulty: lab.lab_type === 'challenge' ? 'advanced' : lab.lab_type === 'guided' ? 'beginner' : 'intermediate',
    estimatedTime: lab.estimated_duration_minutes,
    capsules: lab.capsules?.map((capsule, capsuleIndex) => {
      const questions: Question[] = capsule.items?.map((item, itemIndex) => ({
        id: item.item_id || `item_${capsuleIndex}_${itemIndex}`,
        text: item.question_stem || item.stem || '',
        category: capsule.capsule_type,
        difficulty: item.difficulty,
        itemType: (item.question_type?.toUpperCase() as Question['itemType']) || 'MCQ',
        multipleCorrect: item.question_type === 'MR' || (Array.isArray(item.correct_answer) && item.correct_answer.length > 1),
        correctAnswersCount: Array.isArray(item.correct_answer) ? item.correct_answer.length : 1,
        answerOptions: item.answer_options?.map((opt, optIndex) => ({
          id: opt.option_id || `opt_${capsuleIndex}_${itemIndex}_${optIndex}`,
          text: opt.option_text,
          isCorrect: opt.is_correct || false,
          explanation: opt.explanation
        })) || [],
        reasoning: item.explanation ? {
          rationale: item.explanation
        } : undefined
      })) || []

      return {
        id: capsule.capsule_id || `capsule_${labIndex}_${capsuleIndex}`,
        title: capsule.title,
        description: capsule.description || capsule.learning_goal,
        type: capsule.capsule_type || 'content',
        questions,
        clinicalSituation: undefined,
        isCompleted: false, // Will be set from localStorage on client
        score: undefined,
        timeSpent: undefined,
        reasoningTimeSpent: undefined
      }
    }) || []
  }
}

export async function GET(
  request: NextRequest,
  { params }: { params: Promise<{ slug: string; labId: string }> }
) {
  const { slug, labId } = await params
  const decodedSlug = decodeURIComponent(slug)
  const skeletonPath = path.join(DATA_DIR, decodedSlug, 'course_skeleton.json')

  if (!existsSync(skeletonPath)) {
    return NextResponse.json(
      { error: 'Course not found' },
      { status: 404 }
    )
  }

  try {
    const content = readFileSync(skeletonPath, 'utf-8')
    const skeleton: CourseSkeleton = JSON.parse(content)

    // Find the lab in the skeleton
    let foundLab: Lab | undefined
    let labIndex = 0

    skeleton.domain_modules?.forEach((module) => {
      module.topics?.forEach((topic) => {
        topic.subtopics?.forEach((subtopic) => {
          subtopic.labs?.forEach((lab, idx) => {
            const currentLabId = lab.lab_id || `lab_${labIndex}`
            if (currentLabId === labId) {
              foundLab = lab
            }
            labIndex++
          })
        })
      })
    })

    if (!foundLab) {
      return NextResponse.json(
        { error: 'Lab not found' },
        { status: 404 }
      )
    }

    const interactiveLab = transformLabToInteractive(foundLab, 0)

    return NextResponse.json({
      lab: interactiveLab,
      courseName: skeleton.certification_name
    })
  } catch (error) {
    console.error('Error reading lab:', error)
    return NextResponse.json(
      { error: 'Failed to read lab' },
      { status: 500 }
    )
  }
}
