import { NextRequest, NextResponse } from 'next/server'
import { readdirSync, readFileSync, existsSync } from 'fs'
import path from 'path'

const DATA_DIR = path.join(process.cwd(), '..', 'data')

interface Question {
  id: string
  text: string
  options: { id: string; text: string; isCorrect: boolean }[]
  labTitle: string
  capsuleTitle: string
}

function getAllQuestionsFromCourses(): Question[] {
  const questions: Question[] = []

  if (!existsSync(DATA_DIR)) {
    return questions
  }

  const entries = readdirSync(DATA_DIR, { withFileTypes: true })

  for (const entry of entries) {
    if (entry.isDirectory()) {
      const skeletonPath = path.join(DATA_DIR, entry.name, 'course_skeleton.json')

      if (existsSync(skeletonPath)) {
        try {
          const content = readFileSync(skeletonPath, 'utf-8')
          const skeleton = JSON.parse(content)

          // Extract questions from skeleton
          skeleton.domain_modules?.forEach((module: any) => {
            module.topics?.forEach((topic: any) => {
              topic.subtopics?.forEach((subtopic: any) => {
                subtopic.labs?.forEach((lab: any) => {
                  lab.capsules?.forEach((capsule: any) => {
                    capsule.items?.forEach((item: any, itemIndex: number) => {
                      if (item.question_stem) {
                        const questionId = `${lab.lab_id || 'lab'}_${capsule.capsule_id || 'cap'}_${itemIndex}`

                        questions.push({
                          id: questionId,
                          text: item.question_stem,
                          options: (item.answer_options || []).map((opt: any, optIndex: number) => ({
                            id: `${questionId}_opt_${optIndex}`,
                            text: opt.option_text,
                            isCorrect: opt.is_correct
                          })),
                          labTitle: lab.title || 'Lab',
                          capsuleTitle: capsule.title || 'Capsule'
                        })
                      }
                    })
                  })
                })
              })
            })
          })
        } catch (err) {
          console.error(`Error parsing skeleton for ${entry.name}:`, err)
        }
      }
    }
  }

  return questions
}

export async function GET(request: NextRequest) {
  const searchParams = request.nextUrl.searchParams
  const count = parseInt(searchParams.get('count') || '10', 10)

  const allQuestions = getAllQuestionsFromCourses()

  if (allQuestions.length === 0) {
    return NextResponse.json(
      { error: 'No questions available' },
      { status: 404 }
    )
  }

  // Shuffle and take requested count
  const shuffled = allQuestions.sort(() => Math.random() - 0.5)
  const selected = shuffled.slice(0, Math.min(count, shuffled.length))

  return NextResponse.json({
    questions: selected,
    totalAvailable: allQuestions.length
  })
}
