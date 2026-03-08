'use client'

import { useEffect, useState, use } from 'react'
import Link from 'next/link'
import { useTheme } from 'next-themes'
import { ArrowLeft, Moon, Sun, Loader2, FileQuestion } from 'lucide-react'
import { Button } from '@/components/ui/button'
import CapsuleHeader from '@/components/interactive-labs/CapsuleHeader'
import CapsuleContent from '@/components/interactive-labs/CapsuleContent'
import type { InteractiveLab, LabCapsule } from '@/lib/types'
import { isCapsuleCompleted, getCapsuleScore, getCapsuleAnswers } from '@/lib/progress'

interface CapsulePageProps {
  params: Promise<{ slug: string; labId: string; capsuleId: string }>
}

export default function CapsulePage({ params }: CapsulePageProps) {
  const { slug, labId, capsuleId } = use(params)
  const { theme, setTheme } = useTheme()
  const [lab, setLab] = useState<InteractiveLab | null>(null)
  const [capsule, setCapsule] = useState<LabCapsule | null>(null)
  const [courseName, setCourseName] = useState<string>('')
  const [isLoading, setIsLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [completedCount, setCompletedCount] = useState(0)

  useEffect(() => {
    async function fetchData() {
      try {
        const response = await fetch(`/api/courses/${slug}/labs/${labId}`)
        if (!response.ok) {
          throw new Error('Lab not found')
        }
        const data = await response.json()
        setLab(data.lab)
        setCourseName(data.courseName)

        // Find the capsule
        const foundCapsule = data.lab.capsules.find((c: LabCapsule) => c.id === capsuleId)
        if (!foundCapsule) {
          throw new Error('Capsule not found')
        }

        // Enhance capsule with progress data
        const isCompleted = isCapsuleCompleted(capsuleId)
        const score = getCapsuleScore(capsuleId)
        const answers = getCapsuleAnswers(capsuleId) || {}

        setCapsule({
          ...foundCapsule,
          isCompleted,
          score
        })

        // Calculate completed count
        const count = data.lab.capsules.filter((c: LabCapsule) => isCapsuleCompleted(c.id)).length
        setCompletedCount(count)
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Failed to load capsule')
      } finally {
        setIsLoading(false)
      }
    }

    fetchData()
  }, [slug, labId, capsuleId])

  if (isLoading) {
    return (
      <div className="min-h-screen bg-gradient-to-b from-blue-50 to-white dark:from-gray-900 dark:to-gray-950 flex items-center justify-center">
        <Loader2 className="h-8 w-8 animate-spin text-blue-600" />
      </div>
    )
  }

  if (error || !lab || !capsule) {
    return (
      <div className="min-h-screen bg-gradient-to-b from-blue-50 to-white dark:from-gray-900 dark:to-gray-950 flex flex-col items-center justify-center gap-4">
        <p className="text-lg text-gray-500 dark:text-gray-400">{error || 'Capsule not found'}</p>
        <Link href={`/courses/${slug}/labs/${labId}`}>
          <Button variant="outline" className="border-gray-200 dark:border-gray-700">
            <ArrowLeft className="h-4 w-4 mr-2" />
            Back to Lab
          </Button>
        </Link>
      </div>
    )
  }

  // Find next/prev capsule IDs
  const currentIndex = lab.capsules.findIndex(c => c.id === capsuleId)
  let nextCapsuleId: string | undefined
  let prevCapsuleId: string | undefined

  // Find next incomplete capsule
  for (let i = currentIndex + 1; i < lab.capsules.length; i++) {
    if (!isCapsuleCompleted(lab.capsules[i].id)) {
      nextCapsuleId = lab.capsules[i].id
      break
    }
  }

  // Find prev incomplete capsule
  for (let i = currentIndex - 1; i >= 0; i--) {
    if (!isCapsuleCompleted(lab.capsules[i].id)) {
      prevCapsuleId = lab.capsules[i].id
      break
    }
  }

  // Check if lab is completed
  const isLabCompleted = lab.capsules.every(c => isCapsuleCompleted(c.id))

  // Get user selected answers if capsule is completed
  const userSelectedAnswers = getCapsuleAnswers(capsuleId) || {}

  return (
    <div className="min-h-screen bg-gradient-to-b from-blue-50 to-white dark:from-gray-900 dark:to-gray-950">
      {/* Header */}
      <header className="bg-white/80 dark:bg-gray-900/80 backdrop-blur-md border-b border-gray-100 dark:border-gray-800 sticky top-0 z-50">
        <div className="container mx-auto px-4 py-4 flex items-center justify-between">
          <div className="flex items-center gap-4">
            <Link href={`/courses/${slug}/labs/${labId}`}>
              <Button variant="ghost" size="sm" className="text-gray-600 hover:text-gray-900 dark:text-gray-400 dark:hover:text-white">
                <ArrowLeft className="h-4 w-4 mr-2" />
                Back
              </Button>
            </Link>
            <div className="flex items-center gap-3">
              <div className="p-2 bg-gradient-to-br from-purple-500 to-purple-600 rounded-xl shadow-lg shadow-purple-500/20">
                <FileQuestion className="h-5 w-5 text-white" />
              </div>
              <div>
                <h1 className="text-lg font-bold text-gray-900 dark:text-white">{capsule.title}</h1>
                <p className="text-xs text-gray-500 dark:text-gray-400">{lab.title}</p>
              </div>
            </div>
          </div>
          <Button
            variant="ghost"
            size="icon"
            className="text-gray-600 hover:text-gray-900 dark:text-gray-400 dark:hover:text-white"
            onClick={() => setTheme(theme === 'dark' ? 'light' : 'dark')}
          >
            <Sun className="h-5 w-5 rotate-0 scale-100 transition-all dark:-rotate-90 dark:scale-0" />
            <Moon className="absolute h-5 w-5 rotate-90 scale-0 transition-all dark:rotate-0 dark:scale-100" />
          </Button>
        </div>
      </header>

      <main className="container mx-auto px-4 py-6 space-y-6">
        <CapsuleHeader
          lab={lab}
          capsule={capsule}
          courseSlug={slug}
          completedCount={completedCount}
        />
        <CapsuleContent
          questions={capsule.questions}
          labId={labId}
          capsuleId={capsuleId}
          courseSlug={slug}
          nextCapsuleId={nextCapsuleId}
          prevCapsuleId={prevCapsuleId}
          isCompleted={capsule.isCompleted}
          userSelectedAnswers={userSelectedAnswers}
          isLabCompleted={isLabCompleted}
          score={capsule.score}
          timeSpent={capsule.timeSpent}
        />
      </main>
    </div>
  )
}
