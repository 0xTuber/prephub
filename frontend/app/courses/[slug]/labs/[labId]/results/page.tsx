'use client'

import { useEffect, useState, use } from 'react'
import Link from 'next/link'
import { useRouter } from 'next/navigation'
import { useTheme } from 'next-themes'
import { ArrowLeft, Moon, Sun, Loader2, Trophy } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Card, CardHeader, CardTitle } from '@/components/ui/card'
import LabResultsSummary from '@/components/interactive-labs/LabResultsSummary'
import type { InteractiveLab, LabCapsule } from '@/lib/types'
import { isCapsuleCompleted, getCapsuleScore, getCapsuleProgress } from '@/lib/progress'

interface ResultsPageProps {
  params: Promise<{ slug: string; labId: string }>
}

export default function ResultsPage({ params }: ResultsPageProps) {
  const { slug, labId } = use(params)
  const router = useRouter()
  const { theme, setTheme } = useTheme()
  const [lab, setLab] = useState<InteractiveLab | null>(null)
  const [courseName, setCourseName] = useState<string>('')
  const [isLoading, setIsLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    async function fetchData() {
      try {
        const response = await fetch(`/api/courses/${slug}/labs/${labId}`)
        if (!response.ok) {
          throw new Error('Lab not found')
        }
        const data = await response.json()

        // Enhance lab with progress data
        const enhancedCapsules = data.lab.capsules.map((capsule: LabCapsule) => {
          const progress = getCapsuleProgress(capsule.id)
          return {
            ...capsule,
            isCompleted: progress?.isCompleted || false,
            score: progress?.score,
            timeSpent: progress?.timeSpent,
            reasoningTimeSpent: progress?.reasoningTimeSpent
          }
        })

        const enhancedLab = {
          ...data.lab,
          capsules: enhancedCapsules
        }

        // Check if lab is completed
        const allCompleted = enhancedCapsules.every((c: LabCapsule) => c.isCompleted)
        if (!allCompleted) {
          // Redirect to lab page if not completed
          router.push(`/courses/${slug}/labs/${labId}`)
          return
        }

        setLab(enhancedLab)
        setCourseName(data.courseName)
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Failed to load results')
      } finally {
        setIsLoading(false)
      }
    }

    fetchData()
  }, [slug, labId, router])

  if (isLoading) {
    return (
      <div className="min-h-screen bg-gradient-to-b from-blue-50 to-white dark:from-gray-900 dark:to-gray-950 flex items-center justify-center">
        <Loader2 className="h-8 w-8 animate-spin text-blue-600" />
      </div>
    )
  }

  if (error || !lab) {
    return (
      <div className="min-h-screen bg-gradient-to-b from-blue-50 to-white dark:from-gray-900 dark:to-gray-950 flex flex-col items-center justify-center gap-4">
        <p className="text-lg text-gray-500 dark:text-gray-400">{error || 'Results not found'}</p>
        <Link href={`/courses/${slug}`}>
          <Button variant="outline" className="border-gray-200 dark:border-gray-700">
            <ArrowLeft className="h-4 w-4 mr-2" />
            Back to Course
          </Button>
        </Link>
      </div>
    )
  }

  return (
    <div className="min-h-screen bg-gradient-to-b from-blue-50 to-white dark:from-gray-900 dark:to-gray-950">
      {/* Header */}
      <header className="bg-white/80 dark:bg-gray-900/80 backdrop-blur-md border-b border-gray-100 dark:border-gray-800 sticky top-0 z-50">
        <div className="container mx-auto px-4 py-4 flex items-center justify-between">
          <div className="flex items-center gap-4">
            <Link href={`/courses/${slug}`}>
              <Button variant="ghost" size="sm" className="text-gray-600 hover:text-gray-900 dark:text-gray-400 dark:hover:text-white">
                <ArrowLeft className="h-4 w-4 mr-2" />
                Back to Course
              </Button>
            </Link>
            <div className="flex items-center gap-3">
              <div className="p-2 bg-gradient-to-br from-yellow-500 to-amber-600 rounded-xl shadow-lg shadow-amber-500/20">
                <Trophy className="h-5 w-5 text-white" />
              </div>
              <div>
                <h1 className="text-lg font-bold text-gray-900 dark:text-white">Lab Results</h1>
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
        {/* Lab Completion Header */}
        <Card className="border-emerald-200 dark:border-emerald-800 bg-emerald-50/50 dark:bg-emerald-900/10">
          <CardHeader className="text-center pb-3">
            <div className="flex items-center justify-center gap-2 mb-2">
              <Trophy className="h-8 w-8 text-yellow-500" />
              <CardTitle className="text-2xl text-gray-900 dark:text-white">Lab Complete!</CardTitle>
            </div>
            <div className="text-lg font-semibold text-emerald-700 dark:text-emerald-300">
              Congratulations! You&apos;ve completed &quot;{lab.title}&quot;
            </div>
            <div className="text-sm text-gray-600 dark:text-gray-400">
              All {lab.capsules.length} capsules finished successfully
            </div>
          </CardHeader>
        </Card>

        {/* Lab Results Summary */}
        <LabResultsSummary lab={lab as any} />

        {/* Action Buttons */}
        <div className="flex flex-col sm:flex-row gap-3 justify-center">
          <Button asChild className="bg-blue-600 hover:bg-blue-700">
            <Link href={`/courses/${slug}`}>
              Back to Course
            </Link>
          </Button>
          <Button variant="outline" asChild>
            <Link href={`/courses/${slug}/labs/${labId}`}>
              Review Lab
            </Link>
          </Button>
        </div>
      </main>
    </div>
  )
}
