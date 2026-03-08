'use client'

import { useEffect, useState, use } from 'react'
import Link from 'next/link'
import { useTheme } from 'next-themes'
import { ArrowLeft, Moon, Sun, Loader2, BookOpen } from 'lucide-react'
import { Button } from '@/components/ui/button'
import LabHeader from '@/components/interactive-labs/LabHeader'
import LabCapsulesList from '@/components/interactive-labs/LabCapsulesList'
import type { InteractiveLab } from '@/lib/types'

interface LabPageProps {
  params: Promise<{ slug: string; labId: string }>
}

export default function LabPage({ params }: LabPageProps) {
  const { slug, labId } = use(params)
  const { theme, setTheme } = useTheme()
  const [lab, setLab] = useState<InteractiveLab | null>(null)
  const [courseName, setCourseName] = useState<string>('')
  const [isLoading, setIsLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    async function fetchLab() {
      try {
        const response = await fetch(`/api/courses/${slug}/labs/${labId}`)
        if (!response.ok) {
          throw new Error('Lab not found')
        }
        const data = await response.json()
        setLab(data.lab)
        setCourseName(data.courseName)
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Failed to load lab')
      } finally {
        setIsLoading(false)
      }
    }

    fetchLab()
  }, [slug, labId])

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
        <p className="text-lg text-gray-500 dark:text-gray-400">{error || 'Lab not found'}</p>
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
                Back
              </Button>
            </Link>
            <div className="flex items-center gap-3">
              <div className="p-2 bg-gradient-to-br from-blue-500 to-blue-600 rounded-xl shadow-lg shadow-blue-500/20">
                <BookOpen className="h-5 w-5 text-white" />
              </div>
              <div>
                <h1 className="text-lg font-bold text-gray-900 dark:text-white">{lab.title}</h1>
                <p className="text-xs text-gray-500 dark:text-gray-400">{courseName}</p>
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
        <LabHeader lab={lab} courseSlug={slug} />
        <LabCapsulesList lab={lab} courseSlug={slug} />
      </main>
    </div>
  )
}
