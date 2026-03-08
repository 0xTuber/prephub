'use client'

import { useState, useEffect, useCallback } from 'react'
import { BookOpen, Moon, Sun, Plus, Brain } from 'lucide-react'
import Link from 'next/link'
import { useTheme } from 'next-themes'
import { Button } from '@/components/ui/button'
import CourseGrid from '@/components/courses/CourseGrid'
import GenerationBanner from '@/components/GenerationBanner'
import CreateCourseDialog from '@/components/CreateCourseDialog'
import ReviewWidget from '@/components/ReviewWidget'
import type { Course, GenerationJob } from '@/lib/types'

export default function HomePage() {
  const { theme, setTheme } = useTheme()
  const [courses, setCourses] = useState<Course[]>([])
  const [isLoadingCourses, setIsLoadingCourses] = useState(true)
  const [generationJob, setGenerationJob] = useState<GenerationJob | null>(null)
  const [showCreateDialog, setShowCreateDialog] = useState(false)

  const fetchCourses = useCallback(async () => {
    try {
      const response = await fetch('/api/courses')
      if (response.ok) {
        const data = await response.json()
        setCourses(data.courses)
      }
    } catch (error) {
      console.error('Error fetching courses:', error)
    } finally {
      setIsLoadingCourses(false)
    }
  }, [])

  useEffect(() => {
    fetchCourses()
  }, [fetchCourses])

  useEffect(() => {
    if (!generationJob || generationJob.status === 'completed' || generationJob.status === 'error') {
      return
    }

    const pollInterval = setInterval(async () => {
      try {
        const response = await fetch(`/api/generate/${generationJob.jobId}`)
        if (response.ok) {
          const data = await response.json()
          setGenerationJob(data)

          if (data.status === 'completed') {
            fetchCourses()
          }
        }
      } catch (error) {
        console.error('Error polling job status:', error)
      }
    }, 2000)

    return () => clearInterval(pollInterval)
  }, [generationJob, fetchCourses])

  const handleGenerate = async (bookPath: string, courseName: string) => {
    try {
      const response = await fetch('/api/generate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          bookPath,
          certificationName: courseName,
        }),
      })

      if (response.ok) {
        const data = await response.json()
        setGenerationJob({ ...data, courseName })
      }
    } catch (error) {
      console.error('Error starting generation:', error)
    }
  }

  const handleDismissBanner = () => {
    setGenerationJob(null)
  }

  return (
    <div className="min-h-screen bg-gradient-to-b from-brand-bg-subtle to-white dark:from-gray-900 dark:to-gray-950">
      {/* Generation Progress Banner */}
      <GenerationBanner job={generationJob} onDismiss={handleDismissBanner} />

      {/* Header */}
      <header className="bg-white/80 dark:bg-gray-900/80 backdrop-blur-md border-b border-gray-100 dark:border-gray-800 sticky top-0 z-50">
        <div className="container mx-auto px-4 py-4 flex items-center justify-between">
          <Link href="/" className="flex items-center gap-3">
            <div className="p-2 bg-gradient-to-br from-blue-500 to-blue-600 rounded-xl shadow-lg shadow-blue-500/20">
              <BookOpen className="h-6 w-6 text-white" />
            </div>
            <h1 className="text-xl font-bold text-gray-900 dark:text-white">Course Builder</h1>
          </Link>
          <div className="flex items-center gap-2">
            <Link href="/review">
              <Button
                variant="outline"
                size="sm"
                className="text-gray-600 hover:text-gray-900 dark:text-gray-400 dark:hover:text-white border-gray-200 dark:border-gray-700"
              >
                <Brain className="h-4 w-4 mr-2" />
                Review Center
              </Button>
            </Link>
            <Button
              onClick={() => setShowCreateDialog(true)}
              className="bg-gradient-to-r from-blue-600 to-blue-500 hover:from-blue-700 hover:to-blue-600 text-white shadow-lg shadow-blue-500/25"
              size="sm"
            >
              <Plus className="h-4 w-4 mr-2" />
              New Course
            </Button>
            <Button
              variant="ghost"
              size="icon"
              className="text-gray-600 hover:text-gray-900 dark:text-gray-400 dark:hover:text-white"
              onClick={() => setTheme(theme === 'dark' ? 'light' : 'dark')}
            >
              <Sun className="h-5 w-5 rotate-0 scale-100 transition-all dark:-rotate-90 dark:scale-0" />
              <Moon className="absolute h-5 w-5 rotate-90 scale-0 transition-all dark:rotate-0 dark:scale-100" />
              <span className="sr-only">Toggle theme</span>
            </Button>
          </div>
        </div>
      </header>

      <main className="container mx-auto px-4 py-8 space-y-8">
        {/* Review Widget */}
        <ReviewWidget />

        {/* Courses Section */}
        <section>
          <div className="flex items-center justify-between mb-6">
            <h2 className="text-2xl font-bold text-gray-900 dark:text-white">Your Courses</h2>
            {courses.length > 0 && (
              <span className="text-sm text-gray-500 dark:text-gray-400">
                {courses.length} {courses.length === 1 ? 'course' : 'courses'}
              </span>
            )}
          </div>
          <CourseGrid courses={courses} isLoading={isLoadingCourses} />
        </section>
      </main>

      {/* Create Course Dialog */}
      <CreateCourseDialog
        open={showCreateDialog}
        onOpenChange={setShowCreateDialog}
        onGenerate={handleGenerate}
      />
    </div>
  )
}
