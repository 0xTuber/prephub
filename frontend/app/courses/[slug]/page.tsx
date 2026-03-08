'use client'

import { useEffect, useState, use } from 'react'
import Link from 'next/link'
import { useTheme } from 'next-themes'
import { ArrowLeft, BookOpen, Moon, Sun, Loader2 } from 'lucide-react'
import { Button } from '@/components/ui/button'
import PathNavigator from '@/components/learning-path/PathNavigator'
import type { CourseSkeleton, PathSection, PathNode, PathLab, PathCapsule } from '@/lib/types'

// Transform course skeleton to learning path format
function transformSkeletonToPath(skeleton: CourseSkeleton): PathSection[] {
  const sections: PathSection[] = []

  skeleton.domain_modules?.forEach((module, moduleIndex) => {
    const nodes: PathNode[] = []

    module.topics?.forEach((topic, topicIndex) => {
      topic.subtopics?.forEach((subtopic, subtopicIndex) => {
        const labs: PathLab[] = subtopic.labs?.map((lab, labIndex) => {
          const capsules: PathCapsule[] = lab.capsules?.map((capsule, capsuleIndex) => ({
            id: capsule.capsule_id || `capsule_${labIndex}_${capsuleIndex}`,
            title: capsule.title,
            type: capsule.capsule_type,
            questionCount: capsule.items?.length || 0,
          })) || []

          return {
            id: lab.lab_id || `lab_${labIndex}`,
            title: lab.title,
            description: lab.description,
            difficulty: (lab.lab_type === 'challenge' ? 'advanced' : lab.lab_type === 'guided' ? 'beginner' : 'intermediate') as 'beginner' | 'intermediate' | 'advanced',
            estimatedMinutes: lab.estimated_duration_minutes,
            capsules,
          }
        }) || []

        // Create a node for each subtopic
        nodes.push({
          id: `node_${moduleIndex}_${topicIndex}_${subtopicIndex}`,
          title: subtopic.name,
          description: subtopic.description,
          status: nodes.length === 0 ? 'available' : 'available', // All accessible (no paywall)
          labs,
          estimatedMinutes: labs.reduce((sum, lab) => sum + (lab.estimatedMinutes || 0), 0),
        })
      })
    })

    if (nodes.length > 0) {
      sections.push({
        id: `section_${moduleIndex}`,
        title: module.domain_name,
        description: module.overview,
        nodes,
      })
    }
  })

  return sections
}

interface CoursePageProps {
  params: Promise<{ slug: string }>
}

export default function CoursePage({ params }: CoursePageProps) {
  const { slug } = use(params)
  const { theme, setTheme } = useTheme()
  const [skeleton, setSkeleton] = useState<CourseSkeleton | null>(null)
  const [sections, setSections] = useState<PathSection[]>([])
  const [isLoading, setIsLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    async function fetchCourse() {
      try {
        const response = await fetch(`/api/courses/${slug}`)
        if (!response.ok) {
          throw new Error('Course not found')
        }
        const data = await response.json()
        setSkeleton(data.skeleton)
        setSections(transformSkeletonToPath(data.skeleton))
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Failed to load course')
      } finally {
        setIsLoading(false)
      }
    }

    fetchCourse()
  }, [slug])

  if (isLoading) {
    return (
      <div className="min-h-screen bg-gradient-to-b from-blue-50 to-white dark:from-gray-900 dark:to-gray-950 flex items-center justify-center">
        <Loader2 className="h-8 w-8 animate-spin text-blue-600" />
      </div>
    )
  }

  if (error || !skeleton) {
    return (
      <div className="min-h-screen bg-gradient-to-b from-blue-50 to-white dark:from-gray-900 dark:to-gray-950 flex flex-col items-center justify-center gap-4">
        <p className="text-lg text-gray-500 dark:text-gray-400">{error || 'Course not found'}</p>
        <Link href="/">
          <Button variant="outline" className="border-gray-200 dark:border-gray-700">
            <ArrowLeft className="h-4 w-4 mr-2" />
            Back to Home
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
            <Link href="/">
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
                <h1 className="text-lg font-bold text-gray-900 dark:text-white">{skeleton.certification_name}</h1>
                {skeleton.exam_code && (
                  <p className="text-xs text-gray-500 dark:text-gray-400">{skeleton.exam_code}</p>
                )}
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

      <main className="container mx-auto px-4 py-12">
        {/* Course Overview */}
        {skeleton.overview && (
          <section className="text-center mb-16">
            <h2 className="text-3xl md:text-4xl font-bold mb-4 text-gray-900 dark:text-white">
              {skeleton.certification_name}
            </h2>
            {skeleton.overview.course_description && (
              <p className="text-gray-600 dark:text-gray-400 max-w-3xl mx-auto">
                {skeleton.overview.course_description}
              </p>
            )}
            {skeleton.overview.total_estimated_study_hours && (
              <p className="text-sm text-gray-500 dark:text-gray-500 mt-3">
                Estimated study time: {skeleton.overview.total_estimated_study_hours} hours
              </p>
            )}
          </section>
        )}

        {/* Learning Path */}
        <section>
          <PathNavigator sections={sections} courseSlug={slug} />
        </section>
      </main>
    </div>
  )
}
