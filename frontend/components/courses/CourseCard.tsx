'use client'

import Link from 'next/link'
import { BookOpen, Loader2, CheckCircle, AlertCircle, ArrowRight } from 'lucide-react'
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import type { Course } from '@/lib/types'

interface CourseCardProps {
  course: Course
}

export default function CourseCard({ course }: CourseCardProps) {
  const statusConfig = {
    processing: {
      icon: Loader2,
      iconClass: "animate-spin text-yellow-500",
      badge: "warning" as const,
      label: "Processing",
    },
    ready: {
      icon: CheckCircle,
      iconClass: "text-green-500",
      badge: "success" as const,
      label: "Ready",
    },
    error: {
      icon: AlertCircle,
      iconClass: "text-red-500",
      badge: "destructive" as const,
      label: "Error",
    },
  }

  const config = statusConfig[course.status]
  const StatusIcon = config.icon

  return (
    <Card className="bg-white dark:bg-gray-800 border-gray-200 dark:border-gray-700 hover:shadow-lg hover:shadow-gray-200/50 dark:hover:shadow-gray-900/50 transition-all duration-300 rounded-2xl overflow-hidden">
      <CardHeader className="pb-3">
        <div className="flex items-start justify-between">
          <div className="flex items-center gap-3">
            <div className="p-2.5 bg-blue-50 dark:bg-blue-900/30 rounded-xl">
              <BookOpen className="h-5 w-5 text-blue-600 dark:text-blue-400" />
            </div>
            <div>
              <CardTitle className="text-lg text-gray-900 dark:text-white">{course.name}</CardTitle>
              <CardDescription className="mt-1 text-gray-500 dark:text-gray-400">
                {course.moduleCount} modules · {course.labCount} labs
              </CardDescription>
            </div>
          </div>
          <Badge variant={config.badge} className="rounded-lg">
            <StatusIcon className={`h-3 w-3 mr-1 ${config.iconClass}`} />
            {config.label}
          </Badge>
        </div>
      </CardHeader>
      <CardContent>
        <div className="flex items-center justify-between">
          <span className="text-sm text-gray-500 dark:text-gray-400">
            Created {new Date(course.createdAt).toLocaleDateString()}
          </span>
          {course.status === 'ready' && (
            <Link href={`/courses/${course.slug}`}>
              <Button size="sm" className="bg-blue-600 hover:bg-blue-700 text-white rounded-lg">
                View Course
                <ArrowRight className="h-4 w-4 ml-1" />
              </Button>
            </Link>
          )}
        </div>
      </CardContent>
    </Card>
  )
}
