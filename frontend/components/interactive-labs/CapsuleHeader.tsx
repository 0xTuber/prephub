"use client"

import Link from "next/link"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { Progress } from "@/components/ui/progress"
import { ArrowLeft, BookOpen, FileQuestion, CheckCircle } from "lucide-react"
import type { InteractiveLab, LabCapsule } from "@/lib/types"

interface CapsuleHeaderProps {
  lab: InteractiveLab
  capsule: LabCapsule
  courseSlug: string
  completedCount: number
}

export default function CapsuleHeader({ lab, capsule, courseSlug, completedCount }: CapsuleHeaderProps) {
  // Get the icon for a capsule based on its type
  const getCapsuleIcon = (type: string) => {
    switch (type) {
      case "quiz":
        return <FileQuestion className="h-5 w-5 text-purple-500" />
      default:
        return <BookOpen className="h-5 w-5 text-blue-500" />
    }
  }

  // Format capsule type for display
  const formatCapsuleType = (type: string) => {
    return type.charAt(0).toUpperCase() + type.slice(1).replace(/_/g, ' ')
  }

  // Calculate progress for the lab
  const totalCapsules = lab.capsules.length
  const progressPercentage = totalCapsules > 0 ? Math.round((completedCount / totalCapsules) * 100) : 0

  return (
    <div className="space-y-4">
      {/* Progress Bar */}
      <div className="bg-white dark:bg-gray-800 rounded-lg p-3 border border-gray-200 dark:border-gray-700">
        <div className="flex items-center justify-between mb-2">
          <span className="text-sm font-medium text-gray-700 dark:text-gray-300">{lab.title}</span>
          <span className="text-sm text-gray-500 dark:text-gray-400">
            {completedCount}/{totalCapsules} completed
          </span>
        </div>
        <Progress value={progressPercentage} className="h-2" />
      </div>

      <div className="flex items-center gap-2">
        <Button variant="ghost" size="sm" asChild>
          <Link href={`/courses/${courseSlug}/labs/${lab.id}`} className="flex items-center gap-1">
            <ArrowLeft className="h-4 w-4" />
            Back to Lab
          </Link>
        </Button>
      </div>

      <div className="flex flex-col md:flex-row gap-4 md:items-center justify-between">
        <div>
          <div className="flex items-center gap-2 mb-2 flex-wrap">
            {getCapsuleIcon(capsule.type)}
            <Badge variant="outline">{formatCapsuleType(capsule.type)} Capsule</Badge>
            {capsule.isCompleted && (
              <Badge variant="secondary" className="bg-green-100 text-green-800 dark:bg-green-900/20 dark:text-green-300">
                <CheckCircle className="h-3 w-3 mr-1" />
                Completed
              </Badge>
            )}
          </div>
          <h1 className="text-2xl font-bold text-gray-900 dark:text-white">{capsule.title}</h1>
          {capsule.description && (
            <p className="text-gray-600 dark:text-gray-400 mt-1">{capsule.description}</p>
          )}
        </div>

        <div className="flex items-center gap-4">
          {capsule.isCompleted && capsule.score !== undefined && (
            <div className="flex items-center gap-1">
              <Badge variant="outline" className="bg-blue-50 text-blue-700 dark:bg-blue-900/20 dark:text-blue-300">
                Score: {capsule.score}%
              </Badge>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
