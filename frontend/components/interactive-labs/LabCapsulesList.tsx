"use client"

import { useState, useEffect } from "react"
import Link from "next/link"
import { useRouter } from "next/navigation"
import { Card, CardContent } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Progress } from "@/components/ui/progress"
import { Badge } from "@/components/ui/badge"
import { CheckCircle, ArrowRight, BookOpen, FileQuestion } from "lucide-react"
import type { InteractiveLab, LabCapsule } from "@/lib/types"
import { isCapsuleCompleted, getCapsuleScore } from "@/lib/progress"
import LabResultsSummary from "./LabResultsSummary"

interface LabCapsulesListProps {
  lab: InteractiveLab
  courseSlug: string
}

export default function LabCapsulesList({ lab, courseSlug }: LabCapsulesListProps) {
  const router = useRouter()
  const [capsuleStatuses, setCapsuleStatuses] = useState<Record<string, { completed: boolean; score?: number }>>({})

  // Load progress from localStorage on mount
  useEffect(() => {
    const statuses: Record<string, { completed: boolean; score?: number }> = {}
    lab.capsules.forEach(capsule => {
      statuses[capsule.id] = {
        completed: isCapsuleCompleted(capsule.id),
        score: getCapsuleScore(capsule.id)
      }
    })
    setCapsuleStatuses(statuses)
  }, [lab.capsules])

  // Calculate progress
  const completedCapsules = Object.values(capsuleStatuses).filter(s => s.completed).length
  const progress = lab.capsules.length > 0 ? (completedCapsules / lab.capsules.length) * 100 : 0
  const isLabCompleted = progress === 100

  // Get the icon for a capsule based on its type
  const getCapsuleIcon = (type: string) => {
    switch (type) {
      case "quiz":
        return <FileQuestion className="h-4 w-4 text-purple-500" />
      default:
        return <BookOpen className="h-4 w-4 text-blue-500" />
    }
  }

  // Format capsule type for display
  const formatCapsuleType = (type: string) => {
    return type.charAt(0).toUpperCase() + type.slice(1).replace(/_/g, ' ')
  }

  // Get the next incomplete capsule
  const getNextIncompleteCapsule = () => {
    return lab.capsules.find((capsule) => !capsuleStatuses[capsule.id]?.completed)
  }

  // Handle continue button click
  const handleContinue = () => {
    const nextCapsule = getNextIncompleteCapsule()
    if (nextCapsule) {
      router.push(`/courses/${courseSlug}/labs/${lab.id}/capsules/${nextCapsule.id}`)
    }
  }

  // Create enhanced capsule data with progress
  const enhancedCapsules = lab.capsules.map(capsule => ({
    ...capsule,
    isCompleted: capsuleStatuses[capsule.id]?.completed || false,
    score: capsuleStatuses[capsule.id]?.score
  }))

  const enhancedLab = { ...lab, capsules: enhancedCapsules }

  return (
    <div className="space-y-3">
      {/* Show lab results summary if completed */}
      {isLabCompleted && <LabResultsSummary lab={enhancedLab} />}

      <div className="flex flex-col md:flex-row gap-2 items-start md:items-center justify-between">
        <div className="space-y-0.5">
          <h2 className="text-lg font-semibold text-gray-900 dark:text-white">Lab Capsules</h2>
          <p className="text-xs text-gray-600 dark:text-gray-400">
            Complete all capsules to finish {lab.title.toLowerCase()}
          </p>
        </div>

        <div className="flex items-center gap-2">
          <div className="flex items-center gap-2">
            <Progress value={progress} className="w-24 h-1.5" />
            <span className="text-xs text-gray-600 dark:text-gray-400">
              {completedCapsules}/{lab.capsules.length}
            </span>
          </div>

          <Button onClick={handleContinue} disabled={isLabCompleted} size="sm">
            {isLabCompleted
              ? "Completed"
              : completedCapsules === 0
                ? "Start Lab"
                : "Continue"}
            {!isLabCompleted && <ArrowRight className="ml-1 h-3 w-3" />}
          </Button>
        </div>
      </div>

      <div className="space-y-2">
        {lab.capsules.map((capsule, index) => {
          const status = capsuleStatuses[capsule.id]
          const isCompleted = status?.completed || false
          const score = status?.score

          return (
            <Link
              key={capsule.id}
              href={`/courses/${courseSlug}/labs/${lab.id}/capsules/${capsule.id}`}
              className="block"
            >
              <Card
                className={`transition-all hover:bg-gray-50 dark:hover:bg-gray-800/40 ${
                  isCompleted
                    ? "border-emerald-200 bg-emerald-50/50 dark:bg-emerald-900/10 dark:border-emerald-800"
                    : "border-gray-200 dark:border-gray-700"
                }`}
              >
                <CardContent className="p-2">
                  <div className="flex items-center gap-3">
                    <div className="flex h-8 w-8 items-center justify-center rounded-full bg-gray-100 dark:bg-gray-800 flex-shrink-0">
                      {isCompleted ? (
                        <CheckCircle className="h-4 w-4 text-emerald-500" />
                      ) : (
                        <div className="flex h-5 w-5 items-center justify-center rounded-full bg-blue-600 text-white text-xs font-medium">
                          {index + 1}
                        </div>
                      )}
                    </div>

                    <div className="flex-1">
                      <div className="flex items-center justify-between">
                        <div>
                          <div className="flex items-center gap-1.5 mb-0.5">
                            {getCapsuleIcon(capsule.type)}
                            <Badge variant="outline" className="text-xs">
                              {formatCapsuleType(capsule.type)}
                            </Badge>
                            {isCompleted && (
                              <Badge variant="secondary" className="bg-emerald-100 text-emerald-800 dark:bg-emerald-900/30 dark:text-emerald-200 text-xs">
                                <CheckCircle className="h-3 w-3 mr-1" />
                                Completed
                              </Badge>
                            )}
                          </div>
                          <h3 className="font-medium text-sm text-gray-900 dark:text-white">{capsule.title}</h3>
                          {capsule.description && (
                            <p className="text-xs text-gray-600 dark:text-gray-400 mt-0.5 line-clamp-1">
                              {capsule.description}
                            </p>
                          )}
                          {isCompleted && score !== undefined && (
                            <div className="mt-1">
                              <Badge variant="outline" className="bg-blue-50 text-blue-700 dark:bg-blue-900/30 dark:text-blue-200 text-xs">
                                Score: {score}%
                              </Badge>
                            </div>
                          )}
                        </div>

                        <div className="flex items-center gap-2 ml-2">
                          {isCompleted ? (
                            <Badge variant="outline" className="bg-green-100 text-green-800 border-green-200 text-xs dark:bg-green-900/30 dark:text-green-200">
                              Review
                            </Badge>
                          ) : (
                            <ArrowRight className="h-4 w-4 text-gray-400" />
                          )}
                        </div>
                      </div>
                    </div>
                  </div>
                </CardContent>
              </Card>
            </Link>
          )
        })}
      </div>
    </div>
  )
}
