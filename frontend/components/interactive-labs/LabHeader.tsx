"use client"

import Link from "next/link"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { ArrowLeft, BarChart2 } from "lucide-react"
import type { InteractiveLab } from "@/lib/types"

interface LabHeaderProps {
  lab: InteractiveLab
  courseSlug: string
}

export default function LabHeader({ lab, courseSlug }: LabHeaderProps) {
  return (
    <div className="space-y-4">
      <div className="flex items-center gap-2">
        <Button variant="ghost" size="sm" asChild>
          <Link href={`/courses/${courseSlug}`} className="flex items-center gap-1">
            <ArrowLeft className="h-4 w-4" />
            Back to Course
          </Link>
        </Button>
      </div>

      <div className="flex flex-col md:flex-row gap-4 md:items-center justify-between">
        <div>
          <div className="flex flex-wrap gap-2 mb-2">
            {lab.category && (
              <Badge variant="outline">{lab.category}</Badge>
            )}
            <Badge
              variant="outline"
              className={
                lab.difficulty?.toLowerCase() === "beginner"
                  ? "bg-green-50 text-green-700 border-green-200 dark:bg-green-900/20 dark:text-green-300 dark:border-green-800"
                  : lab.difficulty?.toLowerCase() === "advanced"
                    ? "bg-red-50 text-red-700 border-red-200 dark:bg-red-900/20 dark:text-red-300 dark:border-red-800"
                    : "bg-blue-50 text-blue-700 border-blue-200 dark:bg-blue-900/20 dark:text-blue-300 dark:border-blue-800"
              }
            >
              {lab.difficulty || "Intermediate"}
            </Badge>
          </div>
          <h1 className="text-2xl font-bold text-gray-900 dark:text-white">{lab.title}</h1>
          {lab.description && (
            <p className="text-gray-600 dark:text-gray-400 mt-1">{lab.description}</p>
          )}
        </div>

        <div className="flex items-center gap-4">
          <div className="flex items-center gap-1 text-gray-600 dark:text-gray-400">
            <BarChart2 className="h-4 w-4" />
            <span>{lab.capsules.length} capsules</span>
          </div>
        </div>
      </div>
    </div>
  )
}
