"use client"

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Trophy, Award, BookOpen } from "lucide-react"
import type { InteractiveLab, LabCapsule } from "@/lib/types"

interface LabResultsSummaryProps {
  lab: InteractiveLab & { capsules: (LabCapsule & { isCompleted: boolean; score?: number })[] }
}

export default function LabResultsSummary({ lab }: LabResultsSummaryProps) {
  const completedCapsules = lab.capsules.filter(c => c.isCompleted)
  const totalCapsules = lab.capsules.length
  const progress = totalCapsules > 0 ? (completedCapsules.length / totalCapsules) * 100 : 0
  const isLabCompleted = progress === 100

  // Calculate overall lab statistics
  const totalScore = completedCapsules.reduce((sum, capsule) => sum + (capsule.score || 0), 0)
  const averageScore = completedCapsules.length > 0 ? Math.round(totalScore / completedCapsules.length) : 0
  const totalTimeSpent = completedCapsules.reduce((sum, capsule) => sum + (capsule.timeSpent || 0), 0)
  const totalTimeMinutes = Math.floor(totalTimeSpent / 60)
  const totalTimeSeconds = totalTimeSpent % 60

  // Performance categories
  const excellentCapsules = completedCapsules.filter(c => (c.score || 0) >= 90).length
  const goodCapsules = completedCapsules.filter(c => (c.score || 0) >= 80 && (c.score || 0) < 90).length
  const fairCapsules = completedCapsules.filter(c => (c.score || 0) >= 70 && (c.score || 0) < 80).length
  const needsImprovementCapsules = completedCapsules.filter(c => (c.score || 0) < 70).length

  // Calculate success rate (percentage of capsules with 70%+ score)
  const passingCapsules = excellentCapsules + goodCapsules + fairCapsules
  const successRate = completedCapsules.length > 0 ? Math.round((passingCapsules / completedCapsules.length) * 100) : 0

  // Get performance level
  const getPerformanceLevel = (score: number) => {
    if (score >= 90) return { level: "Excellent", color: 'text-green-600 dark:text-green-400', bg: 'bg-green-100 dark:bg-green-900/20' }
    if (score >= 80) return { level: "Good", color: 'text-blue-600 dark:text-blue-400', bg: 'bg-blue-100 dark:bg-blue-900/20' }
    if (score >= 70) return { level: "Fair", color: 'text-yellow-600 dark:text-yellow-400', bg: 'bg-yellow-100 dark:bg-yellow-900/20' }
    return { level: "Needs Improvement", color: 'text-red-600 dark:text-red-400', bg: 'bg-red-100 dark:bg-red-900/20' }
  }

  // Format capsule type for display
  const formatCapsuleType = (type: string) => {
    return type.charAt(0).toUpperCase() + type.slice(1).replace(/_/g, ' ')
  }

  if (!isLabCompleted) {
    return null
  }

  return (
    <div className="space-y-2">
      {/* Lab Completion Celebration */}
      <Card className="bg-gradient-to-r from-emerald-50 to-green-50 dark:from-emerald-900/20 dark:to-green-900/20 border-emerald-200 dark:border-emerald-800">
        <CardContent className="p-2">
          <div className="flex items-center justify-center gap-2">
            <div className="h-6 w-6 rounded-full bg-emerald-100 dark:bg-emerald-900/30 flex items-center justify-center">
              <Trophy className="h-4 w-4 text-emerald-600 dark:text-emerald-400" />
            </div>
            <div className="text-center">
              <h2 className="text-base font-bold text-emerald-800 dark:text-emerald-200">Lab Completed!</h2>
              <p className="text-xs text-emerald-700 dark:text-emerald-300">Congratulations on finishing this lab</p>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Overall Performance Metrics */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-2">
        <Card>
          <CardContent className="p-2 text-center">
            <div className="text-lg font-bold text-gray-800 dark:text-gray-200">
              {averageScore}%
            </div>
            <div className="text-xs text-gray-600 dark:text-gray-400 font-normal">
              Average Score
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-2 text-center">
            <div className="text-lg font-bold text-gray-800 dark:text-gray-200">
              {completedCapsules.length}
            </div>
            <div className="text-xs text-gray-600 dark:text-gray-400 font-normal">
              Capsules Completed
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-2 text-center">
            <div className="text-lg font-bold text-gray-800 dark:text-gray-200">
              {totalTimeMinutes}:{totalTimeSeconds.toString().padStart(2, '0')}
            </div>
            <div className="text-xs text-gray-600 dark:text-gray-400 font-normal">
              Total Time
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-2 text-center">
            <div className="text-lg font-bold text-gray-800 dark:text-gray-200">
              {successRate}%
            </div>
            <div className="text-xs text-gray-600 dark:text-gray-400 font-normal">
              Success Rate
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Performance Breakdown */}
      <Card>
        <CardHeader className="p-2">
          <CardTitle className="flex items-center gap-2 text-sm text-gray-800 dark:text-gray-200">
            <Award className="h-4 w-4" />
            Performance Breakdown
          </CardTitle>
        </CardHeader>
        <CardContent className="p-2 pt-0">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-2">
            {/* Performance Categories */}
            <div className="space-y-1">
              <h4 className="font-semibold text-gray-700 dark:text-gray-300 text-xs">Score Distribution</h4>
              <div className="space-y-1">
                <div className="flex items-center justify-between p-1.5 bg-emerald-100 dark:bg-emerald-900/30 rounded border">
                  <span className="text-xs font-medium text-emerald-800 dark:text-emerald-200">Excellent (90%+)</span>
                  <Badge variant="secondary" className="bg-emerald-200 text-emerald-900 dark:bg-emerald-800 dark:text-emerald-100 text-xs">
                    {excellentCapsules}
                  </Badge>
                </div>
                <div className="flex items-center justify-between p-1.5 bg-blue-100 dark:bg-blue-900/30 rounded border">
                  <span className="text-xs font-medium text-blue-800 dark:text-blue-200">Good (80-89%)</span>
                  <Badge variant="secondary" className="bg-blue-200 text-blue-900 dark:bg-blue-800 dark:text-blue-100 text-xs">
                    {goodCapsules}
                  </Badge>
                </div>
                <div className="flex items-center justify-between p-1.5 bg-amber-100 dark:bg-amber-900/30 rounded border">
                  <span className="text-xs font-medium text-amber-800 dark:text-amber-200">Fair (70-79%)</span>
                  <Badge variant="secondary" className="bg-amber-200 text-amber-900 dark:bg-amber-800 dark:text-amber-100 text-xs">
                    {fairCapsules}
                  </Badge>
                </div>
                <div className="flex items-center justify-between p-1.5 bg-red-100 dark:bg-red-900/30 rounded border">
                  <span className="text-xs font-medium text-red-800 dark:text-red-200">Needs Improvement (&lt;70%)</span>
                  <Badge variant="secondary" className="bg-red-200 text-red-900 dark:bg-red-800 dark:text-red-100 text-xs">
                    {needsImprovementCapsules}
                  </Badge>
                </div>
              </div>
            </div>

            {/* Learning Insights */}
            <div className="space-y-1">
              <h4 className="font-semibold text-gray-700 dark:text-gray-300 text-xs">Learning Insights</h4>
              <div className="space-y-1">
                <div className="p-1.5 bg-gray-100 dark:bg-gray-800/60 rounded border">
                  <div className="text-xs font-medium text-gray-700 dark:text-gray-300">
                    Overall Performance
                  </div>
                  <div className={`text-sm font-bold ${getPerformanceLevel(averageScore).color}`}>
                    {getPerformanceLevel(averageScore).level}
                  </div>
                </div>
                <div className="p-1.5 bg-gray-100 dark:bg-gray-800/60 rounded border">
                  <div className="text-xs font-medium text-gray-700 dark:text-gray-300">
                    Success Rate
                  </div>
                  <div className={`text-sm font-bold ${successRate >= 80 ? 'text-emerald-600 dark:text-emerald-400' : successRate >= 60 ? 'text-amber-600 dark:text-amber-400' : 'text-red-600 dark:text-red-400'}`}>
                    {successRate}%
                  </div>
                </div>
              </div>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Individual Capsule Results */}
      <Card>
        <CardHeader className="p-2">
          <CardTitle className="flex items-center gap-2 text-sm text-gray-800 dark:text-gray-200">
            <BookOpen className="h-4 w-4" />
            Capsule Results
          </CardTitle>
        </CardHeader>
        <CardContent className="p-2 pt-0">
          <div className="space-y-1">
            {lab.capsules.map((capsule, index) => {
              const performance = getPerformanceLevel(capsule.score || 0)
              const timeMinutes = Math.floor((capsule.timeSpent || 0) / 60)
              const timeSeconds = (capsule.timeSpent || 0) % 60

              return (
                <div
                  key={capsule.id}
                  className="flex items-center justify-between p-1.5 border-b hover:bg-gray-50 dark:hover:bg-gray-800/30"
                >
                  <div className="flex items-center gap-2">
                    <div className="flex h-5 w-5 items-center justify-center rounded-full bg-gray-100 dark:bg-gray-800 flex-shrink-0">
                      <span className="text-xs font-medium text-gray-700 dark:text-gray-300">{index + 1}</span>
                    </div>
                    <div>
                      <div className="font-medium text-gray-800 dark:text-gray-200 text-xs">
                        {capsule.title}
                      </div>
                      <div className="text-xs text-gray-500 dark:text-gray-400">
                        {formatCapsuleType(capsule.type)} Capsule
                      </div>
                    </div>
                  </div>
                  <div className="flex items-center gap-2">
                    <div className="text-right">
                      <div className="font-medium text-gray-800 dark:text-gray-200 text-xs">
                        {capsule.score}%
                      </div>
                      <div className="text-xs text-gray-500 dark:text-gray-400">
                        {timeMinutes}:{timeSeconds.toString().padStart(2, '0')}
                      </div>
                    </div>
                    <Badge variant="secondary" className={`${performance.bg} ${performance.color} text-xs`}>
                      {performance.level}
                    </Badge>
                  </div>
                </div>
              )
            })}
          </div>
        </CardContent>
      </Card>
    </div>
  )
}
