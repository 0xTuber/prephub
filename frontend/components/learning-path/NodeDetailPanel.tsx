'use client'

import Link from 'next/link'
import { Sheet, SheetContent, SheetHeader, SheetTitle, SheetDescription } from '@/components/ui/sheet'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { Card, CardContent } from '@/components/ui/card'
import { Progress } from '@/components/ui/progress'
import { CheckCircle, Clock, ArrowRight, ChevronDown, ChevronUp, Play } from 'lucide-react'
import { useState } from 'react'
import type { PathNode, PathLab, PathCapsule } from '@/lib/types'

interface NodeDetailPanelProps {
  node: PathNode
  isOpen: boolean
  onClose: () => void
  courseSlug: string
}

export default function NodeDetailPanel({ node, isOpen, onClose, courseSlug }: NodeDetailPanelProps) {
  const [expandedLab, setExpandedLab] = useState<string | null>(null)

  const completedLabs = 0 // In view-only mode, no progress tracking
  const totalLabs = node.labs.length
  const progressPercentage = totalLabs > 0 ? (completedLabs / totalLabs) * 100 : 0

  const getDifficultyColor = (difficulty: string) => {
    switch (difficulty.toLowerCase()) {
      case 'beginner':
        return 'bg-green-100 text-green-800 dark:bg-green-900/30 dark:text-green-200'
      case 'intermediate':
        return 'bg-yellow-100 text-yellow-800 dark:bg-yellow-900/30 dark:text-yellow-200'
      case 'advanced':
        return 'bg-red-100 text-red-800 dark:bg-red-900/30 dark:text-red-200'
      default:
        return 'bg-gray-100 text-gray-800 dark:bg-gray-900/30 dark:text-gray-200'
    }
  }

  const toggleLabExpand = (labId: string) => {
    setExpandedLab(expandedLab === labId ? null : labId)
  }

  return (
    <Sheet open={isOpen} onOpenChange={onClose}>
      <SheetContent side="right" className="w-full sm:max-w-2xl overflow-y-auto p-0">
        {/* Header */}
        <div className="sticky top-0 bg-background/95 backdrop-blur z-10 border-b">
          <div className="flex items-start justify-between p-4 sm:p-6">
            <SheetHeader className="flex-1 text-left">
              <SheetTitle className="text-xl sm:text-2xl pr-8">{node.title}</SheetTitle>
              <SheetDescription className="text-sm sm:text-base mt-2">
                {node.description}
              </SheetDescription>
            </SheetHeader>
          </div>
        </div>

        <div className="p-4 sm:p-6">
          {/* Node Stats */}
          <div className="grid grid-cols-2 gap-3 sm:gap-4 mb-6">
            <div className="text-center p-3 sm:p-4 bg-brand-bg-subtle dark:bg-blue-900/20 rounded-lg">
              <div className="text-2xl sm:text-3xl font-bold text-brand-primary">{totalLabs}</div>
              <div className="text-xs sm:text-sm text-muted-foreground mt-1">Labs</div>
            </div>
            {node.estimatedMinutes && (
              <div className="text-center p-3 sm:p-4 bg-brand-bg-subtle dark:bg-brand-primary/20 rounded-lg">
                <div className="text-2xl sm:text-3xl font-bold text-brand-primary">{node.estimatedMinutes}m</div>
                <div className="text-xs sm:text-sm text-muted-foreground mt-1">Est. Time</div>
              </div>
            )}
          </div>

          {/* Labs List */}
          <div className="space-y-3 sm:space-y-4">
            <h3 className="font-semibold text-lg sm:text-xl mb-3 sm:mb-4">Labs</h3>

            {node.labs.map((lab, index) => (
              <Card key={lab.id} className="transition-all border-2 hover:shadow-md border-gray-200 dark:border-gray-800">
                <CardContent className="p-4 sm:p-5">
                  <div className="flex flex-col gap-4">
                    {/* Lab Header */}
                    <div className="flex items-start justify-between">
                      <div className="flex items-start gap-3">
                        <div className="w-10 h-10 rounded-full bg-gradient-to-br from-brand-secondary to-brand-primary flex items-center justify-center shadow-lg flex-shrink-0">
                          <span className="text-white font-bold">{index + 1}</span>
                        </div>
                        <div>
                          <h4 className="font-bold text-base sm:text-lg">{lab.title}</h4>
                          <div className="flex items-center gap-2 mt-1">
                            <Badge className={getDifficultyColor(lab.difficulty)}>
                              {lab.difficulty}
                            </Badge>
                            {lab.estimatedMinutes && (
                              <span className="text-xs text-muted-foreground flex items-center gap-1">
                                <Clock className="h-3 w-3" />
                                {lab.estimatedMinutes} min
                              </span>
                            )}
                          </div>
                        </div>
                      </div>
                      <Button
                        variant="ghost"
                        size="sm"
                        onClick={() => toggleLabExpand(lab.id)}
                      >
                        {expandedLab === lab.id ? (
                          <ChevronUp className="h-4 w-4" />
                        ) : (
                          <ChevronDown className="h-4 w-4" />
                        )}
                      </Button>
                    </div>

                    {/* Lab Description */}
                    {lab.description && (
                      <p className="text-sm text-muted-foreground line-clamp-2">
                        {lab.description}
                      </p>
                    )}

                    {/* Start Lab Button */}
                    <Link href={`/courses/${courseSlug}/labs/${lab.id}`}>
                      <Button className="w-full bg-gradient-to-r from-brand-secondary to-brand-primary hover:opacity-90 text-white">
                        <Play className="h-4 w-4 mr-2" />
                        Start Lab
                      </Button>
                    </Link>

                    {/* Expanded Capsules */}
                    {expandedLab === lab.id && lab.capsules.length > 0 && (
                      <div className="mt-2 space-y-2 pl-4 border-l-2 border-brand-bg-subtle">
                        <h5 className="text-sm font-medium text-muted-foreground">Capsules</h5>
                        {lab.capsules.map((capsule, capsuleIndex) => (
                          <div
                            key={capsule.id}
                            className="flex items-center justify-between p-2 bg-gray-50 dark:bg-gray-800/50 rounded-lg"
                          >
                            <div className="flex items-center gap-2">
                              <span className="text-xs font-medium text-muted-foreground">
                                {capsuleIndex + 1}.
                              </span>
                              <span className="text-sm">{capsule.title}</span>
                            </div>
                            <Badge variant="outline" className="text-xs">
                              {capsule.questionCount} questions
                            </Badge>
                          </div>
                        ))}
                      </div>
                    )}
                  </div>
                </CardContent>
              </Card>
            ))}
          </div>

          {/* Mobile: Close Button at Bottom */}
          <div className="mt-8 sm:hidden">
            <Button onClick={onClose} variant="outline" size="lg" className="w-full">
              Close
            </Button>
          </div>
        </div>
      </SheetContent>
    </Sheet>
  )
}
