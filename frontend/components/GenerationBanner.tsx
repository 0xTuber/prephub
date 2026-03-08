'use client'

import { X, Loader2, CheckCircle, AlertCircle } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Progress } from '@/components/ui/progress'
import type { GenerationJob } from '@/lib/types'

interface GenerationBannerProps {
  job: GenerationJob | null
  onDismiss: () => void
}

export default function GenerationBanner({ job, onDismiss }: GenerationBannerProps) {
  if (!job) return null

  const isComplete = job.status === 'completed'
  const isError = job.status === 'error'
  const isRunning = job.status === 'running' || job.status === 'pending'

  return (
    <div
      className={`
        sticky top-0 z-[60] w-full border-b shadow-sm
        ${isComplete ? 'bg-green-50 dark:bg-green-900/20 border-green-200 dark:border-green-800' : ''}
        ${isError ? 'bg-red-50 dark:bg-red-900/20 border-red-200 dark:border-red-800' : ''}
        ${isRunning ? 'bg-blue-50 dark:bg-blue-900/20 border-blue-200 dark:border-blue-800' : ''}
      `}
    >
      <div className="container mx-auto px-4 py-3">
        <div className="flex items-center justify-between gap-4">
          <div className="flex items-center gap-3 flex-1 min-w-0">
            {isRunning && (
              <Loader2 className="h-5 w-5 text-blue-600 dark:text-blue-400 animate-spin flex-shrink-0" />
            )}
            {isComplete && (
              <CheckCircle className="h-5 w-5 text-green-600 dark:text-green-400 flex-shrink-0" />
            )}
            {isError && (
              <AlertCircle className="h-5 w-5 text-red-600 dark:text-red-400 flex-shrink-0" />
            )}

            <div className="flex-1 min-w-0">
              <div className="flex items-center gap-2 flex-wrap">
                <span className={`font-medium truncate ${
                  isComplete ? 'text-green-700 dark:text-green-300' :
                  isError ? 'text-red-700 dark:text-red-300' :
                  'text-blue-700 dark:text-blue-300'
                }`}>
                  {isComplete && 'Course generated successfully!'}
                  {isError && (job.error || 'Generation failed')}
                  {isRunning && `Generating "${job.courseName || 'Course'}"...`}
                </span>
                {isRunning && (
                  <span className="text-sm text-blue-600 dark:text-blue-400">
                    {job.progress}%
                  </span>
                )}
              </div>

              {isRunning && job.currentStep && (
                <p className="text-sm text-blue-600/80 dark:text-blue-400/80 mt-0.5 truncate">
                  {job.currentStep}
                </p>
              )}
            </div>
          </div>

          {isRunning && (
            <div className="hidden sm:block w-32 flex-shrink-0">
              <Progress
                value={job.progress}
                className="h-2"
                indicatorClassName="bg-gradient-to-r from-blue-600 to-blue-400"
              />
            </div>
          )}

          {(isComplete || isError) && (
            <Button
              variant="ghost"
              size="icon"
              className={`flex-shrink-0 ${
                isComplete ? 'text-green-600 hover:text-green-700 hover:bg-green-100 dark:text-green-400 dark:hover:bg-green-900/30' :
                'text-red-600 hover:text-red-700 hover:bg-red-100 dark:text-red-400 dark:hover:bg-red-900/30'
              }`}
              onClick={onDismiss}
            >
              <X className="h-4 w-4" />
              <span className="sr-only">Dismiss</span>
            </Button>
          )}
        </div>

        {isRunning && (
          <div className="sm:hidden mt-2">
            <Progress
              value={job.progress}
              className="h-2"
              indicatorClassName="bg-gradient-to-r from-blue-600 to-blue-400"
            />
          </div>
        )}
      </div>
    </div>
  )
}
