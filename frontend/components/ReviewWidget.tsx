'use client'

import { useState, useEffect } from 'react'
import Link from 'next/link'
import { Brain, PlayCircle, Loader2 } from 'lucide-react'
import { Button } from '@/components/ui/button'

export default function ReviewWidget() {
  const [dueCount, setDueCount] = useState<number | null>(null)
  const [isLoading, setIsLoading] = useState(true)

  useEffect(() => {
    const fetchDueCount = async () => {
      try {
        const res = await fetch('/api/review/questions?count=1')
        if (res.ok) {
          const data = await res.json()
          setDueCount(data.totalAvailable || 0)
        } else {
          setDueCount(0)
        }
      } catch {
        setDueCount(0)
      } finally {
        setIsLoading(false)
      }
    }

    fetchDueCount()
  }, [])

  // Don't show widget if there are no questions to review
  if (!isLoading && dueCount === 0) {
    return null
  }

  return (
    <div className="bg-gradient-to-r from-blue-50 to-white dark:from-blue-900/20 dark:to-gray-900 border border-blue-200 dark:border-blue-800 rounded-2xl p-6">
      <div className="flex items-center justify-between gap-4 flex-wrap">
        <div className="flex items-center gap-4">
          <div className="p-3 bg-blue-100 dark:bg-blue-900/50 rounded-xl">
            <Brain className="h-6 w-6 text-blue-600 dark:text-blue-400" />
          </div>
          <div>
            <h3 className="font-semibold text-gray-900 dark:text-white">Review Center</h3>
            {isLoading ? (
              <div className="flex items-center gap-2 text-sm text-gray-500 dark:text-gray-400">
                <Loader2 className="h-3 w-3 animate-spin" />
                Loading...
              </div>
            ) : (
              <p className="text-sm text-gray-600 dark:text-gray-400">
                <span className="font-medium text-blue-600 dark:text-blue-400">{dueCount}</span>
                {' '}{dueCount === 1 ? 'question' : 'questions'} ready to review
              </p>
            )}
          </div>
        </div>
        <Button
          asChild
          className="bg-blue-600 hover:bg-blue-700 shadow-lg shadow-blue-500/20"
          disabled={isLoading}
        >
          <Link href="/review">
            <PlayCircle className="h-4 w-4 mr-2" />
            Start Review
          </Link>
        </Button>
      </div>
    </div>
  )
}
