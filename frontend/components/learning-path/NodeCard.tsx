'use client'

import { Star, Sparkles, CheckCircle } from 'lucide-react'
import { cn } from '@/lib/utils'
import type { PathNode } from '@/lib/types'

interface NodeCardProps {
  node: PathNode
  nodeNumber: number
  onClick?: () => void
}

export default function NodeCard({ node, nodeNumber, onClick }: NodeCardProps) {
  const status = node.status
  const completedLabs = node.labs.filter(l => l.capsules.every(() => true)).length
  const totalLabs = node.labs.length

  const isLocked = status === 'locked'
  const isAvailable = status === 'available'
  const isInProgress = status === 'in_progress'
  const isCompleted = status === 'completed'

  return (
    <div
      onClick={onClick}
      className={cn(
        'relative w-40 h-40 sm:w-48 sm:h-48 mx-auto transition-all duration-300',
        'cursor-pointer hover:scale-110 active:scale-95'
      )}
    >
      {/* Outer Circle/Ring */}
      <div
        className={cn(
          'absolute inset-0 rounded-full border-8 transition-all duration-500',
          isLocked && 'border-gray-300 dark:border-gray-700 bg-gray-100 dark:bg-gray-800',
          isAvailable && 'border-brand-bg-hover bg-gradient-to-br from-brand-bg-hover to-brand-primary shadow-xl shadow-brand-primary/40',
          isInProgress && 'border-brand-bg-hover bg-gradient-to-br from-brand-bg-subtle to-brand-bg-hover shadow-xl shadow-brand-bg-hover/40',
          isCompleted && 'border-green-500 bg-gradient-to-br from-green-400 to-green-600 shadow-xl shadow-green-500/40'
        )}
      >
        {/* Inner Circle */}
        <div className="absolute inset-3 rounded-full bg-white dark:bg-gray-900 flex items-center justify-center shadow-inner">
          {/* Number for Locked Nodes */}
          {isLocked && (
            <div className="text-4xl font-bold text-gray-400 dark:text-gray-600">
              {nodeNumber}
            </div>
          )}

          {/* Number and Sparkle Icon for Available Nodes */}
          {isAvailable && (
            <div className="flex flex-col items-center gap-1">
              <div className="text-3xl font-bold text-brand-primary dark:text-brand-bg-hover">
                {nodeNumber}
              </div>
              <Sparkles className="w-12 h-12 text-brand-primary dark:text-brand-bg-hover" />
            </div>
          )}

          {/* Progress Ring for In Progress */}
          {isInProgress && (
            <div className="flex flex-col items-center gap-2">
              <div className="text-2xl font-bold text-brand-primary">
                {nodeNumber}
              </div>
              <svg className="w-20 h-20 transform -rotate-90">
                <circle cx="40" cy="40" r="32" fill="none" stroke="#FEF3C7" strokeWidth="8" />
                <circle
                  cx="40"
                  cy="40"
                  r="32"
                  fill="none"
                  stroke="#F59E0B"
                  strokeWidth="8"
                  strokeDasharray={`${2 * Math.PI * 32}`}
                  strokeDashoffset={`${2 * Math.PI * 32 * (1 - completedLabs / totalLabs)}`}
                  className="transition-all duration-500"
                />
                <text
                  x="40"
                  y="40"
                  textAnchor="middle"
                  dy=".3em"
                  className="text-xl font-bold fill-yellow-600"
                  transform="rotate(90 40 40)"
                >
                  {completedLabs}/{totalLabs}
                </text>
              </svg>
            </div>
          )}

          {/* Checkmark for Completed */}
          {isCompleted && (
            <div className="flex flex-col items-center gap-1">
              <div className="text-2xl font-bold text-green-600">
                {nodeNumber}
              </div>
              <CheckCircle className="w-16 h-16 text-green-500" />
            </div>
          )}
        </div>
      </div>

      {/* Node Title Below Circle */}
      <div className="absolute -bottom-20 sm:-bottom-24 left-1/2 transform -translate-x-1/2 w-48 sm:w-56 text-center px-2">
        <h3 className="font-bold text-sm sm:text-base text-gray-900 dark:text-gray-100 mb-1 line-clamp-2">
          {node.title}
        </h3>
      </div>
    </div>
  )
}
