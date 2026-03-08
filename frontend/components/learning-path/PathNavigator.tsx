'use client'

import { useState } from 'react'
import { Sparkles } from 'lucide-react'
import NodeCard from './NodeCard'
import NodeDetailPanel from './NodeDetailPanel'
import type { PathSection, PathNode } from '@/lib/types'

interface PathNavigatorProps {
  sections: PathSection[]
  courseSlug: string
}

export default function PathNavigator({ sections, courseSlug }: PathNavigatorProps) {
  const [selectedNode, setSelectedNode] = useState<PathNode | null>(null)

  return (
    <>
      <div className="relative">
        <div className="space-y-24 sm:space-y-32 pb-24 sm:pb-32">
          {sections.map((section, sectionIndex) => {
            const completedNodes = section.nodes.filter(n => n.status === 'completed').length
            const totalNodes = section.nodes.length
            const sectionProgress = totalNodes > 0 ? (completedNodes / totalNodes) * 100 : 0

            // Calculate cumulative node count for numbering
            const nodesBeforeThisSection = sections
              .slice(0, sectionIndex)
              .reduce((sum, s) => sum + s.nodes.length, 0)

            return (
              <div key={section.id} className="relative">
                {/* Section Header */}
                <div className="text-center mb-12 sm:mb-20 px-4">
                  <div className="inline-flex items-center gap-2 px-4 sm:px-6 py-2 sm:py-3 rounded-full bg-gradient-to-r from-brand-bg-subtle to-brand-primary/10 dark:from-brand-primary/20 dark:to-brand-primary/10 border-2 border-brand-bg-subtle dark:border-brand-primary/30 mb-3 sm:mb-4">
                    <Sparkles className="h-4 w-4 sm:h-5 sm:w-5 text-brand-primary dark:text-brand-bg-hover" />
                    <span className="font-semibold text-sm sm:text-base text-brand-primary dark:text-brand-bg-subtle">
                      Module {sectionIndex + 1}
                    </span>
                  </div>

                  <h2 className="text-2xl sm:text-3xl font-bold mb-2 sm:mb-3 text-gray-900 dark:text-gray-100">
                    {section.title}
                  </h2>

                  {section.description && (
                    <p className="text-sm sm:text-base text-muted-foreground max-w-2xl mx-auto mb-3 sm:mb-4 px-2">
                      {section.description}
                    </p>
                  )}

                  {/* Section Progress */}
                  <div className="flex items-center justify-center gap-2 text-xs sm:text-sm text-muted-foreground">
                    <div className="w-24 sm:w-32 h-1.5 sm:h-2 bg-gray-200 dark:bg-gray-700 rounded-full overflow-hidden">
                      <div
                        className="h-full bg-gradient-to-r from-brand-secondary to-brand-primary transition-all duration-500"
                        style={{ width: `${sectionProgress}%` }}
                      />
                    </div>
                    <span className="font-medium">{completedNodes}/{totalNodes}</span>
                  </div>
                </div>

                {/* Visual Node Tree */}
                <div className="relative max-w-sm sm:max-w-md mx-auto px-2">
                  {section.nodes.map((node, nodeIndex) => {
                    const prevNode = nodeIndex > 0 ? section.nodes[nodeIndex - 1] : null
                    const nodeNumber = nodesBeforeThisSection + nodeIndex + 1
                    const isLeft = nodeIndex % 2 === 0
                    const offsetClass = isLeft ? 'sm:-translate-x-8' : 'sm:translate-x-8'

                    return (
                      <div key={node.id} className="relative mb-24 sm:mb-32">
                        {/* Connection Line to Previous Node */}
                        {prevNode && (
                          <div className="absolute left-1/2 -top-24 sm:-top-32 transform -translate-x-1/2 w-1 h-24 sm:h-32 z-0">
                            <div
                              className="w-full h-full rounded-full transition-all duration-500"
                              style={{
                                background: node.status === 'completed'
                                  ? 'linear-gradient(to bottom, #10B981, #34D399)'
                                  : '#E5E7EB'
                              }}
                            />
                          </div>
                        )}

                        {/* Node Card */}
                        <div
                          className={`transform transition-all duration-300 ${offsetClass} relative z-10`}
                          onClick={() => setSelectedNode(node)}
                        >
                          <NodeCard
                            node={node}
                            nodeNumber={nodeNumber}
                          />
                        </div>
                      </div>
                    )
                  })}
                </div>

                {/* Decorative separator between sections */}
                {sectionIndex < sections.length - 1 && (
                  <div className="mt-16 sm:mt-24 flex items-center justify-center">
                    <div className="flex items-center gap-3 sm:gap-4">
                      <div className="h-px w-16 sm:w-24 bg-gradient-to-r from-transparent via-gray-300 to-gray-300 dark:via-gray-700 dark:to-gray-700" />
                      <div className="w-2 h-2 sm:w-3 sm:h-3 rounded-full bg-brand-primary" />
                      <div className="h-px w-16 sm:w-24 bg-gradient-to-l from-transparent via-gray-300 to-gray-300 dark:via-gray-700 dark:to-gray-700" />
                    </div>
                  </div>
                )}
              </div>
            )
          })}
        </div>
      </div>

      {/* Node Detail Side Panel */}
      {selectedNode && (
        <NodeDetailPanel
          node={selectedNode}
          isOpen={!!selectedNode}
          onClose={() => setSelectedNode(null)}
          courseSlug={courseSlug}
        />
      )}
    </>
  )
}
