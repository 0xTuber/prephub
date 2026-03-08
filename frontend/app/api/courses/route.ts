import { NextResponse } from 'next/server'
import { readdir, readFile, stat } from 'fs/promises'
import path from 'path'
import type { Course, CourseSkeleton } from '@/lib/types'

export async function GET() {
  try {
    const dataDir = path.join(process.cwd(), '..', 'data')
    const courses: Course[] = []

    // Check if data directory exists
    try {
      await stat(dataDir)
    } catch {
      return NextResponse.json({ courses: [] })
    }

    // Read all certification directories
    const entries = await readdir(dataDir, { withFileTypes: true })

    for (const entry of entries) {
      if (!entry.isDirectory()) continue
      if (entry.name === 'sources') continue // Skip sources directory

      const certDir = path.join(dataDir, entry.name)

      // Look for skeleton JSON files
      try {
        const files = await readdir(certDir)
        const skeletonFile = files.find(f => f.endsWith('.json') && f.includes('skeleton'))

        if (skeletonFile) {
          const skeletonPath = path.join(certDir, skeletonFile)
          const content = await readFile(skeletonPath, 'utf-8')
          const skeleton: CourseSkeleton = JSON.parse(content)

          // Count modules and labs
          let labCount = 0
          for (const module of skeleton.domain_modules || []) {
            for (const topic of module.topics || []) {
              for (const subtopic of topic.subtopics || []) {
                labCount += subtopic.labs?.length || 0
              }
            }
          }

          const fileStat = await stat(skeletonPath)

          courses.push({
            slug: encodeURIComponent(entry.name),
            name: skeleton.certification_name || entry.name,
            status: skeleton.validation_status === 'passed' ? 'ready' : 'ready',
            moduleCount: skeleton.domain_modules?.length || 0,
            labCount,
            createdAt: fileStat.mtime.toISOString(),
          })
        }
      } catch (error) {
        console.error(`Error reading course ${entry.name}:`, error)
      }
    }

    // Sort by creation date, newest first
    courses.sort((a, b) => new Date(b.createdAt).getTime() - new Date(a.createdAt).getTime())

    return NextResponse.json({ courses })
  } catch (error) {
    console.error('Error listing courses:', error)
    return NextResponse.json({ error: 'Failed to list courses' }, { status: 500 })
  }
}
