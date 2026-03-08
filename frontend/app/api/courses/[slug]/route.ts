import { NextRequest, NextResponse } from 'next/server'
import { readdir, readFile, stat } from 'fs/promises'
import path from 'path'

export async function GET(
  request: NextRequest,
  { params }: { params: Promise<{ slug: string }> }
) {
  try {
    const { slug } = await params
    const courseName = decodeURIComponent(slug)
    const courseDir = path.join(process.cwd(), '..', 'data', courseName)

    // Check if directory exists
    try {
      await stat(courseDir)
    } catch {
      return NextResponse.json({ error: 'Course not found' }, { status: 404 })
    }

    // Find skeleton file
    const files = await readdir(courseDir)
    const skeletonFile = files.find(f => f.endsWith('.json') && f.includes('skeleton'))

    if (!skeletonFile) {
      return NextResponse.json({ error: 'Course skeleton not found' }, { status: 404 })
    }

    const skeletonPath = path.join(courseDir, skeletonFile)
    const content = await readFile(skeletonPath, 'utf-8')
    const skeleton = JSON.parse(content)

    return NextResponse.json({
      slug,
      skeleton,
    })
  } catch (error) {
    console.error('Error fetching course:', error)
    return NextResponse.json({ error: 'Failed to fetch course' }, { status: 500 })
  }
}
