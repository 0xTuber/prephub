import { NextRequest, NextResponse } from 'next/server'
import { spawn } from 'child_process'
import path from 'path'
import { createJob, updateJob } from '@/lib/jobs'

export async function POST(request: NextRequest) {
  try {
    const body = await request.json()
    const { bookPath, certificationName } = body

    if (!bookPath || !certificationName) {
      return NextResponse.json(
        { error: 'bookPath and certificationName are required' },
        { status: 400 }
      )
    }

    const jobId = `job_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`
    const job = createJob(jobId, certificationName)

    // Start the course-builder pipeline in the background
    const projectRoot = path.join(process.cwd(), '..')

    // Use uv to run the course-builder CLI
    const child = spawn('uv', ['run', 'course-builder', 'generate', '--book', bookPath, '--name', certificationName], {
      cwd: projectRoot,
      shell: true,
      detached: true,
      stdio: ['ignore', 'pipe', 'pipe'],
    })

    // Track progress from stdout
    let lastProgress = 0
    child.stdout?.on('data', (data: Buffer) => {
      const output = data.toString()
      console.log('[course-builder]', output)

      // Parse progress from output
      if (output.includes('Step 1')) {
        updateJob(jobId, { status: 'running', progress: 10, currentStep: 'Discovering exam format...' })
      } else if (output.includes('Step 2')) {
        updateJob(jobId, { status: 'running', progress: 25, currentStep: 'Generating course structure...' })
      } else if (output.includes('Step 3')) {
        updateJob(jobId, { status: 'running', progress: 40, currentStep: 'Creating labs...' })
      } else if (output.includes('Step 4')) {
        updateJob(jobId, { status: 'running', progress: 55, currentStep: 'Building capsules...' })
      } else if (output.includes('Step 5')) {
        updateJob(jobId, { status: 'running', progress: 70, currentStep: 'Generating content...' })
      } else if (output.includes('Step 6')) {
        updateJob(jobId, { status: 'running', progress: 85, currentStep: 'Validating content...' })
      } else if (output.includes('Complete') || output.includes('Success')) {
        updateJob(jobId, { status: 'completed', progress: 100, currentStep: 'Done!' })
      }
    })

    child.stderr?.on('data', (data: Buffer) => {
      console.error('[course-builder error]', data.toString())
    })

    child.on('close', (code) => {
      if (code === 0) {
        updateJob(jobId, { status: 'completed', progress: 100 })
      } else {
        updateJob(jobId, { status: 'error', error: `Process exited with code ${code}` })
      }
    })

    child.on('error', (error) => {
      console.error('Failed to start course-builder:', error)
      updateJob(jobId, { status: 'error', error: error.message })
    })

    // Unref to allow the parent process to exit independently
    child.unref()

    // Mark as running
    updateJob(jobId, { status: 'running', progress: 5, currentStep: 'Starting pipeline...' })

    return NextResponse.json(job)
  } catch (error) {
    console.error('Generation error:', error)
    return NextResponse.json(
      { error: 'Failed to start generation' },
      { status: 500 }
    )
  }
}
