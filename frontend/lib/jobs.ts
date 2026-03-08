import { readFileSync, writeFileSync, existsSync, mkdirSync } from 'fs'
import path from 'path'
import type { GenerationJob } from './types'

const DATA_DIR = path.join(process.cwd(), '..', 'data')
const JOBS_FILE = path.join(DATA_DIR, 'jobs.json')

function ensureDataDir() {
  if (!existsSync(DATA_DIR)) {
    mkdirSync(DATA_DIR, { recursive: true })
  }
}

function readJobs(): Record<string, GenerationJob> {
  ensureDataDir()
  if (!existsSync(JOBS_FILE)) {
    return {}
  }
  try {
    const content = readFileSync(JOBS_FILE, 'utf-8')
    return JSON.parse(content)
  } catch {
    return {}
  }
}

function writeJobs(jobs: Record<string, GenerationJob>) {
  ensureDataDir()
  writeFileSync(JOBS_FILE, JSON.stringify(jobs, null, 2))
}

export function createJob(jobId: string, courseName: string): GenerationJob {
  const jobs = readJobs()
  const job: GenerationJob = {
    jobId,
    status: 'pending',
    progress: 0,
    courseName,
  }
  jobs[jobId] = job
  writeJobs(jobs)
  return job
}

export function getJob(jobId: string): GenerationJob | undefined {
  const jobs = readJobs()
  return jobs[jobId]
}

export function updateJob(jobId: string, updates: Partial<GenerationJob>): GenerationJob | undefined {
  const jobs = readJobs()
  const job = jobs[jobId]
  if (job) {
    const updated = { ...job, ...updates }
    jobs[jobId] = updated
    writeJobs(jobs)
    return updated
  }
  return undefined
}

export function deleteJob(jobId: string): boolean {
  const jobs = readJobs()
  if (jobs[jobId]) {
    delete jobs[jobId]
    writeJobs(jobs)
    return true
  }
  return false
}

export function getAllJobs(): GenerationJob[] {
  const jobs = readJobs()
  return Object.values(jobs)
}
