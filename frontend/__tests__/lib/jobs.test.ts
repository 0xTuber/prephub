import { createJob, getJob, updateJob, deleteJob } from '@/lib/jobs'

describe('jobs utility', () => {
  beforeEach(() => {
    // Clear any existing jobs by deleting them
    const jobId = 'test-job-id'
    deleteJob(jobId)
  })

  describe('createJob', () => {
    it('creates a new job with pending status', () => {
      const job = createJob('test-job-1', 'Test Course')
      expect(job.jobId).toBe('test-job-1')
      expect(job.status).toBe('pending')
      expect(job.progress).toBe(0)
      expect(job.courseName).toBe('Test Course')
    })
  })

  describe('getJob', () => {
    it('returns the job if it exists', () => {
      createJob('test-job-2', 'Test Course')
      const job = getJob('test-job-2')
      expect(job).toBeDefined()
      expect(job?.jobId).toBe('test-job-2')
    })

    it('returns undefined for non-existent job', () => {
      const job = getJob('non-existent')
      expect(job).toBeUndefined()
    })
  })

  describe('updateJob', () => {
    it('updates job properties', () => {
      createJob('test-job-3', 'Test Course')
      const updated = updateJob('test-job-3', { status: 'running', progress: 50 })
      expect(updated?.status).toBe('running')
      expect(updated?.progress).toBe(50)
    })

    it('returns undefined for non-existent job', () => {
      const updated = updateJob('non-existent', { status: 'running' })
      expect(updated).toBeUndefined()
    })
  })

  describe('deleteJob', () => {
    it('deletes an existing job', () => {
      createJob('test-job-4', 'Test Course')
      expect(deleteJob('test-job-4')).toBe(true)
      expect(getJob('test-job-4')).toBeUndefined()
    })

    it('returns false for non-existent job', () => {
      expect(deleteJob('non-existent')).toBe(false)
    })
  })
})
