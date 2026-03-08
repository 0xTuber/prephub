import { render, screen, fireEvent } from '@testing-library/react'
import GenerationBanner from '@/components/GenerationBanner'
import type { GenerationJob } from '@/lib/types'

describe('GenerationBanner', () => {
  const mockOnDismiss = jest.fn()

  beforeEach(() => {
    mockOnDismiss.mockClear()
  })

  it('renders nothing when job is null', () => {
    const { container } = render(<GenerationBanner job={null} onDismiss={mockOnDismiss} />)
    expect(container.firstChild).toBeNull()
  })

  it('shows running state with progress', () => {
    const job: GenerationJob = {
      jobId: 'test-job',
      status: 'running',
      progress: 45,
      courseName: 'Test Course',
      currentStep: 'Processing chapters...',
    }

    render(<GenerationBanner job={job} onDismiss={mockOnDismiss} />)

    expect(screen.getByText(/Generating "Test Course".../)).toBeInTheDocument()
    expect(screen.getByText('45%')).toBeInTheDocument()
    expect(screen.getByText('Processing chapters...')).toBeInTheDocument()
  })

  it('shows pending state as running', () => {
    const job: GenerationJob = {
      jobId: 'test-job',
      status: 'pending',
      progress: 0,
      courseName: 'Test Course',
    }

    render(<GenerationBanner job={job} onDismiss={mockOnDismiss} />)

    expect(screen.getByText(/Generating "Test Course".../)).toBeInTheDocument()
  })

  it('shows completed state with dismiss button', () => {
    const job: GenerationJob = {
      jobId: 'test-job',
      status: 'completed',
      progress: 100,
    }

    render(<GenerationBanner job={job} onDismiss={mockOnDismiss} />)

    expect(screen.getByText('Course generated successfully!')).toBeInTheDocument()
    expect(screen.getByRole('button', { name: /dismiss/i })).toBeInTheDocument()
  })

  it('shows error state with error message', () => {
    const job: GenerationJob = {
      jobId: 'test-job',
      status: 'error',
      progress: 0,
      error: 'Failed to process PDF',
    }

    render(<GenerationBanner job={job} onDismiss={mockOnDismiss} />)

    expect(screen.getByText('Failed to process PDF')).toBeInTheDocument()
  })

  it('shows default error message when no error provided', () => {
    const job: GenerationJob = {
      jobId: 'test-job',
      status: 'error',
      progress: 0,
    }

    render(<GenerationBanner job={job} onDismiss={mockOnDismiss} />)

    expect(screen.getByText('Generation failed')).toBeInTheDocument()
  })

  it('calls onDismiss when dismiss button is clicked', () => {
    const job: GenerationJob = {
      jobId: 'test-job',
      status: 'completed',
      progress: 100,
    }

    render(<GenerationBanner job={job} onDismiss={mockOnDismiss} />)

    fireEvent.click(screen.getByRole('button', { name: /dismiss/i }))
    expect(mockOnDismiss).toHaveBeenCalledTimes(1)
  })

  it('does not show dismiss button while running', () => {
    const job: GenerationJob = {
      jobId: 'test-job',
      status: 'running',
      progress: 50,
    }

    render(<GenerationBanner job={job} onDismiss={mockOnDismiss} />)

    expect(screen.queryByRole('button', { name: /dismiss/i })).not.toBeInTheDocument()
  })
})
