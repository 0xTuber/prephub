import { render, screen, waitFor } from '@testing-library/react'
import ReviewSessionClient from '@/components/cbr/ReviewSessionClient'

// Mock framer-motion
jest.mock('framer-motion', () => ({
  motion: {
    div: ({ children, ...props }: any) => <div {...props}>{children}</div>,
    circle: ({ children, ...props }: any) => <circle {...props}>{children}</circle>,
  },
  AnimatePresence: ({ children }: any) => children,
}))

// Mock fetch
const mockFetch = jest.fn()
global.fetch = mockFetch

describe('ReviewSessionClient', () => {
  beforeEach(() => {
    mockFetch.mockClear()
  })

  it('shows loading state when autoStart is true', async () => {
    mockFetch.mockImplementation(() => new Promise(() => {})) // Never resolves

    render(<ReviewSessionClient autoStart={true} />)

    expect(screen.getByText('Generating your session...')).toBeInTheDocument()
  })

  it('shows no items message when API returns 404', async () => {
    mockFetch.mockResolvedValueOnce({
      ok: false,
      status: 404,
    })

    render(<ReviewSessionClient autoStart={true} />)

    await waitFor(() => {
      expect(screen.getByText('Start Your Journey!')).toBeInTheDocument()
    })
  })

  it('shows error message when API fails', async () => {
    mockFetch.mockRejectedValueOnce(new Error('Network error'))

    render(<ReviewSessionClient autoStart={true} />)

    await waitFor(() => {
      expect(screen.getByText('Failed to load session')).toBeInTheDocument()
    })
  })

  it('calls onSessionStart when session starts', async () => {
    const onSessionStart = jest.fn()
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: async () => ({
        questions: [
          {
            id: 'q1',
            text: 'Test question?',
            options: [
              { id: 'o1', text: 'Option 1', isCorrect: true },
              { id: 'o2', text: 'Option 2', isCorrect: false },
            ],
            labTitle: 'Test Lab',
            capsuleTitle: 'Test Capsule',
          }
        ],
        totalAvailable: 1,
      }),
    })

    render(<ReviewSessionClient autoStart={true} onSessionStart={onSessionStart} />)

    await waitFor(() => {
      expect(onSessionStart).toHaveBeenCalled()
    })
  })
})
