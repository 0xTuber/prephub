import { render, screen, waitFor } from '@testing-library/react'
import ReviewWidget from '@/components/ReviewWidget'

// Mock fetch
const mockFetch = jest.fn()
global.fetch = mockFetch

describe('ReviewWidget', () => {
  beforeEach(() => {
    mockFetch.mockClear()
  })

  it('shows loading state initially', () => {
    mockFetch.mockImplementation(() => new Promise(() => {})) // Never resolves

    render(<ReviewWidget />)

    expect(screen.getByText('Loading...')).toBeInTheDocument()
  })

  it('shows question count when available', async () => {
    mockFetch.mockResolvedValue({
      ok: true,
      json: () => Promise.resolve({ totalAvailable: 15 }),
    })

    render(<ReviewWidget />)

    await waitFor(() => {
      expect(screen.getByText('15')).toBeInTheDocument()
      expect(screen.getByText(/questions ready to review/)).toBeInTheDocument()
    })
  })

  it('shows singular when 1 question', async () => {
    mockFetch.mockResolvedValue({
      ok: true,
      json: () => Promise.resolve({ totalAvailable: 1 }),
    })

    render(<ReviewWidget />)

    await waitFor(() => {
      expect(screen.getByText('1')).toBeInTheDocument()
      expect(screen.getByText(/question ready to review/)).toBeInTheDocument()
    })
  })

  it('hides widget when no questions available', async () => {
    mockFetch.mockResolvedValue({
      ok: true,
      json: () => Promise.resolve({ totalAvailable: 0 }),
    })

    const { container } = render(<ReviewWidget />)

    await waitFor(() => {
      expect(container.firstChild).toBeNull()
    })
  })

  it('hides widget on fetch error', async () => {
    mockFetch.mockRejectedValue(new Error('Network error'))

    const { container } = render(<ReviewWidget />)

    await waitFor(() => {
      expect(container.firstChild).toBeNull()
    })
  })

  it('has link to review center', async () => {
    mockFetch.mockResolvedValue({
      ok: true,
      json: () => Promise.resolve({ totalAvailable: 10 }),
    })

    render(<ReviewWidget />)

    await waitFor(() => {
      const link = screen.getByRole('link', { name: /start review/i })
      expect(link).toHaveAttribute('href', '/review')
    })
  })
})
