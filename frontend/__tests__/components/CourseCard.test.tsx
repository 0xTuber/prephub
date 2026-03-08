import { render, screen } from '@testing-library/react'
import CourseCard from '@/components/courses/CourseCard'
import type { Course } from '@/lib/types'

// Mock next/link
jest.mock('next/link', () => {
  return ({ children, href }: { children: React.ReactNode; href: string }) => (
    <a href={href}>{children}</a>
  )
})

describe('CourseCard', () => {
  const baseCourse: Course = {
    slug: 'test-course',
    name: 'Test Course',
    status: 'ready',
    moduleCount: 5,
    labCount: 15,
    createdAt: '2024-01-15T10:00:00Z',
  }

  it('renders course name', () => {
    render(<CourseCard course={baseCourse} />)
    expect(screen.getByText('Test Course')).toBeInTheDocument()
  })

  it('renders module and lab counts', () => {
    render(<CourseCard course={baseCourse} />)
    expect(screen.getByText('5 modules · 15 labs')).toBeInTheDocument()
  })

  it('shows Ready badge for ready status', () => {
    render(<CourseCard course={baseCourse} />)
    expect(screen.getByText('Ready')).toBeInTheDocument()
  })

  it('shows Processing badge for processing status', () => {
    const processingCourse = { ...baseCourse, status: 'processing' as const }
    render(<CourseCard course={processingCourse} />)
    expect(screen.getByText('Processing')).toBeInTheDocument()
  })

  it('shows Error badge for error status', () => {
    const errorCourse = { ...baseCourse, status: 'error' as const }
    render(<CourseCard course={errorCourse} />)
    expect(screen.getByText('Error')).toBeInTheDocument()
  })

  it('shows View Course button for ready courses', () => {
    render(<CourseCard course={baseCourse} />)
    expect(screen.getByText('View Course')).toBeInTheDocument()
  })

  it('does not show View Course button for processing courses', () => {
    const processingCourse = { ...baseCourse, status: 'processing' as const }
    render(<CourseCard course={processingCourse} />)
    expect(screen.queryByText('View Course')).not.toBeInTheDocument()
  })

  it('links to correct course page', () => {
    render(<CourseCard course={baseCourse} />)
    const link = screen.getByRole('link')
    expect(link).toHaveAttribute('href', '/courses/test-course')
  })
})
