import { render, screen } from '@testing-library/react'
import CourseGrid from '@/components/courses/CourseGrid'
import type { Course } from '@/lib/types'

// Mock CourseCard
jest.mock('@/components/courses/CourseCard', () => {
  return function MockCourseCard({ course }: { course: Course }) {
    return <div data-testid="course-card">{course.name}</div>
  }
})

describe('CourseGrid', () => {
  const mockCourses: Course[] = [
    {
      slug: 'course-1',
      name: 'Course 1',
      status: 'ready',
      moduleCount: 3,
      labCount: 10,
      createdAt: '2024-01-15T10:00:00Z',
    },
    {
      slug: 'course-2',
      name: 'Course 2',
      status: 'processing',
      moduleCount: 5,
      labCount: 20,
      createdAt: '2024-01-14T10:00:00Z',
    },
  ]

  it('renders course cards when courses are provided', () => {
    render(<CourseGrid courses={mockCourses} />)
    expect(screen.getAllByTestId('course-card')).toHaveLength(2)
    expect(screen.getByText('Course 1')).toBeInTheDocument()
    expect(screen.getByText('Course 2')).toBeInTheDocument()
  })

  it('shows empty state when no courses', () => {
    render(<CourseGrid courses={[]} />)
    expect(screen.getByText('No courses yet')).toBeInTheDocument()
    expect(screen.getByText(/New Course/)).toBeInTheDocument()
  })

  it('shows loading skeleton when loading', () => {
    const { container } = render(<CourseGrid courses={[]} isLoading={true} />)
    const skeletons = container.querySelectorAll('.animate-pulse')
    expect(skeletons.length).toBe(3)
  })

  it('does not show loading skeleton when not loading', () => {
    const { container } = render(<CourseGrid courses={mockCourses} isLoading={false} />)
    const skeletons = container.querySelectorAll('.animate-pulse')
    expect(skeletons.length).toBe(0)
  })
})
