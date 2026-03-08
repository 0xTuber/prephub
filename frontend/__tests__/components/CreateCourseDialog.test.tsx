import { render, screen, fireEvent } from '@testing-library/react'
import CreateCourseDialog from '@/components/CreateCourseDialog'

// Mock BookUploader
jest.mock('@/components/upload/BookUploader', () => {
  return function MockBookUploader({ onUploadComplete }: { onUploadComplete: (path: string, name: string) => void }) {
    return (
      <button
        data-testid="mock-uploader"
        onClick={() => onUploadComplete('/uploads/test.pdf', 'test.pdf')}
      >
        Upload
      </button>
    )
  }
})

describe('CreateCourseDialog', () => {
  const mockOnOpenChange = jest.fn()
  const mockOnGenerate = jest.fn()

  beforeEach(() => {
    mockOnOpenChange.mockClear()
    mockOnGenerate.mockClear()
  })

  it('renders dialog when open', () => {
    render(
      <CreateCourseDialog
        open={true}
        onOpenChange={mockOnOpenChange}
        onGenerate={mockOnGenerate}
      />
    )

    expect(screen.getByText('Create New Course')).toBeInTheDocument()
    expect(screen.getByText('Step 1: Upload Book')).toBeInTheDocument()
  })

  it('does not render dialog when closed', () => {
    render(
      <CreateCourseDialog
        open={false}
        onOpenChange={mockOnOpenChange}
        onGenerate={mockOnGenerate}
      />
    )

    expect(screen.queryByText('Create New Course')).not.toBeInTheDocument()
  })

  it('shows step 2 after upload', () => {
    render(
      <CreateCourseDialog
        open={true}
        onOpenChange={mockOnOpenChange}
        onGenerate={mockOnGenerate}
      />
    )

    // Initially Step 2 is not visible
    expect(screen.queryByText('Step 2: Course Name')).not.toBeInTheDocument()

    // Simulate upload
    fireEvent.click(screen.getByTestId('mock-uploader'))

    // Step 2 should now be visible
    expect(screen.getByText('Step 2: Course Name')).toBeInTheDocument()
  })

  it('auto-fills course name from file name', () => {
    render(
      <CreateCourseDialog
        open={true}
        onOpenChange={mockOnOpenChange}
        onGenerate={mockOnGenerate}
      />
    )

    fireEvent.click(screen.getByTestId('mock-uploader'))

    const input = screen.getByPlaceholderText('Enter course name...')
    expect(input).toHaveValue('test')
  })

  it('disables generate button when no file uploaded', () => {
    render(
      <CreateCourseDialog
        open={true}
        onOpenChange={mockOnOpenChange}
        onGenerate={mockOnGenerate}
      />
    )

    const generateButton = screen.getByRole('button', { name: /generate/i })
    expect(generateButton).toBeDisabled()
  })

  it('disables generate button when course name is empty', () => {
    render(
      <CreateCourseDialog
        open={true}
        onOpenChange={mockOnOpenChange}
        onGenerate={mockOnGenerate}
      />
    )

    fireEvent.click(screen.getByTestId('mock-uploader'))

    const input = screen.getByPlaceholderText('Enter course name...')
    fireEvent.change(input, { target: { value: '' } })

    const generateButton = screen.getByRole('button', { name: /generate/i })
    expect(generateButton).toBeDisabled()
  })

  it('enables generate button when file uploaded and name entered', () => {
    render(
      <CreateCourseDialog
        open={true}
        onOpenChange={mockOnOpenChange}
        onGenerate={mockOnGenerate}
      />
    )

    fireEvent.click(screen.getByTestId('mock-uploader'))

    const generateButton = screen.getByRole('button', { name: /generate/i })
    expect(generateButton).not.toBeDisabled()
  })

  it('calls onGenerate with correct params when generate is clicked', () => {
    render(
      <CreateCourseDialog
        open={true}
        onOpenChange={mockOnOpenChange}
        onGenerate={mockOnGenerate}
      />
    )

    fireEvent.click(screen.getByTestId('mock-uploader'))
    fireEvent.click(screen.getByRole('button', { name: /generate/i }))

    expect(mockOnGenerate).toHaveBeenCalledWith('/uploads/test.pdf', 'test')
    expect(mockOnOpenChange).toHaveBeenCalledWith(false)
  })

  it('calls onOpenChange when cancel is clicked', () => {
    render(
      <CreateCourseDialog
        open={true}
        onOpenChange={mockOnOpenChange}
        onGenerate={mockOnGenerate}
      />
    )

    fireEvent.click(screen.getByRole('button', { name: /cancel/i }))

    expect(mockOnOpenChange).toHaveBeenCalledWith(false)
    expect(mockOnGenerate).not.toHaveBeenCalled()
  })
})
