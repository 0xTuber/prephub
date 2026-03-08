import { render, screen, fireEvent } from '@testing-library/react'
import NodeCard from '@/components/learning-path/NodeCard'
import type { PathNode } from '@/lib/types'

describe('NodeCard', () => {
  const baseNode: PathNode = {
    id: 'node-1',
    title: 'Test Node',
    description: 'Test description',
    status: 'available',
    labs: [
      {
        id: 'lab-1',
        title: 'Lab 1',
        difficulty: 'beginner',
        capsules: [],
      },
    ],
    estimatedMinutes: 30,
  }

  it('renders node title', () => {
    render(<NodeCard node={baseNode} nodeNumber={1} />)
    expect(screen.getByText('Test Node')).toBeInTheDocument()
  })

  it('renders node number', () => {
    render(<NodeCard node={baseNode} nodeNumber={5} />)
    expect(screen.getByText('5')).toBeInTheDocument()
  })

  it('calls onClick when clicked', () => {
    const handleClick = jest.fn()
    render(<NodeCard node={baseNode} nodeNumber={1} onClick={handleClick} />)
    fireEvent.click(screen.getByText('Test Node').closest('div')!)
    expect(handleClick).toHaveBeenCalled()
  })

  it('shows sparkles icon for available nodes', () => {
    render(<NodeCard node={baseNode} nodeNumber={1} />)
    // Sparkles icon should be present
    const sparkles = document.querySelector('.lucide-sparkles')
    expect(sparkles).toBeInTheDocument()
  })

  it('shows checkmark for completed nodes', () => {
    const completedNode = { ...baseNode, status: 'completed' as const }
    const { container } = render(<NodeCard node={completedNode} nodeNumber={1} />)
    // Check for the green border which indicates completed status
    const greenBorder = container.querySelector('.border-green-500')
    expect(greenBorder).toBeInTheDocument()
  })

  it('applies locked styling for locked nodes', () => {
    const lockedNode = { ...baseNode, status: 'locked' as const }
    const { container } = render(<NodeCard node={lockedNode} nodeNumber={1} />)
    const outerCircle = container.querySelector('.border-gray-300')
    expect(outerCircle).toBeInTheDocument()
  })
})
