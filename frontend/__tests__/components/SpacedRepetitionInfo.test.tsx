import { render, screen, fireEvent } from '@testing-library/react'
import { SpacedRepetitionInfo } from '@/components/cbr/SpacedRepetitionInfo'

// Mock framer-motion to avoid animation issues in tests
jest.mock('framer-motion', () => ({
  motion: {
    div: ({ children, ...props }: any) => <div {...props}>{children}</div>,
    circle: ({ children, ...props }: any) => <circle {...props}>{children}</circle>,
    rect: ({ children, ...props }: any) => <rect {...props}>{children}</rect>,
    text: ({ children, ...props }: any) => <text {...props}>{children}</text>,
    path: ({ children, ...props }: any) => <path {...props}>{children}</path>,
    g: ({ children, ...props }: any) => <g {...props}>{children}</g>,
  },
  AnimatePresence: ({ children }: any) => children,
}))

// Mock Radix Dialog
jest.mock('@radix-ui/react-dialog', () => ({
  Root: ({ children, open }: any) => open ? <div data-testid="dialog">{children}</div> : <div>{children}</div>,
  Trigger: ({ children, asChild }: any) => asChild ? children : <button>{children}</button>,
  Portal: ({ children }: any) => children,
  Overlay: ({ children, ...props }: any) => <div data-testid="overlay" {...props}>{children}</div>,
  Content: ({ children, ...props }: any) => <div data-testid="dialog-content" {...props}>{children}</div>,
  Close: ({ children, ...props }: any) => <button {...props}>{children}</button>,
  Title: ({ children, ...props }: any) => <h2 {...props}>{children}</h2>,
  Description: ({ children, ...props }: any) => <p {...props}>{children}</p>,
}))

describe('SpacedRepetitionInfo', () => {
  it('renders the trigger button', () => {
    render(<SpacedRepetitionInfo />)
    expect(screen.getByText('How it works')).toBeInTheDocument()
  })

  it('shows dialog when open prop is true', () => {
    render(<SpacedRepetitionInfo open={true} onOpenChange={() => {}} />)
    expect(screen.getByTestId('dialog-content')).toBeInTheDocument()
    expect(screen.getByText('Spaced Repetition')).toBeInTheDocument()
  })

  it('shows first slide content', () => {
    render(<SpacedRepetitionInfo open={true} onOpenChange={() => {}} />)
    expect(screen.getByText('Answer + Confidence')).toBeInTheDocument()
  })
})
