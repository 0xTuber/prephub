"use client"

import { useState, useEffect } from "react"
import { Card, CardContent } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Loader2, RefreshCw, Target, Clock, AlertTriangle, Brain, X, Check } from "lucide-react"
import Link from "next/link"
import { motion } from "framer-motion"

interface ReviewQuestion {
  id: string
  text: string
  options: { id: string; text: string; isCorrect: boolean }[]
  labTitle: string
  capsuleTitle: string
}

interface SessionSummary {
  itemsCompleted: number
  accuracyRate: number
  averageConfidence: number
  totalTimeSpent: number
}

interface ReviewSessionClientProps {
  onSessionStart?: () => void
  onSessionEnd?: (summary: SessionSummary | null) => void
  autoStart?: boolean
  questionCount?: number
}

// Circular Progress Ring component
function CircularProgressRing({ percentage, size = 160 }: { percentage: number; size?: number }) {
  const strokeWidth = 10
  const radius = (size - strokeWidth) / 2
  const circumference = 2 * Math.PI * radius
  const offset = circumference - (percentage / 100) * circumference

  return (
    <div className="relative" style={{ width: size, height: size }}>
      <svg width={size} height={size} className="-rotate-90">
        <circle
          cx={size / 2}
          cy={size / 2}
          r={radius}
          fill="none"
          stroke="#e5e7eb"
          strokeWidth={strokeWidth}
        />
        <motion.circle
          cx={size / 2}
          cy={size / 2}
          r={radius}
          fill="none"
          stroke={percentage >= 70 ? '#22c55e' : percentage >= 40 ? '#f59e0b' : '#ef4444'}
          strokeWidth={strokeWidth}
          strokeLinecap="round"
          strokeDasharray={circumference}
          initial={{ strokeDashoffset: circumference }}
          animate={{ strokeDashoffset: offset }}
          transition={{ duration: 1.2, ease: "easeOut", delay: 0.3 }}
        />
      </svg>
      <div className="absolute inset-0 flex flex-col items-center justify-center">
        <motion.span
          className="text-4xl font-bold text-gray-900 dark:text-white"
          initial={{ opacity: 0, scale: 0.5 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ duration: 0.5, delay: 0.8 }}
        >
          {percentage}%
        </motion.span>
        <span className="text-xs text-gray-500 font-medium uppercase tracking-wider">Accuracy</span>
      </div>
    </div>
  )
}

// Question Runner Component
function QuestionRunner({
  questions,
  onComplete,
  onQuit
}: {
  questions: ReviewQuestion[]
  onComplete: (results: Array<{ questionId: string; isCorrect: boolean; confidence: number; timeSpent: number }>) => void
  onQuit: (partialResults: Array<{ questionId: string; isCorrect: boolean; confidence: number; timeSpent: number }>) => void
}) {
  const [currentIndex, setCurrentIndex] = useState(0)
  const [selectedOption, setSelectedOption] = useState<string | null>(null)
  const [showResult, setShowResult] = useState(false)
  const [confidence, setConfidence] = useState<number | null>(null)
  const [results, setResults] = useState<Array<{ questionId: string; isCorrect: boolean; confidence: number; timeSpent: number }>>([])
  const [startTime, setStartTime] = useState(Date.now())

  const currentQuestion = questions[currentIndex]
  const progress = ((currentIndex) / questions.length) * 100

  const handleOptionSelect = (optionId: string) => {
    if (showResult) return
    setSelectedOption(optionId)
  }

  const handleSubmit = () => {
    if (!selectedOption) return
    setShowResult(true)
  }

  const handleConfidenceSelect = (conf: number) => {
    setConfidence(conf)

    const isCorrect = currentQuestion.options.find(o => o.id === selectedOption)?.isCorrect || false
    const timeSpent = (Date.now() - startTime) / 1000

    const newResult = {
      questionId: currentQuestion.id,
      isCorrect,
      confidence: conf,
      timeSpent
    }

    const newResults = [...results, newResult]
    setResults(newResults)

    // Move to next question or complete
    if (currentIndex < questions.length - 1) {
      setTimeout(() => {
        setCurrentIndex(currentIndex + 1)
        setSelectedOption(null)
        setShowResult(false)
        setConfidence(null)
        setStartTime(Date.now())
      }, 300)
    } else {
      onComplete(newResults)
    }
  }

  const handleQuit = () => {
    onQuit(results)
  }

  return (
    <div className="space-y-6">
      {/* Progress bar */}
      <div className="flex items-center gap-4">
        <div className="flex-1 h-2 bg-gray-200 dark:bg-gray-700 rounded-full overflow-hidden">
          <motion.div
            className="h-full bg-blue-600"
            initial={{ width: 0 }}
            animate={{ width: `${progress}%` }}
            transition={{ duration: 0.3 }}
          />
        </div>
        <span className="text-sm text-gray-500 dark:text-gray-400">
          {currentIndex + 1} / {questions.length}
        </span>
        <Button variant="ghost" size="sm" onClick={handleQuit} className="text-gray-500">
          <X className="h-4 w-4" />
        </Button>
      </div>

      {/* Question Card */}
      <Card className="border-gray-200 dark:border-gray-700">
        <CardContent className="p-6 space-y-6">
          {/* Context */}
          <div className="text-xs text-gray-500 dark:text-gray-400">
            {currentQuestion.labTitle} • {currentQuestion.capsuleTitle}
          </div>

          {/* Question text */}
          <p className="text-lg font-medium text-gray-900 dark:text-white">
            {currentQuestion.text}
          </p>

          {/* Options */}
          <div className="space-y-3">
            {currentQuestion.options.map((option) => {
              const isSelected = selectedOption === option.id
              const isCorrect = option.isCorrect

              let optionClass = "border-gray-200 dark:border-gray-700 hover:border-blue-300 dark:hover:border-blue-600"
              if (showResult) {
                if (isCorrect) {
                  optionClass = "border-green-500 bg-green-50 dark:bg-green-900/20"
                } else if (isSelected && !isCorrect) {
                  optionClass = "border-red-500 bg-red-50 dark:bg-red-900/20"
                }
              } else if (isSelected) {
                optionClass = "border-blue-500 bg-blue-50 dark:bg-blue-900/20"
              }

              return (
                <button
                  key={option.id}
                  onClick={() => handleOptionSelect(option.id)}
                  disabled={showResult}
                  className={`w-full p-4 rounded-lg border-2 text-left transition-all ${optionClass}`}
                >
                  <div className="flex items-center gap-3">
                    <div className={`w-5 h-5 rounded-full border-2 flex items-center justify-center ${
                      isSelected ? 'border-blue-500' : 'border-gray-300'
                    }`}>
                      {isSelected && <div className="w-2.5 h-2.5 rounded-full bg-blue-500" />}
                    </div>
                    <span className="text-gray-900 dark:text-white">{option.text}</span>
                    {showResult && isCorrect && (
                      <Check className="h-5 w-5 text-green-500 ml-auto" />
                    )}
                  </div>
                </button>
              )
            })}
          </div>

          {/* Submit / Confidence */}
          {!showResult ? (
            <Button
              onClick={handleSubmit}
              disabled={!selectedOption}
              className="w-full bg-blue-600 hover:bg-blue-700"
            >
              Check Answer
            </Button>
          ) : (
            <div className="space-y-3">
              <p className="text-sm text-gray-600 dark:text-gray-400 text-center">
                How confident were you?
              </p>
              <div className="flex justify-center gap-2">
                {[
                  { value: 1, emoji: "😰", label: "Guessed" },
                  { value: 2, emoji: "😕", label: "Unsure" },
                  { value: 3, emoji: "😐", label: "Somewhat" },
                  { value: 4, emoji: "😊", label: "Confident" },
                  { value: 5, emoji: "🤩", label: "Certain" },
                ].map((conf) => (
                  <button
                    key={conf.value}
                    onClick={() => handleConfidenceSelect(conf.value)}
                    className={`flex flex-col items-center p-2 rounded-lg border transition-all ${
                      confidence === conf.value
                        ? 'border-blue-500 bg-blue-50 dark:bg-blue-900/20'
                        : 'border-gray-200 dark:border-gray-700 hover:border-blue-300'
                    }`}
                  >
                    <span className="text-xl">{conf.emoji}</span>
                    <span className="text-xs text-gray-500 dark:text-gray-400 mt-1">{conf.label}</span>
                  </button>
                ))}
              </div>
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  )
}

export default function ReviewSessionClient({ onSessionStart, onSessionEnd, autoStart, questionCount = 10 }: ReviewSessionClientProps) {
  const [questions, setQuestions] = useState<ReviewQuestion[]>([])
  const [summary, setSummary] = useState<SessionSummary | null>(null)
  const [questionResults, setQuestionResults] = useState<Array<{ isCorrect: boolean }>>([])
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [totalSessionQuestions, setTotalSessionQuestions] = useState(0)
  const [wasPartialQuit, setWasPartialQuit] = useState(false)

  // Auto-start on mount
  useEffect(() => {
    if (autoStart && !isLoading && questions.length === 0 && !summary) {
      generateSession()
    }
  }, [autoStart])

  const generateSession = async () => {
    setIsLoading(true)
    setError(null)
    setQuestions([])
    setSummary(null)
    setQuestionResults([])
    setWasPartialQuit(false)
    setTotalSessionQuestions(0)

    onSessionStart?.()

    try {
      // Fetch questions from API
      const response = await fetch(`/api/review/questions?count=${questionCount}`)

      if (!response.ok) {
        if (response.status === 404) {
          setError('no_items')
          return
        }
        throw new Error('Failed to generate session')
      }

      const data = await response.json()

      if (!data.questions || data.questions.length === 0) {
        setError('no_items')
        return
      }

      setQuestions(data.questions)
      setTotalSessionQuestions(data.questions.length)
    } catch (err) {
      console.error('Failed to generate session:', err)
      setError('failed')
    } finally {
      setIsLoading(false)
    }
  }

  const handleCompleteSession = (results: Array<{ questionId: string; isCorrect: boolean; confidence: number; timeSpent: number }>) => {
    setQuestionResults(results.map(r => ({ isCorrect: r.isCorrect })))

    const correct = results.filter(r => r.isCorrect).length
    const accuracyRate = Math.round((correct / results.length) * 100)
    const averageConfidence = results.reduce((sum, r) => sum + r.confidence, 0) / results.length
    const totalTimeSpent = results.reduce((sum, r) => sum + r.timeSpent, 0)

    setSummary({
      itemsCompleted: results.length,
      accuracyRate,
      averageConfidence,
      totalTimeSpent
    })
    setQuestions([])
  }

  const handleRunnerQuit = (partialResults: Array<{ questionId: string; isCorrect: boolean; confidence: number; timeSpent: number }>) => {
    if (partialResults.length > 0) {
      setQuestionResults(partialResults.map(r => ({ isCorrect: r.isCorrect })))
      setWasPartialQuit(true)

      const correct = partialResults.filter(r => r.isCorrect).length
      const accuracyRate = Math.round((correct / partialResults.length) * 100)
      const averageConfidence = partialResults.reduce((sum, r) => sum + r.confidence, 0) / partialResults.length
      const totalTimeSpent = partialResults.reduce((sum, r) => sum + r.timeSpent, 0)

      setSummary({
        itemsCompleted: partialResults.length,
        accuracyRate,
        averageConfidence,
        totalTimeSpent
      })
      setQuestions([])
    } else {
      setQuestions([])
      onSessionEnd?.(null)
    }
  }

  const handleDone = () => {
    setSummary(null)
    setQuestionResults([])
    setWasPartialQuit(false)
    setTotalSessionQuestions(0)
    onSessionEnd?.(null)
  }

  // Loading
  if (isLoading) {
    return (
      <Card className="border-blue-100 dark:border-blue-900/50 bg-gradient-to-b from-blue-50 to-white dark:from-blue-900/20 dark:to-gray-900">
        <CardContent className="p-12">
          <div className="flex flex-col items-center justify-center space-y-4">
            <Loader2 className="h-12 w-12 animate-spin text-blue-600" />
            <p className="text-lg font-medium text-gray-900 dark:text-white">
              Generating your session...
            </p>
            <p className="text-sm text-gray-600 dark:text-gray-400">
              Selecting optimal questions
            </p>
          </div>
        </CardContent>
      </Card>
    )
  }

  // No items at all
  if (error === 'no_items') {
    return (
      <Card className="border-green-100 dark:border-green-900/50 bg-green-50 dark:bg-green-900/20">
        <CardContent className="p-8">
          <div className="flex flex-col items-center text-center space-y-4">
            <div className="flex items-center justify-center w-16 h-16 rounded-full bg-green-100 dark:bg-green-900/50">
              <Target className="h-8 w-8 text-green-600 dark:text-green-400" />
            </div>
            <h3 className="text-2xl font-bold text-gray-900 dark:text-white">
              Start Your Journey!
            </h3>
            <p className="text-gray-600 dark:text-gray-400">
              Complete labs to start using spaced repetition
            </p>
            <p className="text-sm text-gray-600 dark:text-gray-400 max-w-md">
              Browse courses and complete labs to add questions to your review queue.
            </p>
            <Button asChild className="bg-blue-600 hover:bg-blue-700">
              <Link href="/">
                <Target className="mr-2 h-4 w-4" />
                Explore Courses
              </Link>
            </Button>
          </div>
        </CardContent>
      </Card>
    )
  }

  // Error
  if (error && error !== 'no_items') {
    return (
      <Card className="border-red-100 dark:border-red-900/50 bg-red-50 dark:bg-red-900/20">
        <CardContent className="p-12">
          <div className="flex flex-col items-center justify-center space-y-4">
            <AlertTriangle className="h-12 w-12 text-red-600 dark:text-red-400" />
            <p className="text-lg font-medium text-red-900 dark:text-red-200">
              Failed to load session
            </p>
            <Button onClick={generateSession} variant="outline">
              <RefreshCw className="mr-2 h-4 w-4" />
              Try Again
            </Button>
          </div>
        </CardContent>
      </Card>
    )
  }

  // Animated Session Summary
  if (summary) {
    return (
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5 }}
        className="max-w-lg mx-auto space-y-6"
      >
        <Card className="border-gray-200 dark:border-gray-700 shadow-lg overflow-hidden">
          <CardContent className="p-8 space-y-8">
            {/* Partial session note */}
            {wasPartialQuit && totalSessionQuestions > 0 && (
              <motion.div
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                className="text-center text-sm text-amber-700 dark:text-amber-300 bg-amber-50 dark:bg-amber-900/20 border border-amber-200 dark:border-amber-800 rounded-lg p-3"
              >
                Partial session: {summary.itemsCompleted} of {totalSessionQuestions} questions completed
              </motion.div>
            )}

            {/* Circular Progress Ring */}
            <div className="flex justify-center">
              <CircularProgressRing percentage={summary.accuracyRate} />
            </div>

            {/* Per-question Dot Strip */}
            {questionResults.length > 0 && (
              <div className="flex items-center justify-center gap-1.5 flex-wrap">
                {questionResults.map((r, idx) => (
                  <motion.div
                    key={idx}
                    initial={{ scale: 0 }}
                    animate={{ scale: 1 }}
                    transition={{ delay: 1.2 + idx * 0.08, type: "spring", stiffness: 300 }}
                    className={`w-3 h-3 rounded-full ${r.isCorrect ? 'bg-green-500' : 'bg-red-500'}`}
                  />
                ))}
              </div>
            )}

            {/* 2x2 Stat Grid */}
            <div className="grid grid-cols-2 gap-3">
              <motion.div
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.4 }}
                className="p-4 rounded-xl bg-blue-50 dark:bg-blue-900/20 border border-blue-100 dark:border-blue-800 text-center"
              >
                <div className="text-2xl font-bold text-blue-700 dark:text-blue-300">{summary.itemsCompleted}</div>
                <div className="text-xs font-medium text-blue-600 dark:text-blue-400 uppercase tracking-wider mt-1">
                  Questions
                </div>
              </motion.div>
              <motion.div
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.5 }}
                className="p-4 rounded-xl bg-green-50 dark:bg-green-900/20 border border-green-100 dark:border-green-800 text-center"
              >
                <div className="text-2xl font-bold text-green-700 dark:text-green-300">{summary.accuracyRate}%</div>
                <div className="text-xs font-medium text-green-600 dark:text-green-400 uppercase tracking-wider mt-1">
                  Accuracy
                </div>
              </motion.div>
              <motion.div
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.6 }}
                className="p-4 rounded-xl bg-purple-50 dark:bg-purple-900/20 border border-purple-100 dark:border-purple-800 text-center"
              >
                <div className="text-2xl font-bold text-purple-700 dark:text-purple-300">{summary.averageConfidence.toFixed(1)}</div>
                <div className="text-xs font-medium text-purple-600 dark:text-purple-400 uppercase tracking-wider mt-1">
                  Confidence
                </div>
              </motion.div>
              <motion.div
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.7 }}
                className="p-4 rounded-xl bg-amber-50 dark:bg-amber-900/20 border border-amber-100 dark:border-amber-800 text-center"
              >
                <div className="text-2xl font-bold text-amber-700 dark:text-amber-300">{Math.round(summary.totalTimeSpent / 60)}m</div>
                <div className="text-xs font-medium text-amber-600 dark:text-amber-400 uppercase tracking-wider mt-1">
                  Time
                </div>
              </motion.div>
            </div>

            {/* Actions */}
            <div className="flex flex-col sm:flex-row gap-3">
              <Button onClick={generateSession} className="flex-1 bg-blue-600 hover:bg-blue-700">
                <RefreshCw className="mr-2 h-4 w-4" />
                Review More
              </Button>
              <Button onClick={handleDone} className="flex-1" variant="outline">
                Done
              </Button>
            </div>
          </CardContent>
        </Card>
      </motion.div>
    )
  }

  // Active session
  if (questions.length > 0) {
    return (
      <QuestionRunner
        questions={questions}
        onComplete={handleCompleteSession}
        onQuit={handleRunnerQuit}
      />
    )
  }

  return null
}
