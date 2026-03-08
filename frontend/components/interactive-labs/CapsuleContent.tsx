"use client"

import { useState, useEffect, useRef } from "react"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Progress } from "@/components/ui/progress"
import { CheckCircle, XCircle, AlertCircle, ArrowRight, ArrowLeft, Trophy, Loader2 } from "lucide-react"
import { useRouter } from "next/navigation"
import Link from "next/link"
import type { Question, QuestionAnswerOption } from "@/lib/types"
import { completeCapsule, getCurrentQuestionIndex, saveCurrentQuestionIndex, clearCurrentQuestionIndex, getCapsuleAnswers } from "@/lib/progress"
import { motion, AnimatePresence } from "framer-motion"

interface CapsuleContentProps {
  content?: string
  questions: Question[]
  labId: string
  capsuleId: string
  courseSlug: string
  nextCapsuleId?: string
  prevCapsuleId?: string
  isCompleted: boolean
  userSelectedAnswers: Record<string, string | string[]>
  isLabCompleted: boolean
  score?: number
  timeSpent?: number
}

export default function CapsuleContent({
  content,
  questions,
  labId,
  capsuleId,
  courseSlug,
  nextCapsuleId,
  prevCapsuleId,
  isCompleted,
  userSelectedAnswers,
  isLabCompleted,
  score,
  timeSpent
}: CapsuleContentProps) {
  const router = useRouter()
  const [currentQuestionIndex, setCurrentQuestionIndex] = useState(() =>
    isCompleted ? 0 : getCurrentQuestionIndex(capsuleId)
  )
  const [selectedAnswers, setSelectedAnswers] = useState<Record<string, string | string[]>>(userSelectedAnswers || {})
  const [showResults, setShowResults] = useState(isCompleted)
  const [showQuestionReasoning, setShowQuestionReasoning] = useState<Record<string, boolean>>({})
  const [questionConfidences, setQuestionConfidences] = useState<Record<string, number>>({})
  const [startTime] = useState(Date.now())
  const [isSubmitting, setIsSubmitting] = useState(false)
  const isSubmittingRef = useRef(false)

  const currentQuestion = questions[currentQuestionIndex]
  const totalQuestions = questions.length

  // Save current question index to localStorage whenever it changes
  useEffect(() => {
    if (!isCompleted) {
      saveCurrentQuestionIndex(capsuleId, currentQuestionIndex)
    }
  }, [currentQuestionIndex, capsuleId, isCompleted])

  // Clear saved progress when completing the capsule
  useEffect(() => {
    if (isCompleted) {
      clearCurrentQuestionIndex(capsuleId)
    }
  }, [isCompleted, capsuleId])

  const handleAnswerSelect = (questionId: string, answerId: string) => {
    const question = questions.find(q => q.id === questionId)

    if (question?.multipleCorrect) {
      const currentSelections = Array.isArray(selectedAnswers[questionId])
        ? selectedAnswers[questionId] as string[]
        : selectedAnswers[questionId]
          ? [selectedAnswers[questionId] as string]
          : []

      const maxSelections = question.correctAnswersCount || 1

      if (currentSelections.includes(answerId)) {
        const newSelections = currentSelections.filter(id => id !== answerId)
        setSelectedAnswers(prev => ({
          ...prev,
          [questionId]: newSelections
        }))
      } else {
        if (currentSelections.length < maxSelections) {
          const newSelections = [...currentSelections, answerId]
          setSelectedAnswers(prev => ({
            ...prev,
            [questionId]: newSelections
          }))
        }
      }
    } else {
      setSelectedAnswers(prev => ({
        ...prev,
        [questionId]: answerId
      }))
    }
  }

  const getAnswerStatus = (question: Question): 'correct' | 'partially_correct' | 'incorrect' | 'unanswered' => {
    const selectedAnswer = selectedAnswers[question.id]
    const correctAnswers = question.answerOptions.filter(option => option.isCorrect)

    if (!selectedAnswer || (Array.isArray(selectedAnswer) && selectedAnswer.length === 0)) {
      return 'unanswered'
    }

    if (question.multipleCorrect) {
      const selectedArray = Array.isArray(selectedAnswer) ? selectedAnswer : [selectedAnswer]
      const correctAnswerIds = correctAnswers.map(a => a.id)

      const isFullyCorrect = correctAnswerIds.every(id => selectedArray.includes(id)) &&
        selectedArray.every(id => correctAnswerIds.includes(id))

      if (isFullyCorrect) return 'correct'

      const hasAnyCorrect = selectedArray.some(id => correctAnswerIds.includes(id))
      if (hasAnyCorrect) return 'partially_correct'

      return 'incorrect'
    } else {
      const correctAnswer = correctAnswers[0]
      return selectedAnswer === correctAnswer?.id ? 'correct' : 'incorrect'
    }
  }

  const calculateScore = () => {
    let correct = 0
    let partiallyCorrect = 0

    questions.forEach(question => {
      const status = getAnswerStatus(question)
      if (status === 'correct') {
        correct++
      } else if (status === 'partially_correct') {
        partiallyCorrect++
        correct += 0.5
      }
    })

    return {
      correct: Math.floor(correct),
      partiallyCorrect,
      total: totalQuestions,
      percentage: Math.round((correct / totalQuestions) * 100)
    }
  }

  const handleQuestionComplete = () => {
    const currentQuestionId = currentQuestion?.id
    if (currentQuestionId) {
      setShowQuestionReasoning(prev => ({
        ...prev,
        [currentQuestionId]: true
      }))
    }
  }

  const handleContinueToNext = async () => {
    if (isSubmitting) return

    if (currentQuestionIndex < totalQuestions - 1) {
      setCurrentQuestionIndex(prevIndex => prevIndex + 1)
    } else {
      // Last question - complete the capsule
      await saveCompletion()

      if (isLabCompleted) {
        router.push(`/courses/${courseSlug}/labs/${labId}/results`)
      } else {
        setShowResults(true)
      }
    }
  }

  const handlePrevious = () => {
    if (currentQuestionIndex > 0) {
      setCurrentQuestionIndex(prevIndex => prevIndex - 1)
    }
  }

  const saveCompletion = async () => {
    if (isSubmittingRef.current) return

    setIsSubmitting(true)
    isSubmittingRef.current = true

    const calculatedScore = calculateScore()
    const currentSessionTime = Math.round((Date.now() - startTime) / 1000)
    const totalTimeSpent = (timeSpent || 0) + currentSessionTime

    // Save to localStorage
    completeCapsule(
      capsuleId,
      calculatedScore.percentage,
      totalTimeSpent,
      0,
      selectedAnswers
    )

    setIsSubmitting(false)
    isSubmittingRef.current = false
  }

  // If no questions, show placeholder
  if (totalQuestions === 0) {
    return (
      <div className="p-6 bg-gray-50 dark:bg-gray-900 rounded-lg text-center">
        <p className="text-gray-600 dark:text-gray-400 mb-4">No questions available in this capsule.</p>
      </div>
    )
  }

  // Show results
  if (showResults) {
    const calculatedScore = calculateScore()
    const finalScore = score || calculatedScore.percentage
    const effectiveTimeSpent = timeSpent || Math.max(0, Math.round((Date.now() - startTime) / 1000))
    const timeSpentMinutes = Math.floor(effectiveTimeSpent / 60)
    const timeSpentSeconds = effectiveTimeSpent % 60

    return (
      <div className="space-y-4">
        {/* Main Results Card */}
        <Card>
          <CardHeader className="text-center pb-3">
            <div className="flex items-center justify-center gap-2 mb-2">
              <Trophy className="h-5 w-5 text-yellow-500" />
              <CardTitle className="text-lg">Capsule Complete!</CardTitle>
            </div>
            <div className="text-2xl font-bold text-blue-600">
              {calculatedScore.correct}/{calculatedScore.total}
            </div>
            <div className="text-sm text-gray-600 dark:text-gray-400">
              {((calculatedScore.correct / calculatedScore.total) * 100).toFixed(0)}% Correct
            </div>
            {calculatedScore.partiallyCorrect > 0 && (
              <Badge variant="secondary" className="mt-2 text-xs bg-amber-100 text-amber-800 dark:bg-amber-900/20 dark:text-amber-300">
                <AlertCircle className="h-3 w-3 mr-1" />
                {calculatedScore.partiallyCorrect} Partially Correct
              </Badge>
            )}
          </CardHeader>
          <CardContent className="pt-0">
            {/* Performance Metrics */}
            <div className="grid grid-cols-3 gap-2 mb-4">
              <div className="text-center p-2 bg-slate-50 dark:bg-slate-900/30 rounded-lg border">
                <div className="text-lg font-bold text-slate-700 dark:text-slate-300">
                  {finalScore}%
                </div>
                <div className="text-xs text-slate-600 dark:text-slate-400">
                  Accuracy
                </div>
              </div>
              <div className="text-center p-2 bg-slate-50 dark:bg-slate-900/30 rounded-lg border">
                <div className="text-lg font-bold text-slate-700 dark:text-slate-300">
                  {calculatedScore.correct}
                </div>
                <div className="text-xs text-slate-600 dark:text-slate-400">
                  Correct
                </div>
              </div>
              <div className="text-center p-2 bg-slate-50 dark:bg-slate-900/30 rounded-lg border">
                <div className="text-lg font-bold text-slate-700 dark:text-slate-300">
                  {timeSpentMinutes}:{timeSpentSeconds.toString().padStart(2, '0')}
                </div>
                <div className="text-xs text-slate-600 dark:text-slate-400">
                  Time
                </div>
              </div>
            </div>

            {/* Questions Review */}
            <div className="space-y-2 mb-4">
              <h3 className="text-sm font-semibold text-gray-700 dark:text-gray-300">Question Review</h3>
              {questions.map((question, index) => {
                const status = getAnswerStatus(question)
                return (
                  <div key={question.id} className="flex items-center gap-2 p-2 bg-gray-50 dark:bg-gray-800 rounded-lg">
                    <div className="flex-shrink-0">
                      {status === 'correct' && <CheckCircle className="h-4 w-4 text-green-500" />}
                      {status === 'incorrect' && <XCircle className="h-4 w-4 text-red-500" />}
                      {status === 'partially_correct' && <AlertCircle className="h-4 w-4 text-amber-500" />}
                      {status === 'unanswered' && <AlertCircle className="h-4 w-4 text-gray-400" />}
                    </div>
                    <span className="text-sm text-gray-700 dark:text-gray-300">Q{index + 1}</span>
                    <span className="text-xs text-gray-500 dark:text-gray-400 truncate flex-1">
                      {question.text.substring(0, 50)}...
                    </span>
                  </div>
                )
              })}
            </div>

            {/* Navigation */}
            <div className="flex flex-col sm:flex-row gap-2">
              {nextCapsuleId ? (
                <Button asChild className="flex-1 bg-blue-600 hover:bg-blue-700">
                  <Link href={`/courses/${courseSlug}/labs/${labId}/capsules/${nextCapsuleId}`}>
                    Next Capsule <ArrowRight className="h-4 w-4 ml-2" />
                  </Link>
                </Button>
              ) : (
                <Button asChild className="flex-1 bg-blue-600 hover:bg-blue-700">
                  <Link href={`/courses/${courseSlug}/labs/${labId}/results`}>
                    View Lab Results <ArrowRight className="h-4 w-4 ml-2" />
                  </Link>
                </Button>
              )}
              <Button variant="outline" asChild className="flex-1">
                <Link href={`/courses/${courseSlug}/labs/${labId}`}>
                  Back to Lab
                </Link>
              </Button>
            </div>
          </CardContent>
        </Card>
      </div>
    )
  }

  // Show question
  const selectedAnswer = selectedAnswers[currentQuestion.id]
  const isAnswered = selectedAnswer && (Array.isArray(selectedAnswer) ? selectedAnswer.length > 0 : true)
  const showReasoning = showQuestionReasoning[currentQuestion.id]
  const answerStatus = isAnswered ? getAnswerStatus(currentQuestion) : null

  return (
    <div className="space-y-4">
      {/* Clinical Situation / Context */}
      {content && (
        <Card className="bg-gradient-to-r from-slate-50 to-gray-50 dark:from-gray-800 dark:to-gray-900 border-gray-200 dark:border-gray-700">
          <CardContent className="p-4">
            <h3 className="text-sm font-bold text-gray-900 dark:text-white mb-2">Clinical Situation</h3>
            <div className="text-sm text-gray-700 dark:text-gray-300 whitespace-pre-wrap">
              {content}
            </div>
          </CardContent>
        </Card>
      )}

      {/* Question Progress */}
      <div className="flex items-center justify-between">
        <span className="text-sm font-medium text-gray-700 dark:text-gray-300">
          Question {currentQuestionIndex + 1} of {totalQuestions}
        </span>
        <Progress value={((currentQuestionIndex + 1) / totalQuestions) * 100} className="w-32 h-2" />
      </div>

      {/* Question Card */}
      <AnimatePresence mode="wait">
        <motion.div
          key={currentQuestion.id}
          initial={{ opacity: 0, x: 20 }}
          animate={{ opacity: 1, x: 0 }}
          exit={{ opacity: 0, x: -20 }}
          transition={{ duration: 0.2 }}
        >
          <Card>
            <CardContent className="p-4 space-y-4">
              {/* Question Text */}
              <div className="space-y-2">
                {currentQuestion.category && (
                  <Badge variant="outline" className="text-xs">
                    {currentQuestion.category}
                  </Badge>
                )}
                <p className="text-base font-medium text-gray-900 dark:text-white">
                  {currentQuestion.text}
                </p>
                {currentQuestion.multipleCorrect && (
                  <p className="text-xs text-blue-600 dark:text-blue-400">
                    Select {currentQuestion.correctAnswersCount || 'all'} correct answer(s)
                  </p>
                )}
              </div>

              {/* Answer Options */}
              <div className="space-y-2">
                {currentQuestion.answerOptions.map((option) => {
                  const isSelected = currentQuestion.multipleCorrect
                    ? Array.isArray(selectedAnswer) && selectedAnswer.includes(option.id)
                    : selectedAnswer === option.id

                  let optionClass = "border-gray-200 dark:border-gray-700 hover:border-blue-300 dark:hover:border-blue-600"

                  if (showReasoning) {
                    if (option.isCorrect) {
                      optionClass = "border-green-500 bg-green-50 dark:bg-green-900/20"
                    } else if (isSelected && !option.isCorrect) {
                      optionClass = "border-red-500 bg-red-50 dark:bg-red-900/20"
                    }
                  } else if (isSelected) {
                    optionClass = "border-blue-500 bg-blue-50 dark:bg-blue-900/20"
                  }

                  return (
                    <button
                      key={option.id}
                      onClick={() => !showReasoning && handleAnswerSelect(currentQuestion.id, option.id)}
                      disabled={showReasoning}
                      className={`w-full p-3 rounded-lg border-2 text-left transition-all ${optionClass}`}
                    >
                      <div className="flex items-center gap-3">
                        <div className={`w-5 h-5 rounded-full border-2 flex items-center justify-center flex-shrink-0 ${
                          isSelected ? 'border-blue-500' : 'border-gray-300 dark:border-gray-600'
                        }`}>
                          {isSelected && (
                            <div className="w-2.5 h-2.5 rounded-full bg-blue-500" />
                          )}
                        </div>
                        <span className="text-sm text-gray-900 dark:text-white">{option.text}</span>
                        {showReasoning && option.isCorrect && (
                          <CheckCircle className="h-4 w-4 text-green-500 ml-auto flex-shrink-0" />
                        )}
                        {showReasoning && isSelected && !option.isCorrect && (
                          <XCircle className="h-4 w-4 text-red-500 ml-auto flex-shrink-0" />
                        )}
                      </div>
                      {showReasoning && option.explanation && (
                        <p className="mt-2 text-xs text-gray-600 dark:text-gray-400 pl-8">
                          {option.explanation}
                        </p>
                      )}
                    </button>
                  )
                })}
              </div>

              {/* Reasoning/Explanation */}
              {showReasoning && currentQuestion.reasoning && (
                <div className="p-3 bg-blue-50 dark:bg-blue-900/20 rounded-lg border border-blue-200 dark:border-blue-800">
                  <h4 className="text-sm font-semibold text-blue-800 dark:text-blue-200 mb-2">Explanation</h4>
                  {currentQuestion.reasoning.rationale && (
                    <p className="text-sm text-blue-700 dark:text-blue-300">
                      {currentQuestion.reasoning.rationale}
                    </p>
                  )}
                </div>
              )}

              {/* Result Badge */}
              {showReasoning && answerStatus && (
                <div className="flex justify-center">
                  {answerStatus === 'correct' && (
                    <Badge className="bg-green-100 text-green-800 dark:bg-green-900/30 dark:text-green-200">
                      <CheckCircle className="h-3 w-3 mr-1" /> Correct
                    </Badge>
                  )}
                  {answerStatus === 'incorrect' && (
                    <Badge className="bg-red-100 text-red-800 dark:bg-red-900/30 dark:text-red-200">
                      <XCircle className="h-3 w-3 mr-1" /> Incorrect
                    </Badge>
                  )}
                  {answerStatus === 'partially_correct' && (
                    <Badge className="bg-amber-100 text-amber-800 dark:bg-amber-900/30 dark:text-amber-200">
                      <AlertCircle className="h-3 w-3 mr-1" /> Partially Correct
                    </Badge>
                  )}
                </div>
              )}

              {/* Action Buttons */}
              <div className="flex gap-2 pt-2">
                {currentQuestionIndex > 0 && !showReasoning && (
                  <Button variant="outline" onClick={handlePrevious} className="flex-1">
                    <ArrowLeft className="h-4 w-4 mr-2" />
                    Previous
                  </Button>
                )}

                {!showReasoning ? (
                  <Button
                    onClick={handleQuestionComplete}
                    disabled={!isAnswered}
                    className="flex-1 bg-blue-600 hover:bg-blue-700"
                  >
                    Check Answer
                  </Button>
                ) : (
                  <Button
                    onClick={handleContinueToNext}
                    disabled={isSubmitting}
                    className="flex-1 bg-blue-600 hover:bg-blue-700"
                  >
                    {isSubmitting && <Loader2 className="h-4 w-4 mr-2 animate-spin" />}
                    {currentQuestionIndex < totalQuestions - 1 ? (
                      <>Next Question <ArrowRight className="h-4 w-4 ml-2" /></>
                    ) : (
                      "Complete Capsule"
                    )}
                  </Button>
                )}
              </div>
            </CardContent>
          </Card>
        </motion.div>
      </AnimatePresence>
    </div>
  )
}
