"use client"

import { useState, useEffect } from "react"
import Link from "next/link"
import { useTheme } from "next-themes"
import { Card, CardContent } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Brain, BarChart2, PlayCircle, Clock, Loader2, Minus, Plus, ArrowLeft, Moon, Sun, BookOpen } from "lucide-react"
import ReviewSessionClient from "@/components/cbr/ReviewSessionClient"
import { SpacedRepetitionInfo } from "@/components/cbr/SpacedRepetitionInfo"
import { Skeleton } from "@/components/ui/skeleton"
import { Slider } from "@/components/ui/slider"
import { motion, AnimatePresence } from "framer-motion"

interface DueInfo {
  dueCount: number
  isLoading: boolean
}

export default function ReviewPage() {
  const { theme, setTheme } = useTheme()
  const [isSessionActive, setIsSessionActive] = useState(false)
  const [dueInfo, setDueInfo] = useState<DueInfo>({ dueCount: 0, isLoading: true })
  const [showHowItWorks, setShowHowItWorks] = useState(false)
  const [showCountPicker, setShowCountPicker] = useState(false)
  const [selectedCount, setSelectedCount] = useState(10)
  const [shouldAutoStart, setShouldAutoStart] = useState(false)

  // Show "How it works" dialog on first visit
  useEffect(() => {
    const seen = localStorage.getItem('review-how-it-works-seen')
    if (!seen) {
      setShowHowItWorks(true)
    }
  }, [])

  // Fetch available question count
  useEffect(() => {
    const fetchDueCount = async () => {
      try {
        const res = await fetch('/api/review/questions?count=1')
        if (res.ok) {
          const data = await res.json()
          setDueInfo({
            dueCount: data.totalAvailable || 0,
            isLoading: false
          })
        } else {
          setDueInfo(prev => ({ ...prev, dueCount: 0, isLoading: false }))
        }
      } catch {
        setDueInfo(prev => ({ ...prev, dueCount: 0, isLoading: false }))
      }
    }

    fetchDueCount()
  }, [])

  const handleSessionStart = () => {
    setIsSessionActive(true)
  }

  const handleSessionEnd = () => {
    setIsSessionActive(false)
    setShouldAutoStart(false)
    setShowCountPicker(false)
  }

  const maxCount = Math.min(dueInfo.dueCount, 50)

  const handleStartReview = () => {
    if (dueInfo.dueCount <= 1) {
      setSelectedCount(1)
      setShouldAutoStart(true)
      setIsSessionActive(true)
    } else {
      setSelectedCount(Math.min(10, dueInfo.dueCount))
      setShowCountPicker(true)
    }
  }

  const handleBeginSession = () => {
    setShowCountPicker(false)
    setShouldAutoStart(true)
    setIsSessionActive(true)
  }

  return (
    <div className="min-h-screen bg-gradient-to-b from-blue-50 to-white dark:from-gray-900 dark:to-gray-950">
      {/* Header */}
      <header className="bg-white/80 dark:bg-gray-900/80 backdrop-blur-md border-b border-gray-100 dark:border-gray-800 sticky top-0 z-50">
        <div className="container mx-auto px-4 py-4 flex items-center justify-between">
          <div className="flex items-center gap-4">
            <Link href="/">
              <Button variant="ghost" size="sm" className="text-gray-600 hover:text-gray-900 dark:text-gray-400 dark:hover:text-white">
                <ArrowLeft className="h-4 w-4 mr-2" />
                Back
              </Button>
            </Link>
            <div className="flex items-center gap-3">
              <div className="p-2 bg-gradient-to-br from-blue-500 to-blue-600 rounded-xl shadow-lg shadow-blue-500/20">
                <Brain className="h-5 w-5 text-white" />
              </div>
              <div>
                <h1 className="text-lg font-bold text-gray-900 dark:text-white">Review Center</h1>
                <p className="text-xs text-gray-500 dark:text-gray-400">Spaced Repetition</p>
              </div>
            </div>
          </div>
          <Button
            variant="ghost"
            size="icon"
            className="text-gray-600 hover:text-gray-900 dark:text-gray-400 dark:hover:text-white"
            onClick={() => setTheme(theme === 'dark' ? 'light' : 'dark')}
          >
            <Sun className="h-5 w-5 rotate-0 scale-100 transition-all dark:-rotate-90 dark:scale-0" />
            <Moon className="absolute h-5 w-5 rotate-90 scale-0 transition-all dark:rotate-0 dark:scale-100" />
          </Button>
        </div>
      </header>

      <main className="container mx-auto py-6 px-4 max-w-4xl">
        <AnimatePresence mode="wait">
          {/* ============ SESSION MODE ============ */}
          {isSessionActive && (
            <motion.div
              key="session"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              transition={{ duration: 0.3 }}
              className="min-h-[80vh]"
            >
              <ReviewSessionClient
                onSessionStart={handleSessionStart}
                onSessionEnd={handleSessionEnd}
                autoStart={shouldAutoStart}
                questionCount={selectedCount}
              />
            </motion.div>
          )}

          {/* ============ COUNT PICKER MODE ============ */}
          {showCountPicker && !isSessionActive && (
            <motion.div
              key="picker"
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -10 }}
              transition={{ duration: 0.3 }}
              className="flex flex-col gap-6"
            >
              {/* Header */}
              <div>
                <h1 className="text-3xl font-bold tracking-tight text-gray-900 dark:text-white">
                  Review Center
                </h1>
                <p className="text-gray-600 dark:text-gray-400">
                  Master your knowledge with spaced repetition
                </p>
              </div>

              <Card className="border-2 border-blue-500 bg-gradient-to-r from-blue-50 to-white dark:from-blue-900/20 dark:to-gray-900">
                <CardContent className="p-6 sm:p-8">
                  <div className="flex flex-col items-center text-center gap-5">
                    <div className="flex items-center justify-center w-14 h-14 rounded-full bg-blue-100 dark:bg-blue-900/50">
                      <Brain className="h-7 w-7 text-blue-600 dark:text-blue-400" />
                    </div>

                    <div>
                      <h2 className="text-xl sm:text-2xl font-bold tracking-tight text-gray-900 dark:text-white">
                        How many questions?
                      </h2>
                      <p className="text-sm text-gray-600 dark:text-gray-400 mt-1">
                        Choose the number of questions for this session
                      </p>
                    </div>

                    {/* Count display with +/- buttons */}
                    <div className="flex items-center gap-4">
                      <Button
                        variant="outline"
                        size="icon"
                        onClick={() => setSelectedCount(c => Math.max(1, c - 1))}
                        disabled={selectedCount <= 1}
                        className="h-10 w-10 rounded-full"
                      >
                        <Minus className="h-4 w-4" />
                      </Button>
                      <span className="text-4xl sm:text-5xl font-bold text-blue-600 dark:text-blue-400 tabular-nums w-16 text-center">
                        {selectedCount}
                      </span>
                      <Button
                        variant="outline"
                        size="icon"
                        onClick={() => setSelectedCount(c => Math.min(maxCount, c + 1))}
                        disabled={selectedCount >= maxCount}
                        className="h-10 w-10 rounded-full"
                      >
                        <Plus className="h-4 w-4" />
                      </Button>
                    </div>

                    {/* Slider */}
                    <div className="w-full max-w-xs">
                      <Slider
                        value={[selectedCount]}
                        onValueChange={(v) => setSelectedCount(v[0])}
                        min={1}
                        max={maxCount}
                        step={1}
                      />
                      <div className="flex justify-between text-xs text-gray-500 dark:text-gray-400 mt-1.5">
                        <span>1</span>
                        <span>{maxCount} available</span>
                      </div>
                    </div>

                    {/* Actions */}
                    <div className="flex flex-col gap-2 w-full max-w-xs">
                      <Button
                        onClick={handleBeginSession}
                        size="lg"
                        className="bg-blue-600 hover:bg-blue-700 shadow-lg shadow-blue-500/20 h-12 text-lg"
                      >
                        <PlayCircle className="mr-2 h-5 w-5" />
                        Begin Session
                      </Button>
                      <Button
                        variant="ghost"
                        onClick={() => setShowCountPicker(false)}
                        className="text-gray-500"
                      >
                        Back
                      </Button>
                    </div>
                  </div>
                </CardContent>
              </Card>
            </motion.div>
          )}

          {/* ============ HUB MODE ============ */}
          {!isSessionActive && !showCountPicker && (
            <motion.div
              key="hub"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              transition={{ duration: 0.3 }}
              className="flex flex-col gap-6"
            >
              {/* Header */}
              <div className="flex items-center justify-between">
                <div>
                  <h1 className="text-3xl font-bold tracking-tight text-gray-900 dark:text-white">
                    Review Center
                  </h1>
                  <p className="text-gray-600 dark:text-gray-400">
                    Master your knowledge with spaced repetition
                  </p>
                </div>
                <SpacedRepetitionInfo
                  open={showHowItWorks}
                  onOpenChange={(open) => {
                    setShowHowItWorks(open)
                    if (!open) {
                      localStorage.setItem('review-how-it-works-seen', 'true')
                    }
                  }}
                />
              </div>

              {/* Hero Action Card */}
              <Card className={`border-2 ${dueInfo.dueCount > 0 ? 'border-blue-500 bg-gradient-to-r from-blue-50 to-white dark:from-blue-900/20 dark:to-gray-900' : 'border-gray-200 dark:border-gray-700 bg-gray-50 dark:bg-gray-800'}`}>
                <CardContent className="p-8">
                  <div className="flex flex-col items-center text-center gap-4">
                    <div className={`flex items-center justify-center w-16 h-16 rounded-full ${dueInfo.dueCount > 0 ? 'bg-blue-100 dark:bg-blue-900/50' : 'bg-gray-200 dark:bg-gray-700'}`}>
                      {dueInfo.isLoading ? (
                        <Loader2 className="h-8 w-8 animate-spin text-blue-600 dark:text-blue-400" />
                      ) : (
                        <Brain className={`h-8 w-8 ${dueInfo.dueCount > 0 ? 'text-blue-600 dark:text-blue-400' : 'text-gray-400'}`} />
                      )}
                    </div>

                    {dueInfo.isLoading ? (
                      <div className="space-y-2">
                        <Skeleton className="h-10 w-40 mx-auto" />
                        <Skeleton className="h-4 w-56 mx-auto" />
                      </div>
                    ) : dueInfo.dueCount > 0 ? (
                      <>
                        <div>
                          <div className="flex items-baseline justify-center gap-2">
                            <span className="text-5xl font-bold text-blue-600 dark:text-blue-400">{dueInfo.dueCount}</span>
                            <span className="text-lg font-medium text-gray-700 dark:text-gray-300">
                              {dueInfo.dueCount === 1 ? 'question available' : 'questions available'}
                            </span>
                          </div>
                          <p className="text-sm text-gray-600 dark:text-gray-400 flex items-center justify-center gap-1 mt-1">
                            <Clock className="h-3.5 w-3.5" />
                            Ready to review
                          </p>
                        </div>
                        <Button
                          size="lg"
                          className="bg-blue-600 hover:bg-blue-700 shadow-lg shadow-blue-500/20 px-10 h-12 text-lg"
                          onClick={handleStartReview}
                        >
                          <PlayCircle className="mr-2 h-5 w-5" />
                          Start Review
                        </Button>
                      </>
                    ) : (
                      <>
                        <div>
                          <div className="text-xl font-semibold text-blue-600 dark:text-blue-400">
                            No questions available
                          </div>
                          <p className="text-sm text-gray-600 dark:text-gray-400 mt-1">
                            Browse courses to add questions to review
                          </p>
                        </div>
                        <Button className="bg-blue-600 hover:bg-blue-700" asChild>
                          <Link href="/">
                            <BookOpen className="mr-2 h-4 w-4" />
                            Explore Courses
                          </Link>
                        </Button>
                      </>
                    )}
                  </div>
                </CardContent>
              </Card>

              {/* Info Section */}
              <Tabs defaultValue="about" className="space-y-6">
                <TabsList className="grid w-full max-w-md grid-cols-2">
                  <TabsTrigger value="about" className="flex items-center gap-1.5">
                    <Brain className="h-4 w-4" />
                    <span className="hidden sm:inline">About</span>
                  </TabsTrigger>
                  <TabsTrigger value="stats" className="flex items-center gap-1.5">
                    <BarChart2 className="h-4 w-4" />
                    <span className="hidden sm:inline">How It Works</span>
                  </TabsTrigger>
                </TabsList>

                <TabsContent value="about">
                  <Card className="border-gray-200 dark:border-gray-700">
                    <CardContent className="p-6 space-y-4">
                      <h3 className="text-lg font-semibold text-gray-900 dark:text-white">What is Spaced Repetition?</h3>
                      <p className="text-gray-600 dark:text-gray-400">
                        Spaced repetition is a learning technique that incorporates increasing intervals of time between subsequent review of previously learned material. This exploits the psychological spacing effect to optimize long-term retention.
                      </p>
                      <div className="grid grid-cols-1 md:grid-cols-3 gap-4 pt-4">
                        <div className="p-4 rounded-lg bg-blue-50 dark:bg-blue-900/20 border border-blue-100 dark:border-blue-800">
                          <div className="text-2xl font-bold text-blue-600 dark:text-blue-400">2x</div>
                          <p className="text-sm text-gray-600 dark:text-gray-400">Better retention vs cramming</p>
                        </div>
                        <div className="p-4 rounded-lg bg-green-50 dark:bg-green-900/20 border border-green-100 dark:border-green-800">
                          <div className="text-2xl font-bold text-green-600 dark:text-green-400">50%</div>
                          <p className="text-sm text-gray-600 dark:text-gray-400">Less study time needed</p>
                        </div>
                        <div className="p-4 rounded-lg bg-purple-50 dark:bg-purple-900/20 border border-purple-100 dark:border-purple-800">
                          <div className="text-2xl font-bold text-purple-600 dark:text-purple-400">90%</div>
                          <p className="text-sm text-gray-600 dark:text-gray-400">Long-term recall rate</p>
                        </div>
                      </div>
                    </CardContent>
                  </Card>
                </TabsContent>

                <TabsContent value="stats">
                  <Card className="border-gray-200 dark:border-gray-700">
                    <CardContent className="p-6 space-y-4">
                      <h3 className="text-lg font-semibold text-gray-900 dark:text-white">The Algorithm</h3>
                      <div className="space-y-3">
                        <div className="flex items-start gap-3">
                          <div className="w-8 h-8 rounded-full bg-blue-100 dark:bg-blue-900/50 flex items-center justify-center flex-shrink-0">
                            <span className="text-sm font-bold text-blue-600 dark:text-blue-400">1</span>
                          </div>
                          <div>
                            <p className="font-medium text-gray-900 dark:text-white">Answer + Confidence</p>
                            <p className="text-sm text-gray-600 dark:text-gray-400">Rate your confidence after each answer to help the algorithm understand your true knowledge.</p>
                          </div>
                        </div>
                        <div className="flex items-start gap-3">
                          <div className="w-8 h-8 rounded-full bg-blue-100 dark:bg-blue-900/50 flex items-center justify-center flex-shrink-0">
                            <span className="text-sm font-bold text-blue-600 dark:text-blue-400">2</span>
                          </div>
                          <div>
                            <p className="font-medium text-gray-900 dark:text-white">Smart Scheduling</p>
                            <p className="text-sm text-gray-600 dark:text-gray-400">Correct + confident = longer interval. Wrong = reset. The algorithm optimizes review timing.</p>
                          </div>
                        </div>
                        <div className="flex items-start gap-3">
                          <div className="w-8 h-8 rounded-full bg-blue-100 dark:bg-blue-900/50 flex items-center justify-center flex-shrink-0">
                            <span className="text-sm font-bold text-blue-600 dark:text-blue-400">3</span>
                          </div>
                          <div>
                            <p className="font-medium text-gray-900 dark:text-white">Priority System</p>
                            <p className="text-sm text-gray-600 dark:text-gray-400">Questions you got wrong with high confidence are flagged and prioritized.</p>
                          </div>
                        </div>
                      </div>
                    </CardContent>
                  </Card>
                </TabsContent>
              </Tabs>
            </motion.div>
          )}
        </AnimatePresence>
      </main>
    </div>
  )
}
