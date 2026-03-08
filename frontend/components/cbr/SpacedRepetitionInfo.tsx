"use client"

import { useState } from "react"
import {
    Dialog,
    DialogContent,
    DialogHeader,
    DialogTitle,
    DialogTrigger,
} from "@/components/ui/dialog"
import { Button } from "@/components/ui/button"
import { Info, Brain, ChevronLeft, ChevronRight } from "lucide-react"
import { motion, AnimatePresence } from "framer-motion"

const BRAND = {
    primary: "#3b82f6",
    primaryDark: "#2563eb",
    primaryLight: "#60a5fa",
    bgSubtle: "#dbeafe",
    bg50: "#eff6ff",
    blue200: "#bfdbfe",
    blue300: "#93c5fd",
    blue800: "#1e40af",
    gray200: "#e5e7eb",
    gray400: "#9ca3af",
    gray500: "#6b7280",
    gray700: "#374151",
    amber400: "#fbbf24",
    amber100: "#fef3c7",
}

type Labels = Record<string, string>

// --- Slide 1: Answer + Confidence ---
function AnswerConfidenceIllustration({ labels }: { labels: Labels }) {
    return (
        <svg viewBox="0 0 260 160" className="w-full h-full" fill="none">
            <motion.rect x="15" y="15" width="110" height="70" rx="8"
                fill="white" stroke={BRAND.blue200} strokeWidth="1.5"
                initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.2 }}
            />
            <motion.text x="70" y="38" fontSize="9" fill={BRAND.gray500} fontWeight="600" textAnchor="middle"
                initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 0.3 }}>
                {labels.question}
            </motion.text>
            {[0, 1, 2, 3].map((i) => (
                <motion.rect key={i} x="25" y={46 + i * 9} width="90" height="6" rx="3"
                    fill={i === 1 ? BRAND.primary : BRAND.gray200}
                    initial={{ scaleX: 0 }} animate={{ scaleX: 1 }}
                    transition={{ delay: 0.4 + i * 0.08 }}
                    style={{ transformOrigin: "25px center" }}
                />
            ))}
            <motion.text x="120" y="60" fontSize="12" fill={BRAND.primary}
                initial={{ opacity: 0, scale: 0 }} animate={{ opacity: 1, scale: 1 }}
                transition={{ delay: 0.9, type: "spring" }}>
                ✓
            </motion.text>

            <motion.path d="M 130 50 L 148 50" stroke={BRAND.gray400} strokeWidth="1.5" strokeDasharray="4 3"
                initial={{ pathLength: 0 }} animate={{ pathLength: 1 }} transition={{ delay: 1.1, duration: 0.3 }}
            />
            <motion.path d="M 145 46 L 150 50 L 145 54" stroke={BRAND.gray400} strokeWidth="1.5" fill="none"
                initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 1.3 }}
            />

            {[
                { emoji: "😰", x: 163, delay: 1.4, bg: BRAND.bgSubtle },
                { emoji: "😕", x: 183, delay: 1.5, bg: BRAND.bgSubtle },
                { emoji: "😐", x: 203, delay: 1.6, bg: BRAND.bgSubtle },
                { emoji: "😊", x: 223, delay: 1.7, bg: BRAND.blue200 },
                { emoji: "🤩", x: 243, delay: 1.8, bg: BRAND.bgSubtle },
            ].map((item, i) => (
                <motion.g key={i}>
                    <motion.rect x={item.x - 8} y="35" width="16" height="22" rx="4"
                        fill={item.bg} stroke={BRAND.blue200} strokeWidth="1"
                        initial={{ scale: 0 }} animate={{ scale: 1 }}
                        transition={{ delay: item.delay, type: "spring", stiffness: 300 }}
                    />
                    <motion.text x={item.x} y="51" fontSize="10" textAnchor="middle"
                        initial={{ scale: 0 }} animate={{ scale: 1 }}
                        transition={{ delay: item.delay + 0.05, type: "spring" }}>
                        {item.emoji}
                    </motion.text>
                </motion.g>
            ))}

            <motion.rect x="213" y="33" width="20" height="26" rx="6"
                fill="none" stroke={BRAND.primary} strokeWidth="2"
                initial={{ opacity: 0 }} animate={{ opacity: [0, 0, 1] }}
                transition={{ delay: 2.2, duration: 0.3 }}
            />

            <motion.path d="M 70 88 L 70 110 Q 70 118 80 118 L 118 118"
                stroke={BRAND.primary} strokeWidth="1.5" fill="none"
                initial={{ pathLength: 0 }} animate={{ pathLength: 1 }} transition={{ delay: 2.2, duration: 0.4 }}
            />
            <motion.path d="M 203 60 L 203 100 Q 203 118 190 118 L 162 118"
                stroke={BRAND.primary} strokeWidth="1.5" fill="none"
                initial={{ pathLength: 0 }} animate={{ pathLength: 1 }} transition={{ delay: 2.2, duration: 0.4 }}
            />

            <motion.rect x="108" y="108" width="64" height="20" rx="10"
                fill={BRAND.primary}
                initial={{ opacity: 0, scale: 0 }} animate={{ opacity: 1, scale: 1 }}
                transition={{ delay: 2.6, type: "spring", stiffness: 200 }}
            />
            <motion.text x="140" y="122" fontSize="7" fill="white" fontWeight="700" textAnchor="middle"
                initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 2.8 }}>
                {labels.nextReview}
            </motion.text>
        </svg>
    )
}

// --- Slide 2: Interval Calculation ---
function IntervalIllustration({ labels }: { labels: Labels }) {
    const scenarios = [
        { y: 12, label: labels.correctConfident, barWidth: 180, labelRight: labels.longInterval, delay: 0.3, barColor: BRAND.primary },
        { y: 52, label: labels.correctUnsure, barWidth: 70, labelRight: labels.shortInterval, delay: 0.6, barColor: BRAND.blue300 },
        { y: 92, label: labels.wrongAny, barWidth: 25, labelRight: labels.resetHours, delay: 0.9, barColor: BRAND.gray400 },
    ]

    return (
        <svg viewBox="0 0 260 150" className="w-full h-full" fill="none">
            {scenarios.map((s, i) => (
                <motion.g key={i}>
                    <motion.text x="15" y={s.y + 8} fontSize="9" fill={BRAND.gray700} fontWeight="600"
                        initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: s.delay }}>
                        {s.label}
                    </motion.text>
                    <rect x="15" y={s.y + 14} width="200" height="8" rx="4" fill={BRAND.bg50} />
                    <motion.rect x="15" y={s.y + 14} height="8" rx="4"
                        fill={s.barColor}
                        initial={{ width: 0 }} animate={{ width: s.barWidth }}
                        transition={{ delay: s.delay + 0.2, duration: 0.6, ease: "easeOut" }}
                    />
                    <motion.text x="15" y={s.y + 32} fontSize="8" fill={BRAND.gray500}
                        initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: s.delay + 0.6 }}>
                        → {s.labelRight}
                    </motion.text>
                </motion.g>
            ))}

            <motion.text x="130" y="145" fontSize="8" fill={BRAND.gray400} textAnchor="middle"
                initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 1.8 }}>
                {labels.maxInterval}
            </motion.text>
        </svg>
    )
}

// --- Slide 3: CHM (Confident but Wrong) ---
function CHMIllustration({ labels }: { labels: Labels }) {
    return (
        <svg viewBox="0 0 260 150" className="w-full h-full" fill="none">
            <motion.rect x="20" y="15" width="90" height="50" rx="8"
                fill={BRAND.bg50} stroke={BRAND.blue200} strokeWidth="1.5"
                initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 0.2 }}
            />
            <motion.text x="65" y="35" fontSize="9" fill={BRAND.gray500} fontWeight="600" textAnchor="middle"
                initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 0.3 }}>
                {labels.wrong}
            </motion.text>
            <motion.text x="65" y="52" fontSize="11" textAnchor="middle"
                initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 0.5 }}>
                {labels.confident}
            </motion.text>

            <motion.text x="125" y="45" fontSize="18" fill={BRAND.gray400} textAnchor="middle" fontWeight="bold"
                initial={{ opacity: 0, scale: 0 }} animate={{ opacity: 1, scale: 1 }}
                transition={{ delay: 0.7, type: "spring" }}>
                =
            </motion.text>

            <motion.rect x="145" y="18" width="100" height="44" rx="8"
                fill={BRAND.amber100} stroke={BRAND.amber400} strokeWidth="2"
                initial={{ opacity: 0, scale: 0.8 }} animate={{ opacity: 1, scale: 1 }}
                transition={{ delay: 1, type: "spring" }}
            />
            <motion.text x="195" y="36" fontSize="13" textAnchor="middle"
                initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 1.2 }}>
                ⚠️
            </motion.text>
            <motion.text x="195" y="52" fontSize="7" fill={BRAND.gray700} fontWeight="600" textAnchor="middle"
                initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 1.3 }}>
                {labels.priorityAlert}
            </motion.text>

            <motion.path d="M 195 65 L 195 80" stroke={BRAND.amber400} strokeWidth="2"
                initial={{ pathLength: 0 }} animate={{ pathLength: 1 }} transition={{ delay: 1.5, duration: 0.3 }}
            />
            <motion.path d="M 191 77 L 195 83 L 199 77" stroke={BRAND.amber400} strokeWidth="2" fill="none"
                initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 1.7 }}
            />

            <motion.rect x="40" y="88" width="180" height="36" rx="8"
                fill={BRAND.bg50} stroke={BRAND.blue200} strokeWidth="1.5"
                initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 1.8 }}
            />
            <motion.text x="55" y="110" fontSize="7" fill={BRAND.gray500} fontWeight="600"
                initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 1.9 }}>
                {labels.nextSession}
            </motion.text>

            {["⚠️ Q7", "Q3", "Q12", "Q5"].map((label, i) => (
                <motion.g key={i}>
                    <motion.rect x={120 + i * 24} y="97" width="20" height="18" rx="3"
                        fill={i === 0 ? BRAND.amber100 : "white"}
                        stroke={i === 0 ? BRAND.amber400 : BRAND.blue200}
                        strokeWidth={i === 0 ? "1.5" : "1"}
                        initial={{ opacity: 0, y: 5 }} animate={{ opacity: 1, y: 0 }}
                        transition={{ delay: 2 + i * 0.1 }}
                    />
                    <motion.text x={130 + i * 24} y={109} fontSize="6" fill={BRAND.gray700}
                        fontWeight={i === 0 ? "700" : "400"} textAnchor="middle"
                        initial={{ opacity: 0 }} animate={{ opacity: 1 }}
                        transition={{ delay: 2.1 + i * 0.1 }}>
                        {label}
                    </motion.text>
                </motion.g>
            ))}
        </svg>
    )
}

// --- Slide 4: Mastery & Session Limits ---
function MasteryIllustration({ labels }: { labels: Labels }) {
    const criteria = [
        { label: labels.strength, y: 18, delay: 0.3, fillWidth: 102, trackWidth: 120 },
        { label: labels.streak, y: 50, delay: 0.5, fillWidth: 84, trackWidth: 120 },
        { label: labels.confidence, y: 82, delay: 0.7, fillWidth: 66, trackWidth: 120 },
    ]

    return (
        <svg viewBox="0 0 280 150" className="w-full h-full" fill="none">
            <motion.text x="10" y="10" fontSize="8" fill={BRAND.primary} fontWeight="700" letterSpacing="0.5"
                initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 0.1 }}>
                {labels.masteryCriteria}
            </motion.text>

            {criteria.map((c, i) => (
                <motion.g key={i}>
                    <motion.text x="10" y={c.y + 6} fontSize="9" fill={BRAND.gray700} fontWeight="500"
                        initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: c.delay }}>
                        {c.label}
                    </motion.text>
                    <rect x="10" y={c.y + 10} width={c.trackWidth} height="7" rx="3.5" fill={BRAND.bg50} />
                    <motion.rect x="10" y={c.y + 10} height="7" rx="3.5"
                        fill={BRAND.primary}
                        initial={{ width: 0 }} animate={{ width: c.fillWidth }}
                        transition={{ delay: c.delay + 0.2, duration: 0.5 }}
                    />
                </motion.g>
            ))}

            <motion.rect x="10" y="115" width="70" height="22" rx="11"
                fill={BRAND.primary}
                initial={{ opacity: 0, scale: 0 }} animate={{ opacity: 1, scale: 1 }}
                transition={{ delay: 1.5, type: "spring", stiffness: 200 }}
            />
            <motion.text x="45" y="130" fontSize="8" fill="white" fontWeight="700" textAnchor="middle"
                initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 1.7 }}>
                {labels.mastered}
            </motion.text>

            <line x1="148" y1="5" x2="148" y2="145" stroke={BRAND.gray200} strokeWidth="1" strokeDasharray="4 3" />

            <motion.text x="160" y="10" fontSize="8" fill={BRAND.primary} fontWeight="700" letterSpacing="0.5"
                initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 0.2 }}>
                {labels.sessionLimit}
            </motion.text>

            {[0, 1, 2, 3, 4, 5, 6].map((i) => (
                <motion.circle key={i}
                    cx={172 + (i % 4) * 18} cy={32 + Math.floor(i / 4) * 22} r="6"
                    fill={i < 5 ? BRAND.primary : BRAND.bgSubtle}
                    stroke={BRAND.blue200} strokeWidth="1"
                    initial={{ scale: 0 }} animate={{ scale: 1 }}
                    transition={{ delay: 1.8 + i * 0.08, type: "spring" }}
                />
            ))}

            <motion.text x="210" y="105" fontSize="32" fill={BRAND.primary} fontWeight="700" textAnchor="middle"
                initial={{ opacity: 0, scale: 0.5 }} animate={{ opacity: 1, scale: 1 }}
                transition={{ delay: 2.4, type: "spring" }}>
                7
            </motion.text>
            <motion.text x="210" y="120" fontSize="8" fill={BRAND.gray500} fontWeight="500" textAnchor="middle"
                initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 2.5 }}>
                {labels.maxWeak}
            </motion.text>
            <motion.text x="210" y="132" fontSize="8" fill={BRAND.gray500} fontWeight="500" textAnchor="middle"
                initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 2.6 }}>
                {labels.perSession}
            </motion.text>
        </svg>
    )
}

// --- Main Component ---

const slideVariants = {
    enter: (direction: number) => ({ x: direction > 0 ? 200 : -200, opacity: 0 }),
    center: { x: 0, opacity: 1 },
    exit: (direction: number) => ({ x: direction < 0 ? 200 : -200, opacity: 0 }),
}

interface SpacedRepetitionInfoProps {
    open?: boolean
    onOpenChange?: (open: boolean) => void
}

export function SpacedRepetitionInfo({ open: controlledOpen, onOpenChange: controlledOnOpenChange }: SpacedRepetitionInfoProps = {}) {
    const [current, setCurrent] = useState(0)
    const [direction, setDirection] = useState(0)

    // Build labels object
    const labels: Labels = {
        question: 'Question',
        nextReview: 'Next Review',
        correctConfident: '✓ Correct + Confident',
        correctUnsure: '✓ Correct + Unsure',
        wrongAny: '✗ Wrong',
        longInterval: 'Long interval (days)',
        shortInterval: 'Short interval',
        resetHours: 'Reset (hours)',
        maxInterval: 'Max interval: 180 days',
        wrong: '✗ Wrong',
        confident: '+ 😊 Confident',
        priorityAlert: 'PRIORITY ALERT',
        nextSession: 'Next session:',
        masteryCriteria: 'MASTERY CRITERIA',
        strength: 'Strength ≥ 90%',
        streak: 'Streak ≥ 3',
        confidence: 'Confidence ≥ 4',
        mastered: 'MASTERED',
        sessionLimit: 'SESSION LIMIT',
        maxWeak: 'max weak',
        perSession: 'per session',
    }

    // Slide content
    const slides = [
        {
            key: "answer-confidence",
            Illustration: AnswerConfidenceIllustration,
            title: "Answer + Confidence",
            text: "After each answer, rate your confidence. This helps the algorithm understand what you truly know vs. what you guessed.",
        },
        {
            key: "intervals",
            Illustration: IntervalIllustration,
            title: "Smart Intervals",
            text: "Correct + high confidence = longer interval. Wrong = reset. The algorithm optimizes when you see each question again.",
        },
        {
            key: "chm",
            Illustration: CHMIllustration,
            title: "Confident Misinformation",
            text: "Being wrong with high confidence is dangerous for exams. These questions are flagged and prioritized in your next session.",
        },
        {
            key: "mastery",
            Illustration: MasteryIllustration,
            title: "Achieving Mastery",
            text: "A question is mastered when: strength ≥90%, 3+ correct streak, and confidence ≥4. Sessions limit weak questions to prevent overwhelm.",
        },
    ]

    const slide = slides[current]

    const goTo = (index: number) => {
        setDirection(index > current ? 1 : -1)
        setCurrent(index)
    }

    const next = () => { if (current < slides.length - 1) goTo(current + 1) }
    const prev = () => { if (current > 0) goTo(current - 1) }

    const handleOpenChange = (open: boolean) => {
        if (open) { setCurrent(0); setDirection(0) }
        controlledOnOpenChange?.(open)
    }

    return (
        <Dialog open={controlledOpen} onOpenChange={handleOpenChange}>
            <DialogTrigger asChild>
                <Button variant="ghost" size="sm" className="gap-2 text-muted-foreground hover:text-blue-600">
                    <Info className="h-4 w-4" />
                    How it works
                </Button>
            </DialogTrigger>
            <DialogContent className="max-w-md p-0 overflow-hidden">
                <DialogHeader className="px-6 pt-6 pb-2">
                    <DialogTitle className="text-xl flex items-center gap-2 text-blue-600">
                        <Brain className="h-5 w-5" />
                        Spaced Repetition
                    </DialogTitle>
                    <p className="text-sm text-muted-foreground">
                        Learn smarter, not harder.
                    </p>
                </DialogHeader>

                <div className="px-6">
                    <div className="relative h-[160px] bg-gray-50 rounded-xl overflow-hidden border border-gray-100">
                        <AnimatePresence mode="wait" custom={direction}>
                            <motion.div
                                key={slide.key}
                                custom={direction}
                                variants={slideVariants}
                                initial="enter"
                                animate="center"
                                exit="exit"
                                transition={{ duration: 0.35, ease: "easeInOut" }}
                                className="absolute inset-0 p-2"
                            >
                                <slide.Illustration labels={labels} />
                            </motion.div>
                        </AnimatePresence>
                    </div>

                    <AnimatePresence mode="wait">
                        <motion.div
                            key={slide.key + "-text"}
                            initial={{ opacity: 0, y: 8 }}
                            animate={{ opacity: 1, y: 0 }}
                            exit={{ opacity: 0, y: -8 }}
                            transition={{ duration: 0.25 }}
                            className="mt-4 min-h-[80px]"
                        >
                            <h3 className="font-semibold text-gray-900 text-base">
                                {slide.title}
                            </h3>
                            <p className="text-sm text-gray-600 leading-relaxed mt-1.5">
                                {slide.text}
                            </p>
                        </motion.div>
                    </AnimatePresence>
                </div>

                <div className="flex items-center justify-between px-6 py-4 border-t bg-gray-50/50 mt-2">
                    <Button variant="ghost" size="sm" onClick={prev} disabled={current === 0} className="text-gray-500">
                        <ChevronLeft className="h-4 w-4 mr-1" />
                        Prev
                    </Button>

                    <div className="flex items-center gap-2">
                        {slides.map((_, idx) => (
                            <button
                                key={idx}
                                onClick={() => goTo(idx)}
                                className={`rounded-full transition-all duration-300 ${
                                    idx === current
                                        ? 'w-6 h-2 bg-blue-600'
                                        : 'w-2 h-2 bg-gray-300 hover:bg-gray-400'
                                }`}
                            />
                        ))}
                    </div>

                    {current < slides.length - 1 ? (
                        <Button size="sm" onClick={next} className="bg-blue-600 hover:bg-blue-700">
                            Next
                            <ChevronRight className="h-4 w-4 ml-1" />
                        </Button>
                    ) : (
                        <DialogTrigger asChild>
                            <Button size="sm" className="bg-blue-600 hover:bg-blue-700">
                                Got it!
                            </Button>
                        </DialogTrigger>
                    )}
                </div>
            </DialogContent>
        </Dialog>
    )
}
