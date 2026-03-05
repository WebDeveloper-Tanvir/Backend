"use client"

import { useEffect, useState } from "react"
import { useParams, useRouter } from "next/navigation"
import { createClient } from "@/lib/supabase/client"
import { QuizContainer } from "@/components/quiz-container"
import { QuizQuestion } from "@/components/quiz-question"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Progress } from "@/components/ui/progress"
import Link from "next/link"

const MOCK_QUIZZES: Record<string, any> = {
  "python-101": {
    id: "python-101",
    title: "Python Basics Quiz",
    description: "Test your Python knowledge",
    time_limit_minutes: 15,
    passing_score: 70,
  },
  "javascript-101": {
    id: "javascript-101",
    title: "JavaScript Essentials Quiz",
    description: "Test your JavaScript knowledge",
    time_limit_minutes: 15,
    passing_score: 70,
  },
  "cpp-101": {
    id: "cpp-101",
    title: "C++ Programming Quiz",
    description: "Test your C++ knowledge",
    time_limit_minutes: 20,
    passing_score: 70,
  },
  "java-101": {
    id: "java-101",
    title: "Java Fundamentals Quiz",
    description: "Test your Java knowledge",
    time_limit_minutes: 15,
    passing_score: 70,
  },
  "swift-101": {
    id: "swift-101",
    title: "Swift for iOS Quiz",
    description: "Test your Swift knowledge",
    time_limit_minutes: 15,
    passing_score: 70,
  },
  "ml-101": {
    id: "ml-101",
    title: "Machine Learning Basics Quiz",
    description: "Test your ML knowledge",
    time_limit_minutes: 20,
    passing_score: 70,
  },
}

const MOCK_QUESTIONS: Record<string, any[]> = {
  "python-101": [
    {
      id: "q1",
      question_text: "What is the correct way to create a list in Python?",
      question_type: "multiple_choice",
      order_index: 1,
      options: [
        { id: "opt1", option_text: "list = [1, 2, 3]", is_correct: true },
        { id: "opt2", option_text: "list = (1, 2, 3)", is_correct: false },
        { id: "opt3", option_text: "list = {1, 2, 3}", is_correct: false },
        { id: "opt4", option_text: "list = <1, 2, 3>", is_correct: false },
      ],
    },
    {
      id: "q2",
      question_text: "Python is a compiled language.",
      question_type: "true_false",
      order_index: 2,
      options: [
        { id: "opt5", option_text: "True", is_correct: false },
        { id: "opt6", option_text: "False", is_correct: true },
      ],
    },
    {
      id: "q3",
      question_text: "What does the print() function do?",
      question_type: "multiple_choice",
      order_index: 3,
      options: [
        { id: "opt7", option_text: "Displays output to the console", is_correct: true },
        { id: "opt8", option_text: "Prints to a physical printer", is_correct: false },
        { id: "opt9", option_text: "Creates a new variable", is_correct: false },
        { id: "opt10", option_text: "Imports a module", is_correct: false },
      ],
    },
  ],
  "javascript-101": [
    {
      id: "q1",
      question_text: "What is the correct way to declare a variable in JavaScript?",
      question_type: "multiple_choice",
      order_index: 1,
      options: [
        { id: "opt1", option_text: "const name = 'John';", is_correct: true },
        { id: "opt2", option_text: "variable name = 'John';", is_correct: false },
        { id: "opt3", option_text: "name := 'John';", is_correct: false },
        { id: "opt4", option_text: "declare name = 'John';", is_correct: false },
      ],
    },
    {
      id: "q2",
      question_text: "JavaScript is a statically typed language.",
      question_type: "true_false",
      order_index: 2,
      options: [
        { id: "opt5", option_text: "True", is_correct: false },
        { id: "opt6", option_text: "False", is_correct: true },
      ],
    },
  ],
  "cpp-101": [
    {
      id: "q1",
      question_text: "What is the correct syntax for a C++ function?",
      question_type: "multiple_choice",
      order_index: 1,
      options: [
        { id: "opt1", option_text: "void functionName() { }", is_correct: true },
        { id: "opt2", option_text: "function functionName() { }", is_correct: false },
        { id: "opt3", option_text: "def functionName(): pass", is_correct: false },
        { id: "opt4", option_text: "func functionName() { }", is_correct: false },
      ],
    },
  ],
  "java-101": [
    {
      id: "q1",
      question_text: "Java is platform-independent.",
      question_type: "true_false",
      order_index: 1,
      options: [
        { id: "opt1", option_text: "True", is_correct: true },
        { id: "opt2", option_text: "False", is_correct: false },
      ],
    },
  ],
  "swift-101": [
    {
      id: "q1",
      question_text: "Swift is used for iOS development.",
      question_type: "true_false",
      order_index: 1,
      options: [
        { id: "opt1", option_text: "True", is_correct: true },
        { id: "opt2", option_text: "False", is_correct: false },
      ],
    },
  ],
  "ml-101": [
    {
      id: "q1",
      question_text: "What is supervised learning?",
      question_type: "multiple_choice",
      order_index: 1,
      options: [
        { id: "opt1", option_text: "Learning with labeled data", is_correct: true },
        { id: "opt2", option_text: "Learning without labels", is_correct: false },
        { id: "opt3", option_text: "Learning from a teacher", is_correct: false },
        { id: "opt4", option_text: "Learning in a classroom", is_correct: false },
      ],
    },
  ],
}

export default function QuizPage() {
  const params = useParams()
  const router = useRouter()
  const quizId = params.quizId as string

  const [quiz, setQuiz] = useState<any>(null)
  const [questions, setQuestions] = useState<any[]>([])
  const [currentQuestionIndex, setCurrentQuestionIndex] = useState(0)
  const [answers, setAnswers] = useState<Record<string, string[]>>({})
  const [isLoading, setIsLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [quizSubmitted, setQuizSubmitted] = useState(false)
  const [score, setScore] = useState(0)

  useEffect(() => {
    try {
      const mockQuiz = MOCK_QUIZZES[quizId]
      const mockQuestions = MOCK_QUESTIONS[quizId] || []

      if (!mockQuiz || mockQuestions.length === 0) {
        setError("Quiz not found")
      } else {
        setQuiz(mockQuiz)
        setQuestions(mockQuestions)
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to load quiz")
    } finally {
      setIsLoading(false)
    }
  }, [quizId])

  const handleAnswer = (selectedOptionIds: string[]) => {
    const currentQuestion = questions[currentQuestionIndex]
    setAnswers((prev) => ({
      ...prev,
      [currentQuestion.id]: selectedOptionIds,
    }))
  }

  const handleNext = () => {
    if (currentQuestionIndex < questions.length - 1) {
      setCurrentQuestionIndex(currentQuestionIndex + 1)
    }
  }

  const handlePrevious = () => {
    if (currentQuestionIndex > 0) {
      setCurrentQuestionIndex(currentQuestionIndex - 1)
    }
  }

  const handleSubmitQuiz = async () => {
    // Calculate score
    let correctCount = 0
    questions.forEach((question) => {
      const selectedOptions = answers[question.id] || []
      const correctOptions = question.options.filter((opt: any) => opt.is_correct).map((opt: any) => opt.id)

      if (
        selectedOptions.length === correctOptions.length &&
        selectedOptions.every((id) => correctOptions.includes(id))
      ) {
        correctCount++
      }
    })

    const percentage = Math.round((correctCount / questions.length) * 100)
    setScore(percentage)
    setQuizSubmitted(true)

    // Try to save quiz attempt to database
    try {
      const supabase = createClient()
      const { data: userData } = await supabase.auth.getUser()

      if (userData.user) {
        await supabase.from("quiz_attempts").insert({
          user_id: userData.user.id,
          quiz_id: quizId,
          score: correctCount,
          total_questions: questions.length,
          percentage_score: percentage,
          passed: percentage >= (quiz?.passing_score || 70),
          time_taken_seconds: 0,
        })
      }
    } catch (err) {
      console.error("[v0] Failed to save quiz attempt:", err)
    }
  }

  const handleTimeUp = () => {
    handleSubmitQuiz()
  }

  if (isLoading) {
    return (
      <div className="flex items-center justify-center min-h-screen">
        <p className="text-muted-foreground">Loading quiz...</p>
      </div>
    )
  }

  if (error || !quiz || questions.length === 0) {
    return (
      <div className="flex flex-col items-center justify-center min-h-screen gap-4">
        <p className="text-red-500">{error || "Quiz not found"}</p>
        <Link href="/dashboard">
          <Button>Back to Dashboard</Button>
        </Link>
      </div>
    )
  }

  if (quizSubmitted) {
    const passed = score >= (quiz.passing_score || 70)
    return (
      <div className="min-h-screen bg-background flex items-center justify-center p-4">
        <Card className="w-full max-w-md border-border/50">
          <CardHeader className="text-center">
            <div className="text-6xl mb-4">{passed ? "🎉" : "📚"}</div>
            <CardTitle className="text-2xl">{passed ? "Quiz Passed!" : "Quiz Completed"}</CardTitle>
            <CardDescription>Your score: {score}%</CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="space-y-2">
              <p className="text-sm text-muted-foreground">
                You answered {Math.round((score / 100) * questions.length)} out of {questions.length} questions
                correctly.
              </p>
              <Progress value={score} className="h-2" />
            </div>
            <div className="flex gap-2">
              <Button variant="outline" onClick={() => router.push("/dashboard")} className="flex-1 bg-transparent">
                Dashboard
              </Button>
              <Button onClick={() => window.location.reload()} className="flex-1">
                Retake Quiz
              </Button>
            </div>
          </CardContent>
        </Card>
      </div>
    )
  }

  const currentQuestion = questions[currentQuestionIndex]
  const progress = ((currentQuestionIndex + 1) / questions.length) * 100

  return (
    <div className="min-h-screen bg-background p-4 md:p-8">
      <div className="container mx-auto max-w-4xl">
        <QuizContainer quizTitle={quiz.title} timeLimit={quiz.time_limit_minutes} onTimeUp={handleTimeUp}>
          <div className="p-6 space-y-6">
            <Progress value={progress} className="h-2" />

            <QuizQuestion
              questionNumber={currentQuestionIndex + 1}
              totalQuestions={questions.length}
              questionText={currentQuestion.question_text}
              questionType={currentQuestion.question_type}
              options={currentQuestion.options.map((opt: any) => ({
                id: opt.id,
                text: opt.option_text,
                isCorrect: opt.is_correct,
              }))}
              onAnswer={handleAnswer}
              onNext={handleNext}
              onPrevious={handlePrevious}
              isFirst={currentQuestionIndex === 0}
              isLast={currentQuestionIndex === questions.length - 1}
              isAnswered={!!answers[currentQuestion.id]}
            />

            {currentQuestionIndex === questions.length - 1 && (
              <Button onClick={handleSubmitQuiz} className="w-full" size="lg">
                Submit Quiz
              </Button>
            )}
          </div>
        </QuizContainer>
      </div>
    </div>
  )
}
