"use client"

import { useUser, useClerk } from "@clerk/nextjs"
import { useRouter } from "next/navigation"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Progress } from "@/components/ui/progress"
import { LogOut, BookOpen, Trophy, Code } from "lucide-react"
import Link from "next/link"

const MOCK_COURSES = [
  {
    id: "python-101",
    title: "Python Basics",
    description: "Learn the fundamentals of Python programming",
    language: "python",
    icon: "🐍",
    color: "#3776ab",
    difficulty_level: "Beginner",
  },
  {
    id: "javascript-101",
    title: "JavaScript Essentials",
    description: "Master JavaScript from basics to advanced concepts",
    language: "javascript",
    icon: "⚡",
    color: "#f7df1e",
    difficulty_level: "Beginner",
  },
  {
    id: "cpp-101",
    title: "C++ Programming",
    description: "Deep dive into C++ and object-oriented programming",
    language: "cpp",
    icon: "⚙️",
    color: "#00599c",
    difficulty_level: "Intermediate",
  },
  {
    id: "java-101",
    title: "Java Fundamentals",
    description: "Learn Java programming and OOP concepts",
    language: "java",
    icon: "☕",
    color: "#007396",
    difficulty_level: "Intermediate",
  },
  {
    id: "swift-101",
    title: "Swift for iOS",
    description: "Build iOS apps with Swift",
    language: "swift",
    icon: "🍎",
    color: "#fa7343",
    difficulty_level: "Intermediate",
  },
  {
    id: "ml-101",
    title: "Machine Learning Basics",
    description: "Introduction to ML and AI concepts",
    language: "python",
    icon: "🤖",
    color: "#ff6b6b",
    difficulty_level: "Advanced",
  },
]

const MOCK_PROGRESS: Record<string, any> = {
  "python-101": { progress_percentage: 45, completed_lessons: 9, total_lessons: 20 },
  "javascript-101": { progress_percentage: 60, completed_lessons: 12, total_lessons: 20 },
  "cpp-101": { progress_percentage: 30, completed_lessons: 6, total_lessons: 20 },
}

interface UserProfile {
  full_name: string
  email: string
  avatar_url: string | null
}

export default function DashboardPage() {
  const router = useRouter()
  const { user, isLoaded } = useUser()
  const { signOut } = useClerk()
  const stats = { coursesEnrolled: Object.keys(MOCK_PROGRESS).length, quizzesPassed: 5, hoursLearned: 15 }

  if (!isLoaded) {
    return (
      <div className="flex items-center justify-center min-h-screen">
        <p className="text-muted-foreground">Loading dashboard...</p>
      </div>
    )
  }

  const handleLogout = () => {
    signOut({ redirectUrl: "/" })
  }

  return (
    <div className="min-h-screen bg-background">
      {/* Header */}
      <header className="border-b border-border/50 bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60">
        <div className="container mx-auto px-4 py-4 flex items-center justify-between">
          <div>
            <h1 className="text-2xl font-bold">Errorless</h1>
            <p className="text-sm text-muted-foreground">Learn to Code, Master Programming</p>
          </div>
          <div className="flex items-center gap-4">
            <div className="text-right hidden md:block">
              <p className="font-medium">{user?.fullName || user?.firstName || "User"}</p>
              <p className="text-sm text-muted-foreground">{user?.primaryEmailAddress?.emailAddress}</p>
            </div>
            <Button variant="outline" size="sm" onClick={handleLogout} className="bg-transparent">
              <LogOut className="h-4 w-4 mr-2" />
              Logout
            </Button>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="container mx-auto px-4 py-8">
        {/* Stats Section */}
        <div className="grid gap-4 md:grid-cols-3 mb-8">
          <Card className="border-border/50">
            <CardHeader className="pb-3">
              <CardTitle className="text-sm font-medium text-muted-foreground">Courses Enrolled</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-3xl font-bold">{stats.coursesEnrolled}</div>
              <p className="text-xs text-muted-foreground mt-1">Active learning paths</p>
            </CardContent>
          </Card>

          <Card className="border-border/50">
            <CardHeader className="pb-3">
              <CardTitle className="text-sm font-medium text-muted-foreground flex items-center gap-2">
                <Trophy className="h-4 w-4" />
                Quizzes Passed
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-3xl font-bold">{stats.quizzesPassed}</div>
              <p className="text-xs text-muted-foreground mt-1">Assessments completed</p>
            </CardContent>
          </Card>

          <Card className="border-border/50">
            <CardHeader className="pb-3">
              <CardTitle className="text-sm font-medium text-muted-foreground">Hours Learned</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-3xl font-bold">{stats.hoursLearned}</div>
              <p className="text-xs text-muted-foreground mt-1">Total learning time</p>
            </CardContent>
          </Card>
        </div>

        {/* Courses Section */}
        <div>
          <div className="mb-6">
            <h2 className="text-2xl font-bold mb-2">Available Courses</h2>
            <p className="text-muted-foreground">Choose a course and start learning today</p>
          </div>

          <div className="grid gap-6 md:grid-cols-2 lg:grid-cols-3">
            {MOCK_COURSES.map((course) => {
              const courseProgress = MOCK_PROGRESS[course.id]
              const progressPercentage = courseProgress?.progress_percentage || 0

              return (
                <Card
                  key={course.id}
                  className="border-border/50 hover:border-border transition-colors overflow-hidden"
                >
                  <CardHeader className="pb-3">
                    <div className="flex items-start justify-between mb-2">
                      <div className="text-3xl">{course.icon}</div>
                      <span
                        className="px-2 py-1 rounded text-xs font-medium"
                        style={{
                          backgroundColor: `${course.color}20`,
                          color: course.color,
                        }}
                      >
                        {course.difficulty_level}
                      </span>
                    </div>
                    <CardTitle className="text-lg">{course.title}</CardTitle>
                    <CardDescription className="line-clamp-2">{course.description}</CardDescription>
                  </CardHeader>
                  <CardContent className="space-y-4">
                    {courseProgress && (
                      <div className="space-y-2">
                        <div className="flex justify-between text-sm">
                          <span className="text-muted-foreground">Progress</span>
                          <span className="font-medium">{progressPercentage}%</span>
                        </div>
                        <Progress value={progressPercentage} className="h-2" />
                        <p className="text-xs text-muted-foreground">
                          {courseProgress.completed_lessons} of {courseProgress.total_lessons} lessons
                        </p>
                      </div>
                    )}

                    <div className="flex gap-2 pt-2">
                      <Link href={`/learn/${course.id}`} className="flex-1">
                        <Button variant="outline" size="sm" className="w-full bg-transparent">
                          <BookOpen className="h-4 w-4 mr-2" />
                          Learn
                        </Button>
                      </Link>
                      <Link href={`/quiz/${course.id}`} className="flex-1">
                        <Button variant="outline" size="sm" className="w-full bg-transparent">
                          <Code className="h-4 w-4 mr-2" />
                          Quiz
                        </Button>
                      </Link>
                    </div>
                  </CardContent>
                </Card>
              )
            })}
          </div>
        </div>
      </main>
    </div>
  )
}
