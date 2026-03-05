"use client"

import { useEffect, useState } from "react"
import { useParams } from "next/navigation"
import { CodeEditorAdvanced } from "@/components/code-editor-advanced"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { ChevronLeft, ChevronRight } from "lucide-react"
import Link from "next/link"

const MOCK_LESSONS: Record<string, any[]> = {
  "python-101": [
    {
      id: "lesson-1",
      title: "Introduction to Python",
      description: "Learn what Python is and why it's popular",
      content:
        "Python is a high-level, interpreted programming language known for its simplicity and readability. It's widely used in web development, data science, and automation.",
      code_example: `# Your first Python program
print("Hello, World!")
print("Welcome to Python!")`,
      order_index: 1,
    },
    {
      id: "lesson-2",
      title: "Variables and Data Types",
      description: "Understand variables, strings, numbers, and lists",
      content:
        "Variables are containers for storing data values. Python has various data types including strings, integers, floats, and lists.",
      code_example: `# Variables and data types
name = "Alice"
age = 25
height = 5.6
hobbies = ["reading", "coding", "gaming"]

print(f"Name: {name}")
print(f"Age: {age}")
print(f"Hobbies: {hobbies}")`,
      order_index: 2,
    },
    {
      id: "lesson-3",
      title: "Control Flow",
      description: "Learn if statements, loops, and conditionals",
      content: "Control flow statements allow you to make decisions and repeat code blocks based on conditions.",
      code_example: `# If statements and loops
for i in range(1, 6):
    if i % 2 == 0:
        print(f"{i} is even")
    else:
        print(f"{i} is odd")`,
      order_index: 3,
    },
  ],
  "javascript-101": [
    {
      id: "lesson-1",
      title: "JavaScript Basics",
      description: "Introduction to JavaScript",
      content:
        "JavaScript is a versatile programming language that runs in browsers and servers. It's essential for web development.",
      code_example: `// Your first JavaScript program
console.log("Hello, World!");
console.log("Welcome to JavaScript!");`,
      order_index: 1,
    },
    {
      id: "lesson-2",
      title: "Variables and Functions",
      description: "Learn about variables and functions",
      content: "Variables store data, and functions are reusable blocks of code that perform specific tasks.",
      code_example: `// Variables and functions
const name = "Bob";
const age = 30;

function greet(person) {
    console.log(\`Hello, \${person}!\`);
}

greet(name);`,
      order_index: 2,
    },
  ],
  "cpp-101": [
    {
      id: "lesson-1",
      title: "C++ Fundamentals",
      description: "Introduction to C++",
      content:
        "C++ is a powerful, compiled language used for system software, game development, and performance-critical applications.",
      code_example: `#include <iostream>
using namespace std;

int main() {
    cout << "Hello, World!" << endl;
    return 0;
}`,
      order_index: 1,
    },
  ],
  "java-101": [
    {
      id: "lesson-1",
      title: "Java Basics",
      description: "Introduction to Java",
      content: "Java is an object-oriented language known for its 'write once, run anywhere' philosophy.",
      code_example: `public class HelloWorld {
    public static void main(String[] args) {
        System.out.println("Hello, World!");
    }
}`,
      order_index: 1,
    },
  ],
  "swift-101": [
    {
      id: "lesson-1",
      title: "Swift Essentials",
      description: "Introduction to Swift",
      content: "Swift is a modern programming language for iOS, macOS, and other Apple platforms.",
      code_example: `import Foundation

print("Hello, World!")
print("Welcome to Swift!")`,
      order_index: 1,
    },
  ],
  "ml-101": [
    {
      id: "lesson-1",
      title: "ML Fundamentals",
      description: "Introduction to Machine Learning",
      content: "Machine Learning is a subset of AI that enables systems to learn and improve from experience.",
      code_example: `# Machine Learning Example
import numpy as np

# Sample data
X = np.array([1, 2, 3, 4, 5])
y = np.array([2, 4, 6, 8, 10])

print("Data loaded successfully!")`,
      order_index: 1,
    },
  ],
}

const MOCK_COURSES: Record<string, any> = {
  "python-101": {
    id: "python-101",
    title: "Python Basics",
    description: "Learn Python programming",
    language: "python",
  },
  "javascript-101": {
    id: "javascript-101",
    title: "JavaScript Essentials",
    description: "Master JavaScript",
    language: "javascript",
  },
  "cpp-101": { id: "cpp-101", title: "C++ Programming", description: "Learn C++", language: "cpp" },
  "java-101": { id: "java-101", title: "Java Fundamentals", description: "Learn Java", language: "java" },
  "swift-101": { id: "swift-101", title: "Swift for iOS", description: "Learn Swift", language: "swift" },
  "ml-101": { id: "ml-101", title: "Machine Learning Basics", description: "Learn ML", language: "python" },
}

export default function LearnPage() {
  const params = useParams()
  const courseId = params.courseId as string
  const [course, setCourse] = useState<any>(null)
  const [lessons, setLessons] = useState<any[]>([])
  const [currentLessonIndex, setCurrentLessonIndex] = useState(0)
  const [isLoading, setIsLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    try {
      const mockCourse = MOCK_COURSES[courseId]
      const mockLessons = MOCK_LESSONS[courseId] || []

      if (!mockCourse || mockLessons.length === 0) {
        setError("Course not found")
      } else {
        setCourse(mockCourse)
        setLessons(mockLessons)
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to load course")
    } finally {
      setIsLoading(false)
    }
  }, [courseId])

  if (isLoading) {
    return (
      <div className="flex items-center justify-center min-h-screen">
        <p className="text-muted-foreground">Loading course...</p>
      </div>
    )
  }

  if (error || !course || lessons.length === 0) {
    return (
      <div className="flex flex-col items-center justify-center min-h-screen gap-4">
        <p className="text-red-500">{error || "Course not found"}</p>
        <Link href="/dashboard">
          <Button>Back to Dashboard</Button>
        </Link>
      </div>
    )
  }

  const currentLesson = lessons[currentLessonIndex]

  return (
    <div className="min-h-screen bg-background">
      <div className="container mx-auto px-4 py-8">
        <div className="mb-8">
          <Link href="/dashboard" className="text-primary hover:underline text-sm mb-4 inline-block">
            ← Back to Dashboard
          </Link>
          <h1 className="text-3xl font-bold mb-2">{course.title}</h1>
          <p className="text-muted-foreground">{course.description}</p>
        </div>

        <div className="grid gap-8 md:grid-cols-3">
          <div className="md:col-span-2 space-y-6">
            <Card className="border-border/50">
              <CardHeader>
                <CardTitle>{currentLesson.title}</CardTitle>
                <CardDescription>
                  Lesson {currentLessonIndex + 1} of {lessons.length}
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="prose prose-invert max-w-none">
                  <p className="text-foreground">{currentLesson.description}</p>
                  <div className="mt-4 p-4 bg-muted rounded-lg">
                    <p className="text-sm whitespace-pre-wrap">{currentLesson.content}</p>
                  </div>
                </div>
              </CardContent>
            </Card>

            <CodeEditorAdvanced
              initialLanguage={course.language}
              initialCode={currentLesson.code_example}
              onCodeChange={(code) => {
                // Save code submission
              }}
            />
          </div>

          <div className="space-y-4">
            <Card className="border-border/50">
              <CardHeader>
                <CardTitle className="text-lg">Lessons</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-2 max-h-96 overflow-y-auto">
                  {lessons.map((lesson, index) => (
                    <button
                      key={lesson.id}
                      onClick={() => setCurrentLessonIndex(index)}
                      className={`w-full text-left p-3 rounded-lg transition-colors ${
                        index === currentLessonIndex
                          ? "bg-primary text-primary-foreground"
                          : "bg-muted hover:bg-muted/80"
                      }`}
                    >
                      <p className="text-sm font-medium">Lesson {index + 1}</p>
                      <p className="text-xs opacity-75">{lesson.title}</p>
                    </button>
                  ))}
                </div>
              </CardContent>
            </Card>

            <div className="flex gap-2">
              <Button
                variant="outline"
                size="sm"
                onClick={() => setCurrentLessonIndex(Math.max(0, currentLessonIndex - 1))}
                disabled={currentLessonIndex === 0}
                className="flex-1"
              >
                <ChevronLeft className="h-4 w-4" />
              </Button>
              <Button
                variant="outline"
                size="sm"
                onClick={() => setCurrentLessonIndex(Math.min(lessons.length - 1, currentLessonIndex + 1))}
                disabled={currentLessonIndex === lessons.length - 1}
                className="flex-1"
              >
                <ChevronRight className="h-4 w-4" />
              </Button>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
