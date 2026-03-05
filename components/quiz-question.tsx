"use client"
import { useState } from "react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { RadioGroup, RadioGroupItem } from "@/components/ui/radio-group"
import { Label } from "@/components/ui/label"
import { Checkbox } from "@/components/ui/checkbox"

interface QuizOption {
  id: string
  text: string
  isCorrect: boolean
}

interface QuizQuestionProps {
  questionNumber: number
  totalQuestions: number
  questionText: string
  questionType: "multiple_choice" | "true_false" | "code_snippet"
  options: QuizOption[]
  onAnswer: (selectedOptionIds: string[]) => void
  onNext: () => void
  onPrevious: () => void
  isFirst: boolean
  isLast: boolean
  isAnswered: boolean
}

export function QuizQuestion({
  questionNumber,
  totalQuestions,
  questionText,
  questionType,
  options,
  onAnswer,
  onNext,
  onPrevious,
  isFirst,
  isLast,
  isAnswered,
}: QuizQuestionProps) {
  const [selectedOptions, setSelectedOptions] = useState<string[]>([])

  const handleSelectOption = (optionId: string) => {
    if (questionType === "multiple_choice" || questionType === "true_false") {
      setSelectedOptions([optionId])
    } else {
      setSelectedOptions((prev) =>
        prev.includes(optionId) ? prev.filter((id) => id !== optionId) : [...prev, optionId],
      )
    }
  }

  const handleSubmitAnswer = () => {
    onAnswer(selectedOptions)
    onNext()
  }

  return (
    <Card className="border-border/50">
      <CardHeader>
        <CardTitle>
          Question {questionNumber} of {totalQuestions}
        </CardTitle>
        <CardDescription>{questionType === "code_snippet" ? "Code Analysis" : "Multiple Choice"}</CardDescription>
      </CardHeader>
      <CardContent className="space-y-6">
        <div className="space-y-4">
          <p className="text-lg font-medium">{questionText}</p>

          {questionType === "code_snippet" && (
            <div className="bg-muted p-4 rounded-lg font-mono text-sm overflow-x-auto">
              <pre>{`function fibonacci(n) {
  if (n <= 1) return n;
  return fibonacci(n - 1) + fibonacci(n - 2);
}`}</pre>
            </div>
          )}

          <div className="space-y-3">
            {questionType === "multiple_choice" || questionType === "true_false" ? (
              <RadioGroup value={selectedOptions[0] || ""} onValueChange={(value) => handleSelectOption(value)}>
                {options.map((option) => (
                  <div key={option.id} className="flex items-center space-x-2 p-3 rounded-lg hover:bg-muted/50">
                    <RadioGroupItem value={option.id} id={option.id} />
                    <Label htmlFor={option.id} className="cursor-pointer flex-1">
                      {option.text}
                    </Label>
                  </div>
                ))}
              </RadioGroup>
            ) : (
              <div className="space-y-3">
                {options.map((option) => (
                  <div key={option.id} className="flex items-center space-x-2 p-3 rounded-lg hover:bg-muted/50">
                    <Checkbox
                      id={option.id}
                      checked={selectedOptions.includes(option.id)}
                      onCheckedChange={() => handleSelectOption(option.id)}
                    />
                    <Label htmlFor={option.id} className="cursor-pointer flex-1">
                      {option.text}
                    </Label>
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>

        <div className="flex flex-col gap-2 md:flex-row md:justify-between">
          <Button variant="outline" onClick={onPrevious} disabled={isFirst} className="bg-transparent md:order-1">
            Previous
          </Button>
          <Button onClick={handleSubmitAnswer} disabled={selectedOptions.length === 0} className="md:order-2">
            {isLast ? "Submit Quiz" : "Next Question"}
          </Button>
        </div>
      </CardContent>
    </Card>
  )
}
