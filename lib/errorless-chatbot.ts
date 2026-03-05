// Errorless Chatbot - Combines NLP and LLM
import { AdvancedNLP } from "./nlp/advanced-nlp"
import { CustomLLMModel } from "./llm/custom-llm"

export interface ChatMessage {
  id: string
  role: "user" | "assistant"
  content: string
  timestamp: Date
  nlpAnalysis?: any
  codeExample?: string
}

export class ErrorlessChatbot {
  private nlp: AdvancedNLP
  private llm: CustomLLMModel
  private conversationHistory: ChatMessage[] = []

  constructor() {
    this.nlp = new AdvancedNLP()
    this.llm = new CustomLLMModel()
  }

  async processMessage(userMessage: string): Promise<ChatMessage> {
    const nlpResult = this.nlp.analyze(userMessage)
    const response = this.llm.generateResponse(nlpResult.intent, nlpResult.detectedLanguage, userMessage)

    const assistantMessage: ChatMessage = {
      id: Date.now().toString(),
      role: "assistant",
      content: response.response,
      timestamp: new Date(),
      nlpAnalysis: nlpResult,
      codeExample: response.codeExample,
    }

    this.conversationHistory.push({
      id: Date.now().toString(),
      role: "user",
      content: userMessage,
      timestamp: new Date(),
    })
    this.conversationHistory.push(assistantMessage)

    return assistantMessage
  }

  getConversationHistory(): ChatMessage[] {
    return this.conversationHistory
  }

  clearHistory(): void {
    this.conversationHistory = []
  }
}
