// Advanced NLP Engine for Errorless Chatbot
// Handles tokenization, keyword extraction, intent detection, language detection, entity extraction, sentiment analysis

export interface NLPResult {
  tokens: string[]
  keywords: { word: string; frequency: number }[]
  intent: string
  detectedLanguage: string
  entities: { type: string; value: string }[]
  sentiment: "positive" | "negative" | "neutral"
  complexity: "simple" | "moderate" | "complex"
}

export class AdvancedNLP {
  private stopWords = new Set([
    "the",
    "a",
    "an",
    "and",
    "or",
    "but",
    "in",
    "on",
    "at",
    "to",
    "for",
    "of",
    "with",
    "by",
    "from",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "being",
    "have",
    "has",
    "had",
    "do",
    "does",
    "did",
    "will",
    "would",
    "could",
    "should",
    "may",
    "might",
    "must",
    "can",
    "this",
    "that",
    "these",
    "those",
    "i",
    "you",
    "he",
    "she",
    "it",
    "we",
    "they",
    "what",
    "which",
    "who",
    "when",
    "where",
    "why",
    "how",
  ])

  private intentPatterns = {
    "code-generation": /generate|create|write|make|build|code|function|class|method|script/i,
    debugging: /debug|error|fix|bug|issue|problem|wrong|broken|crash|fail/i,
    explanation: /explain|what|how|why|understand|learn|teach|describe|tell|show/i,
    optimization: /optimize|improve|faster|performance|efficient|speed|reduce|minimize/i,
    "best-practices": /best|practice|standard|convention|pattern|approach|way|should|recommend/i,
    comparison: /compare|difference|vs|versus|better|worse|similar|like|same/i,
  }

  private languagePatterns: Record<string, RegExp> = {
    python: /python|py|django|flask|pandas|numpy|matplotlib/i,
    javascript: /javascript|js|node|react|vue|angular|typescript|ts/i,
    java: /java|spring|maven|gradle|junit/i,
    cpp: /c\+\+|cpp|c plus plus/i,
    c: /\bc\b|c language|gcc/i,
    swift: /swift|ios|xcode|objective-c/i,
    go: /golang|go language/i,
    rust: /rust|cargo/i,
    php: /php|laravel|symfony/i,
    ruby: /ruby|rails|gem/i,
    sql: /sql|database|query|mysql|postgresql/i,
    html: /html|markup|web/i,
    css: /css|styling|style|sass|scss/i,
    kotlin: /kotlin|android/i,
    csharp: /c#|csharp|\.net|dotnet/i,
    ml: /machine learning|ml|tensorflow|pytorch|scikit|keras|neural|deep learning/i,
  }

  tokenize(text: string): string[] {
    return text
      .toLowerCase()
      .split(/\s+/)
      .filter((token) => token.length > 0)
  }

  removeStopWords(tokens: string[]): string[] {
    return tokens.filter((token) => !this.stopWords.has(token))
  }

  extractKeywords(tokens: string[]): { word: string; frequency: number }[] {
    const frequency: Record<string, number> = {}
    tokens.forEach((token) => {
      frequency[token] = (frequency[token] || 0) + 1
    })
    return Object.entries(frequency)
      .map(([word, freq]) => ({ word, frequency: freq }))
      .sort((a, b) => b.frequency - a.frequency)
      .slice(0, 10)
  }

  detectIntent(text: string): string {
    for (const [intent, pattern] of Object.entries(this.intentPatterns)) {
      if (pattern.test(text)) {
        return intent
      }
    }
    return "general"
  }

  detectLanguage(text: string): string {
    for (const [language, pattern] of Object.entries(this.languagePatterns)) {
      if (pattern.test(text)) {
        return language
      }
    }
    return "unknown"
  }

  extractEntities(text: string): { type: string; value: string }[] {
    const entities: { type: string; value: string }[] = []
    const language = this.detectLanguage(text)
    if (language !== "unknown") {
      entities.push({ type: "language", value: language })
    }
    const codeBlockMatch = text.match(/```[\s\S]*?```/g)
    if (codeBlockMatch) {
      entities.push({ type: "code-block", value: codeBlockMatch[0] })
    }
    return entities
  }

  analyzeSentiment(text: string): "positive" | "negative" | "neutral" {
    const positiveWords = /good|great|excellent|perfect|amazing|awesome|love|best|helpful|thanks/i
    const negativeWords = /bad|terrible|awful|hate|worst|broken|error|problem|issue|fail/i

    const positiveCount = (text.match(positiveWords) || []).length
    const negativeCount = (text.match(negativeWords) || []).length

    if (positiveCount > negativeCount) return "positive"
    if (negativeCount > positiveCount) return "negative"
    return "neutral"
  }

  assessComplexity(text: string): "simple" | "moderate" | "complex" {
    const wordCount = text.split(/\s+/).length
    const hasCodeBlock = /```/.test(text)
    const hasMultipleLanguages = (text.match(/python|javascript|java|cpp|swift/gi) || []).length > 1

    if (hasCodeBlock && hasMultipleLanguages) return "complex"
    if (hasCodeBlock || wordCount > 50) return "moderate"
    return "simple"
  }

  analyze(text: string): NLPResult {
    const tokens = this.tokenize(text)
    const filteredTokens = this.removeStopWords(tokens)
    const keywords = this.extractKeywords(filteredTokens)
    const intent = this.detectIntent(text)
    const detectedLanguage = this.detectLanguage(text)
    const entities = this.extractEntities(text)
    const sentiment = this.analyzeSentiment(text)
    const complexity = this.assessComplexity(text)

    return {
      tokens,
      keywords,
      intent,
      detectedLanguage,
      entities,
      sentiment,
      complexity,
    }
  }
}
