// Custom LLM Model for Errorless Chatbot
// Provides code generation, debugging, explanations, optimization, best practices, comparisons

export interface LLMResponse {
  response: string
  codeExample?: string
  explanation?: string
  tips?: string[]
}

export class CustomLLMModel {
  private knowledgeBase: Record<string, Record<string, string>> = {
    python: {
      syntax: "Python uses indentation for code blocks. Use colons (:) to start blocks.",
      functions: "Define functions with def keyword: def function_name(params): ...",
      loops: "Use for and while loops. for item in list: or while condition:",
      lists: "Lists are mutable sequences: my_list = [1, 2, 3]",
      dicts: 'Dictionaries store key-value pairs: my_dict = {"key": "value"}',
      classes: "Define classes with class keyword: class ClassName: ...",
      imports: "Import modules with import or from...import statements",
      errors: "Common errors: NameError, TypeError, ValueError, IndexError, KeyError",
    },
    javascript: {
      syntax: "JavaScript uses semicolons and curly braces for code blocks.",
      functions: "Define functions with function keyword or arrow functions: () => {}",
      loops: "Use for, while, forEach, map, filter, reduce for iteration",
      arrays: "Arrays are mutable: const arr = [1, 2, 3]",
      objects: 'Objects store key-value pairs: const obj = {key: "value"}',
      classes: "Define classes with class keyword: class ClassName {}",
      async: "Use async/await for asynchronous operations",
      errors: "Common errors: ReferenceError, TypeError, SyntaxError, RangeError",
    },
    java: {
      syntax: "Java requires semicolons and curly braces. Strongly typed language.",
      functions: "Define methods: public void methodName(params) {}",
      loops: "Use for, while, do-while, enhanced for loops",
      arrays: "Arrays are fixed size: int[] arr = new int[10]",
      classes: "Define classes: public class ClassName {}",
      inheritance: "Use extends keyword for inheritance",
      interfaces: "Define interfaces: public interface InterfaceName {}",
      errors: "Common errors: NullPointerException, ArrayIndexOutOfBoundsException",
    },
    cpp: {
      syntax: "C++ uses semicolons and curly braces. Requires compilation.",
      functions: "Define functions: return_type functionName(params) {}",
      pointers: "Pointers store memory addresses: int* ptr = &variable",
      arrays: "Arrays: int arr[10] or dynamic with new",
      classes: "Define classes: class ClassName {}",
      templates: "Use templates for generic programming: template<typename T>",
      memory: "Manage memory with new/delete or smart pointers",
      errors: "Common errors: Segmentation fault, memory leak, null pointer",
    },
    swift: {
      syntax: "Swift uses semicolons optional. Modern, safe language.",
      functions: "Define functions: func functionName(params) -> ReturnType {}",
      optionals: "Use ? for optional values: var value: Int?",
      arrays: "Arrays: var arr: [Int] = [1, 2, 3]",
      dictionaries: "Dictionaries: var dict: [String: Int] = [:]",
      classes: "Define classes: class ClassName {}",
      structs: "Define structs: struct StructName {}",
      errors: "Common errors: Force unwrap crash, type mismatch, nil reference",
    },
  }

  generateCode(intent: string, language: string, query: string): string {
    const examples: Record<string, Record<string, string>> = {
      "code-generation": {
        python: `def solve_problem(data):
    """Solution for your problem"""
    result = []
    for item in data:
        result.append(process(item))
    return result`,
        javascript: `function solveProblem(data) {
    const result = [];
    data.forEach(item => {
        result.push(process(item));
    });
    return result;
}`,
        java: `public static List<Object> solveProblem(List<Object> data) {
    List<Object> result = new ArrayList<>();
    for (Object item : data) {
        result.add(process(item));
    }
    return result;
}`,
        cpp: `std::vector<int> solveProblem(std::vector<int> data) {
    std::vector<int> result;
    for (int item : data) {
        result.push_back(process(item));
    }
    return result;
}`,
      },
    }
    return examples[intent]?.[language] || "Code generation template for your query"
  }

  debugCode(language: string, errorMessage: string): string {
    const debugTips: Record<string, string[]> = {
      python: [
        "Check indentation - Python is sensitive to whitespace",
        "Verify variable names are spelled correctly",
        "Ensure all parentheses and brackets are balanced",
        "Check for missing colons after if, for, while, def, class",
        "Use print() statements to debug variable values",
      ],
      javascript: [
        "Check browser console for error messages",
        "Verify all variables are declared before use",
        "Check for missing semicolons (though optional)",
        "Ensure all parentheses and brackets are balanced",
        "Use console.log() for debugging",
      ],
      java: [
        "Check for NullPointerException - verify objects are initialized",
        "Verify array indices are within bounds",
        "Check for type mismatches",
        "Ensure all imports are correct",
        "Use try-catch blocks for exception handling",
      ],
    }
    const tips = debugTips[language] || ["Check syntax", "Verify logic", "Test with sample data"]
    return `Debug Tips for ${language}:\n${tips.map((t, i) => `${i + 1}. ${t}`).join("\n")}`
  }

  explainConcept(language: string, concept: string): string {
    const knowledge = this.knowledgeBase[language]?.[concept.toLowerCase()]
    return knowledge || `Explanation for ${concept} in ${language}`
  }

  optimizeCode(language: string): string {
    const optimizationTips: Record<string, string[]> = {
      python: [
        "Use list comprehensions instead of loops for better performance",
        "Use built-in functions like map(), filter(), reduce()",
        "Avoid global variables - use function parameters",
        "Use generators for large datasets to save memory",
        "Profile your code with cProfile to find bottlenecks",
      ],
      javascript: [
        "Use const/let instead of var for better scoping",
        "Avoid nested callbacks - use async/await or promises",
        "Use arrow functions for cleaner syntax",
        "Memoize expensive function calls",
        "Use Web Workers for CPU-intensive tasks",
      ],
      java: [
        "Use StringBuilder instead of String concatenation in loops",
        "Cache frequently accessed objects",
        "Use appropriate data structures (HashMap vs TreeMap)",
        "Minimize object creation in loops",
        "Use try-with-resources for automatic resource management",
      ],
    }
    const tips = optimizationTips[language] || ["Profile code", "Reduce complexity", "Cache results"]
    return `Optimization Tips for ${language}:\n${tips.map((t, i) => `${i + 1}. ${t}`).join("\n")}`
  }

  bestPractices(language: string): string {
    const practices: Record<string, string[]> = {
      python: [
        "Follow PEP 8 style guide for consistent code",
        "Use meaningful variable and function names",
        "Write docstrings for functions and classes",
        "Use type hints for better code clarity",
        "Write unit tests for your code",
      ],
      javascript: [
        'Use strict mode: "use strict"',
        "Follow consistent naming conventions (camelCase)",
        "Use ESLint for code quality",
        "Write JSDoc comments for functions",
        "Use async/await instead of callbacks",
      ],
      java: [
        "Follow Java naming conventions (PascalCase for classes)",
        "Use access modifiers appropriately (private, protected, public)",
        "Write Javadoc comments for public APIs",
        "Use design patterns (Singleton, Factory, Observer)",
        "Write unit tests with JUnit",
      ],
    }
    const tips = practices[language] || ["Write clean code", "Document your code", "Test thoroughly"]
    return `Best Practices for ${language}:\n${tips.map((t, i) => `${i + 1}. ${t}`).join("\n")}`
  }

  compare(item1: string, item2: string): string {
    const comparisons: Record<string, string> = {
      "python-javascript":
        "Python: Interpreted, indentation-based, great for data science. JavaScript: Event-driven, runs in browsers, async-first.",
      "java-cpp":
        "Java: Platform-independent, garbage collected, verbose. C++: Fast, manual memory management, complex.",
      "python-java": "Python: Easy to learn, dynamic typing, slower. Java: Strongly typed, faster, more verbose.",
      "javascript-typescript":
        "JavaScript: Dynamic typing, flexible. TypeScript: Static typing, better IDE support, compiles to JavaScript.",
    }
    const key = `${item1.toLowerCase()}-${item2.toLowerCase()}`
    return comparisons[key] || `Comparison between ${item1} and ${item2}`
  }

  generateResponse(intent: string, language: string, query: string): LLMResponse {
    let response = ""
    let codeExample = ""
    let explanation = ""
    const tips: string[] = []

    switch (intent) {
      case "code-generation":
        codeExample = this.generateCode(intent, language, query)
        response = `Here's a ${language} solution for your query:\n\n${codeExample}`
        break
      case "debugging":
        response = this.debugCode(language, query)
        tips.push("Check the error message carefully")
        tips.push("Use debugging tools or print statements")
        break
      case "explanation":
        explanation = this.explainConcept(language, query)
        response = explanation
        break
      case "optimization":
        response = this.optimizeCode(language)
        break
      case "best-practices":
        response = this.bestPractices(language)
        break
      case "comparison":
        response = this.compare(query.split(" vs ")[0], query.split(" vs ")[1])
        break
      default:
        response = `I can help you with ${language} programming. Ask me about code generation, debugging, explanations, optimization, best practices, or comparisons.`
    }

    return { response, codeExample, explanation, tips }
  }
}
