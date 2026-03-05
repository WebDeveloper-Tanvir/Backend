import { useState, useEffect } from 'react'

export interface MonacoEditorOptions {
  fontSize?: number
  theme?: 'vs' | 'vs-dark' | 'hc-black'
  minimap?: boolean
  wordWrap?: 'on' | 'off' | 'wordWrapColumn' | 'bounded'
  folding?: boolean
  showUnused?: boolean
  bracketPairColorization?: boolean
  smoothScrolling?: boolean
}

export function useMonacoEditor(initialOptions?: MonacoEditorOptions) {
  const [theme, setTheme] = useState<'vs' | 'vs-dark' | 'hc-black'>(initialOptions?.theme || 'vs-dark')
  const [fontSize, setFontSize] = useState(initialOptions?.fontSize || 14)

  // Detect system theme preference
  useEffect(() => {
    const mediaQuery = window.matchMedia('(prefers-color-scheme: dark)')
    const handleThemeChange = (e: MediaQueryListEvent | MediaQueryList) => {
      setTheme(e.matches ? 'vs-dark' : 'vs')
    }
    handleThemeChange(mediaQuery)
    mediaQuery.addEventListener('change', handleThemeChange)
    return () => mediaQuery.removeEventListener('change', handleThemeChange)
  }, [])

  const getEditorOptions = () => ({
    minimap: { enabled: initialOptions?.minimap ?? true },
    fontSize,
    fontFamily: 'Fira Code, monospace',
    lineNumbers: 'on',
    scrollBeyondLastLine: false,
    automaticLayout: true,
    tabSize: 2,
    wordWrap: (initialOptions?.wordWrap ?? 'on') as 'on' | 'off' | 'wordWrapColumn' | 'bounded',
    folding: initialOptions?.folding ?? true,
    foldingStrategy: 'indentation',
    lineDecorationsWidth: 20,
    lineNumbersMinChars: 3,
    renderLineHighlight: 'line',
    roundedSelection: false,
    smoothScrolling: initialOptions?.smoothScrolling ?? true,
    showUnused: initialOptions?.showUnused ?? true,
    bracketPairColorization: {
      enabled: initialOptions?.bracketPairColorization ?? true,
    },
  })

  const updateFontSize = (delta: number) => {
    setFontSize((prev) => Math.max(10, Math.min(32, prev + delta)))
  }

  return {
    theme,
    setTheme,
    fontSize,
    updateFontSize,
    setFontSize,
    editorOptions: getEditorOptions(),
  }
}
