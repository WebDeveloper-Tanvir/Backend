/**
 * VS Code IDE Configuration
 * This file contains all configuration and utilities for the VS Code-like IDE implementation
 */

export const IDE_FEATURES = {
  MONACO_EDITOR: 'Monaco Editor - Professional code editor with syntax highlighting, autocomplete, and smart intellisense',
  THEME_SUPPORT: 'Theme Support - Dark, Light, and High Contrast themes with system preference detection',
  COMMAND_PALETTE: 'Command Palette - Searchable command palette (Cmd/Ctrl+K) for IDE operations',
  CODE_FOLDING: 'Code Folding - Collapse/expand code sections for better navigation',
  MINIMAP: 'Minimap - Visual overview and quick navigation in large files',
  SYNTAX_HIGHLIGHTING: 'Syntax Highlighting - Multi-language support with proper color coding',
  FONT_SCALING: 'Font Scaling - Adjustable editor font size (10px - 32px)',
  BRACKET_COLORIZATION: 'Bracket Colorization - Color-coded bracket pairs for better code readability',
  LINE_NUMBERS: 'Line Numbers - Easy line reference and navigation',
  WORD_WRAP: 'Word Wrap - Automatic line wrapping for better readability',
} as const

export const SUPPORTED_LANGUAGES = [
  'javascript',
  'typescript',
  'jsx',
  'tsx',
  'python',
  'java',
  'cpp',
  'c',
  'swift',
  'go',
  'rust',
  'php',
  'ruby',
  'sql',
  'html',
  'css',
  'json',
  'xml',
  'yaml',
  'markdown',
] as const

export const KEYBOARD_SHORTCUTS = {
  COMMAND_PALETTE: {
    mac: 'Cmd+K',
    windows: 'Ctrl+K',
    linux: 'Ctrl+K',
  },
  SAVE: {
    mac: 'Cmd+S',
    windows: 'Ctrl+S',
    linux: 'Ctrl+S',
  },
  FORMAT_CODE: {
    mac: 'Shift+Option+F',
    windows: 'Shift+Alt+F',
    linux: 'Shift+Alt+F',
  },
  TOGGLE_COMMENT: {
    mac: 'Cmd+/',
    windows: 'Ctrl+/',
    linux: 'Ctrl+/',
  },
  TOGGLE_TERMINAL: {
    mac: 'Ctrl+`',
    windows: 'Ctrl+`',
    linux: 'Ctrl+`',
  },
  TOGGLE_SIDEBAR: {
    mac: 'Cmd+B',
    windows: 'Ctrl+B',
    linux: 'Ctrl+B',
  },
} as const

export interface IDESettings {
  theme: 'dark' | 'light' | 'high-contrast'
  fontSize: number
  fontFamily: string
  minimap: boolean
  lineNumbers: boolean
  wordWrap: boolean
  folding: boolean
  bracketColorization: boolean
  showUnused: boolean
  autoSave: boolean
  showCommandPaletteHint: boolean
}

export const DEFAULT_IDE_SETTINGS: IDESettings = {
  theme: 'dark',
  fontSize: 14,
  fontFamily: 'Fira Code, monospace',
  minimap: true,
  lineNumbers: true,
  wordWrap: true,
  folding: true,
  bracketColorization: true,
  showUnused: true,
  autoSave: false,
  showCommandPaletteHint: true,
}

export function getShortcutForPlatform(
  shortcuts: Record<'mac' | 'windows' | 'linux', string>
): string {
  if (typeof window === 'undefined') return shortcuts.windows

  const isMac = /Mac|iPhone|iPad|iPod/.test(navigator.platform)
  const isWindows = /Win/.test(navigator.platform)

  if (isMac) return shortcuts.mac
  if (isWindows) return shortcuts.windows
  return shortcuts.linux
}

export function getMonacoTheme(
  theme: 'dark' | 'light' | 'high-contrast'
): 'vs' | 'vs-dark' | 'hc-black' {
  const themeMap = {
    dark: 'vs-dark' as const,
    light: 'vs' as const,
    'high-contrast': 'hc-black' as const,
  }
  return themeMap[theme]
}
