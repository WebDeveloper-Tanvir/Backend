import { LucideIcon } from 'lucide-react'

export interface Command {
  id: string
  title: string
  description: string
  icon?: LucideIcon
  category: 'Editor' | 'File' | 'View' | 'Terminal' | 'Help'
  shortcut?: string
  action?: () => void | Promise<void>
}

export const ideCommands: Command[] = [
  // Editor commands
  {
    id: 'editor.save',
    title: 'Save',
    description: 'Save current file',
    category: 'Editor',
    shortcut: 'Ctrl+S',
  },
  {
    id: 'editor.saveAll',
    title: 'Save All',
    description: 'Save all open files',
    category: 'Editor',
    shortcut: 'Ctrl+Shift+S',
  },
  {
    id: 'editor.formatCode',
    title: 'Format Code',
    description: 'Format the current file',
    category: 'Editor',
    shortcut: 'Shift+Alt+F',
  },
  {
    id: 'editor.toggleLineComment',
    title: 'Toggle Line Comment',
    description: 'Comment or uncomment current line',
    category: 'Editor',
    shortcut: 'Ctrl+/',
  },
  {
    id: 'editor.toggleBlockComment',
    title: 'Toggle Block Comment',
    description: 'Toggle block comment',
    category: 'Editor',
    shortcut: 'Shift+Alt+A',
  },

  // File commands
  {
    id: 'file.new',
    title: 'New File',
    description: 'Create a new file',
    category: 'File',
    shortcut: 'Ctrl+N',
  },
  {
    id: 'file.open',
    title: 'Open File',
    description: 'Open a file',
    category: 'File',
    shortcut: 'Ctrl+O',
  },
  {
    id: 'file.delete',
    title: 'Delete File',
    description: 'Delete the current file',
    category: 'File',
  },
  {
    id: 'file.rename',
    title: 'Rename File',
    description: 'Rename current file',
    category: 'File',
    shortcut: 'F2',
  },

  // View commands
  {
    id: 'view.toggleSidebar',
    title: 'Toggle Sidebar',
    description: 'Show or hide the sidebar',
    category: 'View',
    shortcut: 'Ctrl+B',
  },
  {
    id: 'view.toggleTerminal',
    title: 'Toggle Terminal',
    description: 'Show or hide the terminal',
    category: 'View',
    shortcut: 'Ctrl+`',
  },
  {
    id: 'view.zoomIn',
    title: 'Zoom In',
    description: 'Increase editor font size',
    category: 'View',
    shortcut: 'Ctrl++',
  },
  {
    id: 'view.zoomOut',
    title: 'Zoom Out',
    description: 'Decrease editor font size',
    category: 'View',
    shortcut: 'Ctrl+-',
  },

  // Terminal commands
  {
    id: 'terminal.new',
    title: 'New Terminal',
    description: 'Open a new terminal',
    category: 'Terminal',
    shortcut: 'Ctrl+Shift+`',
  },
  {
    id: 'terminal.clear',
    title: 'Clear Terminal',
    description: 'Clear terminal output',
    category: 'Terminal',
  },

  // Help commands
  {
    id: 'help.showShortcuts',
    title: 'Keyboard Shortcuts',
    description: 'Show keyboard shortcuts',
    category: 'Help',
    shortcut: 'Ctrl+K Ctrl+S',
  },
]

export function getCommandsByCategory(category: Command['category']): Command[] {
  return ideCommands.filter((cmd) => cmd.category === category)
}

export function searchCommands(query: string): Command[] {
  const lowerQuery = query.toLowerCase()
  return ideCommands.filter(
    (cmd) =>
      cmd.title.toLowerCase().includes(lowerQuery) ||
      cmd.description.toLowerCase().includes(lowerQuery)
  )
}
