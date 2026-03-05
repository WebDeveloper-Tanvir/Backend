'use client'

import { useEffect, useState, useCallback } from 'react'
import { Command, searchCommands, ideCommands } from '@/lib/commands'
import { cn } from '@/lib/utils'
import {
  CommandDialog,
  CommandEmpty,
  CommandGroup,
  CommandInput,
  CommandItem,
  CommandList,
} from '@/components/ui/command'
import { Badge } from '@/components/ui/badge'

export interface CommandPaletteProps {
  isOpen?: boolean
  onClose?: () => void
}

export function CommandPalette({ isOpen = false, onClose }: CommandPaletteProps) {
  const [open, setOpen] = useState(isOpen)
  const [searchResults, setSearchResults] = useState<Command[]>(ideCommands)

  useEffect(() => {
    const down = (e: KeyboardEvent) => {
      if (e.key === 'k' && (e.metaKey || e.ctrlKey)) {
        e.preventDefault()
        setOpen((prev) => !prev)
      }
    }

    document.addEventListener('keydown', down)
    return () => document.removeEventListener('keydown', down)
  }, [])

  const handleSearch = useCallback((value: string) => {
    if (value.trim() === '') {
      setSearchResults(ideCommands)
    } else {
      setSearchResults(searchCommands(value))
    }
  }, [])

  const handleSelectCommand = useCallback(
    (command: Command) => {
      if (command.action) {
        command.action()
      }
      setOpen(false)
      onClose?.()
    },
    [onClose]
  )

  const handleOpenChange = useCallback(
    (newOpen: boolean) => {
      setOpen(newOpen)
      if (!newOpen) {
        onClose?.()
      }
    },
    [onClose]
  )

  const groupedCommands = searchResults.reduce(
    (acc, cmd) => {
      if (!acc[cmd.category]) {
        acc[cmd.category] = []
      }
      acc[cmd.category].push(cmd)
      return acc
    },
    {} as Record<string, Command[]>
  )

  return (
    <CommandDialog open={open} onOpenChange={handleOpenChange}>
      <CommandInput
        placeholder="Search commands or files..."
        onValueChange={handleSearch}
      />
      <CommandList>
        <CommandEmpty>No commands found.</CommandEmpty>
        {Object.entries(groupedCommands).map(([category, commands]) => (
          <CommandGroup key={category} heading={category}>
            {commands.map((command) => (
              <CommandItem
                key={command.id}
                value={command.id}
                onSelect={() => handleSelectCommand(command)}
              >
                <div className="flex w-full items-center justify-between">
                  <div className="flex flex-col">
                    <span>{command.title}</span>
                    <span className="text-xs text-muted-foreground">
                      {command.description}
                    </span>
                  </div>
                  {command.shortcut && (
                    <Badge variant="secondary" className="ml-2 text-xs">
                      {command.shortcut}
                    </Badge>
                  )}
                </div>
              </CommandItem>
            ))}
          </CommandGroup>
        ))}
      </CommandList>
    </CommandDialog>
  )
}
