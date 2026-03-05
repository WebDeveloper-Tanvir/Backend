"use client"

import { useState } from "react"
import { ChevronRight, ChevronDown, FileCode, FileJson, FileText, Folder, FolderOpen, Plus, FolderPlus, Trash2 } from "lucide-react"

export interface FileNode {
  id: string
  name: string
  type: "file" | "folder"
  children?: FileNode[]
  content?: string
  language?: string
}

interface FileExplorerProps {
  files: FileNode[]
  onSelectFile: (file: FileNode) => void
  onCreateFile: (parentId: string, name: string) => void
  onCreateFolder: (parentId: string, name: string) => void
  onDeleteFile: (fileId: string) => void
  selectedFileId?: string
}

function getFileIcon(name: string) {
  const ext = name.split(".").pop()?.toLowerCase()
  const base = "shrink-0"
  if (["js", "jsx", "ts", "tsx"].includes(ext ?? "")) return <FileCode size={14} className={`${base} text-[#00b4d8]`} />
  if (ext === "py") return <FileCode size={14} className={`${base} text-[#00c9a7]`} />
  if (ext === "json") return <FileJson size={14} className={`${base} text-[#ffd166]`} />
  if (["md", "txt"].includes(ext ?? "")) return <FileText size={14} className={`${base} text-[rgba(200,230,220,0.5)]`} />
  return <FileCode size={14} className={`${base} text-[#4361ee]`} />
}

export function FileExplorer({ files, onSelectFile, onCreateFile, onCreateFolder, onDeleteFile, selectedFileId }: FileExplorerProps) {
  const [expanded, setExpanded] = useState<Set<string>>(new Set(["1"]))
  const [newItemName, setNewItemName] = useState("")
  const [creatingIn, setCreatingIn] = useState<string | null>(null)
  const [creatingType, setCreatingType] = useState<"file" | "folder" | null>(null)
  const [hoveredId, setHoveredId] = useState<string | null>(null)

  const toggle = (id: string) => {
    const next = new Set(expanded)
    next.has(id) ? next.delete(id) : next.add(id)
    setExpanded(next)
  }

  const startCreate = (parentId: string, type: "file" | "folder") => {
    setCreatingIn(parentId)
    setCreatingType(type)
    setNewItemName("")
    if (!expanded.has(parentId)) {
      const next = new Set(expanded)
      next.add(parentId)
      setExpanded(next)
    }
  }

  const confirmCreate = () => {
    if (newItemName.trim() && creatingIn && creatingType) {
      creatingType === "file" ? onCreateFile(creatingIn, newItemName) : onCreateFolder(creatingIn, newItemName)
      setCreatingIn(null); setCreatingType(null); setNewItemName("")
    }
  }

  const renderTree = (nodes: FileNode[], depth = 0): React.ReactNode =>
    nodes.map(node => {
      const isSelected = selectedFileId === node.id
      const isHovered  = hoveredId === node.id
      const isExpanded = expanded.has(node.id)

      return (
        <div key={node.id}>
          <div
            className="relative flex items-center gap-1.5 py-[3px] pr-2 cursor-pointer select-none group transition-all duration-100"
            style={{
              paddingLeft: `${depth * 14 + 10}px`,
              background: isSelected
                ? "linear-gradient(90deg, rgba(0,201,167,0.18), rgba(0,201,167,0.06))"
                : isHovered ? "rgba(0,201,167,0.05)" : "transparent",
            }}
            onClick={() => node.type === "file" ? onSelectFile(node) : toggle(node.id)}
            onMouseEnter={() => setHoveredId(node.id)}
            onMouseLeave={() => setHoveredId(null)}
          >
            {/* Active indicator */}
            {isSelected && (
              <div className="absolute left-0 top-0 bottom-0 w-[2px] rounded-r"
                style={{ background: "linear-gradient(180deg, #00c9a7, #4361ee)" }} />
            )}

            {/* Expand chevron */}
            {node.type === "folder" ? (
              <span className="text-[rgba(200,230,220,0.4)] shrink-0">
                {isExpanded ? <ChevronDown size={13} /> : <ChevronRight size={13} />}
              </span>
            ) : (
              <span className="w-[13px] shrink-0" />
            )}

            {/* Icon */}
            {node.type === "folder"
              ? isExpanded
                ? <FolderOpen size={14} className="shrink-0 text-[#ffd166]" />
                : <Folder     size={14} className="shrink-0 text-[#ffd166]" />
              : getFileIcon(node.name)
            }

            {/* Name */}
            <span className={`flex-1 text-xs truncate font-medium tracking-wide ${
              isSelected ? "text-[#00c9a7]" : "text-[rgba(200,230,220,0.75)]"
            } group-hover:text-white transition-colors`}>
              {node.name}
            </span>

            {/* Actions */}
            <div className="flex gap-0.5 opacity-0 group-hover:opacity-100 transition-opacity ml-1">
              {node.type === "folder" && (
                <>
                  <button
                    onClick={e => { e.stopPropagation(); startCreate(node.id, "file") }}
                    className="w-5 h-5 flex items-center justify-center rounded hover:bg-[rgba(0,201,167,0.15)] text-[rgba(200,230,220,0.5)] hover:text-[#00c9a7] transition-colors"
                  ><Plus size={11} /></button>
                  <button
                    onClick={e => { e.stopPropagation(); startCreate(node.id, "folder") }}
                    className="w-5 h-5 flex items-center justify-center rounded hover:bg-[rgba(0,201,167,0.15)] text-[rgba(200,230,220,0.5)] hover:text-[#00c9a7] transition-colors"
                  ><FolderPlus size={11} /></button>
                </>
              )}
              <button
                onClick={e => { e.stopPropagation(); onDeleteFile(node.id) }}
                className="w-5 h-5 flex items-center justify-center rounded hover:bg-[rgba(255,77,109,0.15)] text-[rgba(200,230,220,0.5)] hover:text-[#ff4d6d] transition-colors"
              ><Trash2 size={11} /></button>
            </div>
          </div>

          {/* Create input */}
          {node.type === "folder" && isExpanded && creatingIn === node.id && (
            <div className="flex items-center gap-1.5 py-1" style={{ paddingLeft: `${(depth + 1) * 14 + 10}px` }}>
              {creatingType === "file" ? <FileCode size={13} className="text-[#00c9a7]" /> : <Folder size={13} className="text-[#ffd166]" />}
              <input
                autoFocus
                value={newItemName}
                onChange={e => setNewItemName(e.target.value)}
                onKeyDown={e => { if (e.key === "Enter") confirmCreate(); if (e.key === "Escape") setCreatingIn(null) }}
                placeholder={`New ${creatingType}…`}
                className="flex-1 px-2 py-0.5 rounded text-xs text-[#00c9a7] outline-none font-mono"
                style={{ background: "rgba(0,201,167,0.08)", border: "1px solid rgba(0,201,167,0.3)" }}
              />
            </div>
          )}

          {/* Children */}
          {node.type === "folder" && isExpanded && node.children && (
            <div>{renderTree(node.children, depth + 1)}</div>
          )}
        </div>
      )
    })

  return (
    <div className="flex flex-col h-full" style={{ background: "#030e09", borderRight: "1px solid rgba(0,201,167,0.1)" }}>
      {/* Header */}
      <div className="flex items-center justify-between px-3 py-2.5 shrink-0"
        style={{ borderBottom: "1px solid rgba(0,201,167,0.1)" }}>
        <span className="text-[10px] font-bold tracking-widest uppercase text-[rgba(200,230,220,0.35)]">Explorer</span>
      </div>

      {/* Tree */}
      <div className="flex-1 overflow-y-auto py-1 scrollbar-thin">
        {renderTree(files)}
      </div>
    </div>
  )
}
