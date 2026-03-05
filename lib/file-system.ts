// In-memory file system for the IDE
import type { FileNode } from "@/components/ide/file-explorer"

export class FileSystem {
  private files: FileNode[] = []
  private nextId = 1

  constructor() {
    this.initializeDefaultFiles()
  }

  private initializeDefaultFiles() {
    this.files = [
      {
        id: "1",
        name: "projects",
        type: "folder",
        children: [
          {
            id: "2",
            name: "hello-world.py",
            type: "file",
            language: "python",
            content: `# Hello World in Python
print("Hello, World!")
print("Welcome to Errorless IDE")

# Variables and Data Types
name = "Developer"
age = 25
print(f"Name: {name}, Age: {age}")

# Lists
numbers = [1, 2, 3, 4, 5]
print("Numbers:", numbers)

# Loops
for num in numbers:
    print(f"Number: {num}")`,
          },
          {
            id: "3",
            name: "hello-world.js",
            type: "file",
            language: "javascript",
            content: `// Hello World in JavaScript
console.log("Hello, World!");
console.log("Welcome to Errorless IDE");

// Variables
const name = "Developer";
const age = 25;
console.log(\`Name: \${name}, Age: \${age}\`);

// Arrays
const numbers = [1, 2, 3, 4, 5];
console.log("Numbers:", numbers);

// Loops
numbers.forEach(num => {
    console.log(\`Number: \${num}\`);
});`,
          },
          {
            id: "4",
            name: "algorithms",
            type: "folder",
            children: [
              {
                id: "5",
                name: "bubble-sort.py",
                type: "file",
                language: "python",
                content: `# Bubble Sort Algorithm
def bubble_sort(arr):
    n = len(arr)
    for i in range(n):
        for j in range(0, n - i - 1):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
    return arr

# Test
numbers = [64, 34, 25, 12, 22, 11, 90]
print("Original:", numbers)
print("Sorted:", bubble_sort(numbers))`,
              },
              {
                id: "6",
                name: "fibonacci.js",
                type: "file",
                language: "javascript",
                content: `// Fibonacci Sequence
function fibonacci(n) {
    if (n <= 1) return n;
    return fibonacci(n - 1) + fibonacci(n - 2);
}

// Generate Fibonacci sequence
console.log("Fibonacci Sequence:");
for (let i = 0; i < 10; i++) {
    console.log(\`F(\${i}) = \${fibonacci(i)}\`);
}`,
              },
            ],
          },
        ],
      },
    ]
    this.nextId = 7
  }

  getFiles(): FileNode[] {
    return this.files
  }

  getFileById(id: string): FileNode | null {
    return this.findFileById(this.files, id)
  }

  private findFileById(nodes: FileNode[], id: string): FileNode | null {
    for (const node of nodes) {
      if (node.id === id) return node
      if (node.children) {
        const found = this.findFileById(node.children, id)
        if (found) return found
      }
    }
    return null
  }

  createFile(parentId: string, name: string, language = "python"): FileNode {
    const newFile: FileNode = {
      id: String(this.nextId++),
      name,
      type: "file",
      language,
      content: `// New file: ${name}\n`,
    }

    const parent = this.getFileById(parentId)
    if (parent && parent.type === "folder") {
      if (!parent.children) parent.children = []
      parent.children.push(newFile)
    }

    return newFile
  }

  createFolder(parentId: string, name: string): FileNode {
    const newFolder: FileNode = {
      id: String(this.nextId++),
      name,
      type: "folder",
      children: [],
    }

    const parent = this.getFileById(parentId)
    if (parent && parent.type === "folder") {
      if (!parent.children) parent.children = []
      parent.children.push(newFolder)
    }

    return newFolder
  }

  deleteFile(fileId: string): boolean {
    return this.deleteFileRecursive(this.files, fileId)
  }

  private deleteFileRecursive(nodes: FileNode[], fileId: string): boolean {
    for (let i = 0; i < nodes.length; i++) {
      if (nodes[i].id === fileId) {
        nodes.splice(i, 1)
        return true
      }
      if (nodes[i].children) {
        if (this.deleteFileRecursive(nodes[i].children!, fileId)) {
          return true
        }
      }
    }
    return false
  }

  updateFileContent(fileId: string, content: string): boolean {
    const file = this.getFileById(fileId)
    if (file && file.type === "file") {
      file.content = content
      return true
    }
    return false
  }

  renameFile(fileId: string, newName: string): boolean {
    const file = this.getFileById(fileId)
    if (file) {
      file.name = newName
      return true
    }
    return false
  }
}
