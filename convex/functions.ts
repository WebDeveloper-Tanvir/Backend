import { v } from "convex/values"
import { query, mutation } from "./_generated/server"
import { getAuthUserId } from "@convex-dev/auth/server"

// User functions
export const createOrUpdateUser = mutation({
  args: {
    clerkId: v.string(),
    email: v.string(),
    name: v.optional(v.string()),
    profileImage: v.optional(v.string()),
  },
  handler: async (ctx, args) => {
    const existing = await ctx.db
      .query("users")
      .withIndex("by_clerk_id", (q) => q.eq("clerkId", args.clerkId))
      .first()

    if (existing) {
      await ctx.db.patch(existing._id, {
        email: args.email,
        name: args.name,
        profileImage: args.profileImage,
        updatedAt: Date.now(),
      })
      return existing._id
    }

    return await ctx.db.insert("users", {
      clerkId: args.clerkId,
      email: args.email,
      name: args.name,
      profileImage: args.profileImage,
      createdAt: Date.now(),
    })
  },
})

export const getUserByClerkId = query({
  args: { clerkId: v.string() },
  handler: async (ctx, args) => {
    return await ctx.db
      .query("users")
      .withIndex("by_clerk_id", (q) => q.eq("clerkId", args.clerkId))
      .first()
  },
})

// Project functions
export const createProject = mutation({
  args: {
    userId: v.id("users"),
    name: v.string(),
    description: v.optional(v.string()),
  },
  handler: async (ctx, args) => {
    return await ctx.db.insert("projects", {
      userId: args.userId,
      name: args.name,
      description: args.description,
      createdAt: Date.now(),
      updatedAt: Date.now(),
    })
  },
})

export const getProjectsByUserId = query({
  args: { userId: v.id("users") },
  handler: async (ctx, args) => {
    return await ctx.db
      .query("projects")
      .withIndex("by_user_id", (q) => q.eq("userId", args.userId))
      .collect()
  },
})

export const updateProject = mutation({
  args: {
    projectId: v.id("projects"),
    name: v.optional(v.string()),
    description: v.optional(v.string()),
  },
  handler: async (ctx, args) => {
    const { projectId, ...updates } = args
    await ctx.db.patch(projectId, {
      ...updates,
      updatedAt: Date.now(),
    })
    return projectId
  },
})

export const deleteProject = mutation({
  args: { projectId: v.id("projects") },
  handler: async (ctx, args) => {
    await ctx.db.delete(args.projectId)
  },
})

// File functions
export const createFile = mutation({
  args: {
    projectId: v.id("projects"),
    userId: v.id("users"),
    name: v.string(),
    type: v.union(v.literal("file"), v.literal("folder")),
    content: v.optional(v.string()),
    language: v.optional(v.string()),
    parentId: v.optional(v.id("files")),
  },
  handler: async (ctx, args) => {
    return await ctx.db.insert("files", {
      projectId: args.projectId,
      userId: args.userId,
      name: args.name,
      type: args.type,
      content: args.content,
      language: args.language,
      parentId: args.parentId,
      createdAt: Date.now(),
      updatedAt: Date.now(),
    })
  },
})

export const getFilesByProjectId = query({
  args: { projectId: v.id("projects") },
  handler: async (ctx, args) => {
    return await ctx.db
      .query("files")
      .withIndex("by_project_id", (q) => q.eq("projectId", args.projectId))
      .collect()
  },
})

export const updateFile = mutation({
  args: {
    fileId: v.id("files"),
    name: v.optional(v.string()),
    content: v.optional(v.string()),
    language: v.optional(v.string()),
  },
  handler: async (ctx, args) => {
    const { fileId, ...updates } = args
    await ctx.db.patch(fileId, {
      ...updates,
      updatedAt: Date.now(),
    })
    return fileId
  },
})

export const deleteFile = mutation({
  args: { fileId: v.id("files") },
  handler: async (ctx, args) => {
    // Delete file and all children if it's a folder
    const children = await ctx.db
      .query("files")
      .withIndex("by_parent_id", (q) => q.eq("parentId", args.fileId))
      .collect()

    for (const child of children) {
      await ctx.db.delete(child._id)
    }

    await ctx.db.delete(args.fileId)
  },
})

// Algorithm functions
export const createAlgorithm = mutation({
  args: {
    projectId: v.id("projects"),
    userId: v.id("users"),
    name: v.string(),
    description: v.optional(v.string()),
    language: v.string(),
    code: v.string(),
    complexity: v.optional(v.string()),
  },
  handler: async (ctx, args) => {
    return await ctx.db.insert("algorithms", {
      projectId: args.projectId,
      userId: args.userId,
      name: args.name,
      description: args.description,
      language: args.language,
      code: args.code,
      complexity: args.complexity,
      createdAt: Date.now(),
      updatedAt: Date.now(),
    })
  },
})

export const getAlgorithmsByProjectId = query({
  args: { projectId: v.id("projects") },
  handler: async (ctx, args) => {
    return await ctx.db
      .query("algorithms")
      .withIndex("by_project_id", (q) => q.eq("projectId", args.projectId))
      .collect()
  },
})

export const updateAlgorithm = mutation({
  args: {
    algorithmId: v.id("algorithms"),
    name: v.optional(v.string()),
    description: v.optional(v.string()),
    code: v.optional(v.string()),
    complexity: v.optional(v.string()),
  },
  handler: async (ctx, args) => {
    const { algorithmId, ...updates } = args
    await ctx.db.patch(algorithmId, {
      ...updates,
      updatedAt: Date.now(),
    })
    return algorithmId
  },
})

export const deleteAlgorithm = mutation({
  args: { algorithmId: v.id("algorithms") },
  handler: async (ctx, args) => {
    await ctx.db.delete(args.algorithmId)
  },
})
