"use client"

import { createContext, useContext, useState, useEffect, ReactNode } from "react"
import { classifyEmail, type UserType } from "@/lib/emailClassifier"

export interface AuthUser {
  id: string
  name: string
  email: string
  avatar?: string
  userType: UserType
}

interface Ctx {
  user: AuthUser | null
  setUser: (u: AuthUser | null) => void
  logout: () => void
  isLoading: boolean
}

const UserContext = createContext<Ctx>({
  user: null, setUser: () => {}, logout: () => {}, isLoading: true,
})

const KEY = "errorless_user"

export function UserProvider({ children }: { children: ReactNode }) {
  const [user, _setUser] = useState<AuthUser | null>(null)
  const [isLoading, setIsLoading] = useState(true)

  useEffect(() => {
    try {
      const raw = localStorage.getItem(KEY)
      if (raw) {
        const u = JSON.parse(raw) as AuthUser
        // Always re-classify on hydrate so logic changes take effect
        u.userType = classifyEmail(u.email)
        _setUser(u)
      }
    } catch {
      localStorage.removeItem(KEY)
    } finally {
      setIsLoading(false)
    }
  }, [])

  const setUser = (u: AuthUser | null) => {
    if (u) {
      u.userType = classifyEmail(u.email)
      localStorage.setItem(KEY, JSON.stringify(u))
    } else {
      localStorage.removeItem(KEY)
    }
    _setUser(u)
  }

  const logout = () => {
    localStorage.removeItem(KEY)
    _setUser(null)
    window.location.href = "/"
  }

  return (
    <UserContext.Provider value={{ user, setUser, logout, isLoading }}>
      {children}
    </UserContext.Provider>
  )
}

export const useUser = () => useContext(UserContext)

/** Build an AuthUser from a Supabase user object */
export function buildAuthUser(supabaseUser: {
  id: string
  email?: string
  user_metadata?: { full_name?: string; name?: string; avatar_url?: string }
}): AuthUser {
  const email = supabaseUser.email || ""
  return {
    id: supabaseUser.id,
    email,
    name:
      supabaseUser.user_metadata?.full_name ||
      supabaseUser.user_metadata?.name ||
      email.split("@")[0],
    avatar: supabaseUser.user_metadata?.avatar_url,
    userType: classifyEmail(email),
  }
}