/**
 * lib/emailClassifier.ts
 *
 * Email → UserType classifier.
 * Exports both getNavItem() and getSmartNavItem() (alias)
 * so all existing imports work regardless of which name they use.
 */

export type UserType = "student" | "personal" | "org"

const EDU_TLDS = [
  ".edu.in", ".ac.in", ".k12.in", ".ernet.in",
  ".edu",
  ".ac.uk",
  ".edu.au", ".ac.nz", ".ac.za",
  ".edu.sg", ".edu.pk", ".edu.bd", ".edu.np", ".edu.lk",
  ".edu.my", ".edu.ph", ".edu.vn", ".edu.hk",
  ".ac.jp", ".ac.kr", ".edu.cn", ".edu.tw",
  ".edu.br", ".edu.mx", ".edu.co", ".edu.ar", ".edu.pe", ".edu.cl",
  ".edu.tr", ".edu.eg", ".ac.ir", ".edu.gh", ".edu.ng",
  ".edu.pl", ".sch.id",
]

const EDU_KEYWORDS = [
  "university", "college", "institute", "school", "academy",
  "polytechnic", "campus", "seminary",
  "iit.", "nit.", "iiit.", "bits.", "vit.", "dtu.", "thapar.",
  "manipal.", "amity.", "lpu.", "gcet.", "gtu.", "mu.", "du.", "cu.",
  "mit.", "stanford.", "harvard.", "oxford.", "cambridge.", "caltech.",
  "berkeley.", "yale.", "princeton.",
]

const PERSONAL_DOMAINS = new Set([
  "gmail.com", "googlemail.com",
  "outlook.com", "hotmail.com", "hotmail.in", "hotmail.co.uk",
  "live.com", "live.in", "msn.com", "windowslive.com",
  "yahoo.com", "yahoo.in", "yahoo.co.in", "yahoo.co.uk", "yahoo.fr", "yahoo.de",
  "icloud.com", "me.com", "mac.com",
  "protonmail.com", "proton.me", "tutanota.com", "fastmail.com",
  "rediffmail.com",
  "zoho.com", "yandex.com", "yandex.ru",
  "mail.com", "email.com", "inbox.com", "aol.com", "gmx.com", "gmx.net",
])

export function classifyEmail(email: string): UserType {
  if (!email?.includes("@")) return "personal"
  const lower = email.trim().toLowerCase()
  const domain = lower.split("@")[1]
  if (!domain?.includes(".")) return "personal"

  for (const tld of EDU_TLDS) {
    if (domain.endsWith(tld)) return "student"
  }
  if (domain.endsWith(".edu")) return "student"
  for (const kw of EDU_KEYWORDS) {
    if (domain.includes(kw)) return "student"
  }
  if (PERSONAL_DOMAINS.has(domain)) return "personal"
  return "org"
}

export interface NavItem {
  href: string
  label: string
  icon: string
  color: string
  badge: string | null
  description: string
}

function buildNavItem(userType: UserType): NavItem {
  if (userType === "student") {
    return {
      href: "/ide",
      label: "IDE",
      icon: "💻",
      color: "#58a6ff",
      badge: null,
      description: "Open your coding environment",
    }
  }
  return {
    href: "/devmode",
    label: "Dev Mode",
    icon: "⚙",
    color: "#f59e0b",
    badge: "NEW",
    description: "Generate · Analyze · Optimize",
  }
}

/** Primary export used in SmartNavbar */
export const getNavItem = buildNavItem

/** Alias — used in the login page (getSmartNavItem) */
export const getSmartNavItem = buildNavItem

export function getRedirectPath(userType: UserType): string {
  return userType === "student" ? "/ide" : "/devmode"
}

export function accountLabel(userType: UserType): string {
  if (userType === "student") return "Student Account"
  if (userType === "org")     return "Organisation Account"
  return "Personal Account"
}

export function accountEmoji(userType: UserType): string {
  if (userType === "student") return "🎓"
  if (userType === "org")     return "🏢"
  return "👤"
}