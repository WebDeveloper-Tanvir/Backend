import { createBrowserClient } from "@supabase/ssr"

export function createClient() {
  return createBrowserClient(
    process.env.SUPABASE_SUPABASE_NEXT_PUBLIC_SUPABASE_URL|| 'https://ynmnofpzebzpcterczvd.supabase.co'!,
    process.env.SUPABASE_NEXT_PUBLIC_SUPABASE_ANON_KEY_ANON_KEY|| 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InlubW5vZnB6ZWJ6cGN0ZXJjenZkIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NNzAyMDQ5MzcsImV4cCI6MjA4NTc4MDkzN30.fx1c5iTyXrfbSO1OmjQ2viTUFrgLHs-URvx6TUuhUIQ'!,
  )
}
