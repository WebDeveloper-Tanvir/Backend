-- Create courses table
create table if not exists public.courses (
  id uuid primary key default gen_random_uuid(),
  title text not null,
  description text,
  language text not null,
  difficulty_level text not null check (difficulty_level in ('beginner', 'intermediate', 'advanced')),
  icon text,
  color text,
  created_at timestamp with time zone default now(),
  updated_at timestamp with time zone default now()
);

alter table public.courses enable row level security;

-- Allow anyone to view courses
create policy "courses_select_all"
  on public.courses for select
  using (true);
