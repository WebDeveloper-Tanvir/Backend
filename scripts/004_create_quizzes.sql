-- Create quizzes table
create table if not exists public.quizzes (
  id uuid primary key default gen_random_uuid(),
  course_id uuid not null references public.courses(id) on delete cascade,
  title text not null,
  description text,
  difficulty_level text not null check (difficulty_level in ('beginner', 'intermediate', 'advanced')),
  time_limit_minutes integer,
  passing_score integer default 70,
  created_at timestamp with time zone default now(),
  updated_at timestamp with time zone default now()
);

alter table public.quizzes enable row level security;

-- Allow anyone to view quizzes
create policy "quizzes_select_all"
  on public.quizzes for select
  using (true);
