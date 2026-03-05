-- Create quiz attempts table
create table if not exists public.quiz_attempts (
  id uuid primary key default gen_random_uuid(),
  user_id uuid not null references auth.users(id) on delete cascade,
  quiz_id uuid not null references public.quizzes(id) on delete cascade,
  score integer,
  total_questions integer,
  percentage_score integer,
  passed boolean,
  time_taken_seconds integer,
  started_at timestamp with time zone default now(),
  completed_at timestamp with time zone,
  created_at timestamp with time zone default now()
);

alter table public.quiz_attempts enable row level security;

create policy "quiz_attempts_select_own"
  on public.quiz_attempts for select
  using (auth.uid() = user_id);

create policy "quiz_attempts_insert_own"
  on public.quiz_attempts for insert
  with check (auth.uid() = user_id);
