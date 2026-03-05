-- Create quiz questions table
create table if not exists public.quiz_questions (
  id uuid primary key default gen_random_uuid(),
  quiz_id uuid not null references public.quizzes(id) on delete cascade,
  question_text text not null,
  question_type text not null check (question_type in ('multiple_choice', 'true_false', 'code_snippet')),
  order_index integer not null,
  created_at timestamp with time zone default now(),
  updated_at timestamp with time zone default now()
);

alter table public.quiz_questions enable row level security;

-- Allow anyone to view questions
create policy "quiz_questions_select_all"
  on public.quiz_questions for select
  using (true);
