-- Create quiz responses table (user answers)
create table if not exists public.quiz_responses (
  id uuid primary key default gen_random_uuid(),
  attempt_id uuid not null references public.quiz_attempts(id) on delete cascade,
  question_id uuid not null references public.quiz_questions(id) on delete cascade,
  selected_option_id uuid references public.quiz_options(id) on delete set null,
  is_correct boolean,
  created_at timestamp with time zone default now()
);

alter table public.quiz_responses enable row level security;

create policy "quiz_responses_select_own"
  on public.quiz_responses for select
  using (
    exists (
      select 1 from public.quiz_attempts
      where id = attempt_id and user_id = auth.uid()
    )
  );

create policy "quiz_responses_insert_own"
  on public.quiz_responses for insert
  with check (
    exists (
      select 1 from public.quiz_attempts
      where id = attempt_id and user_id = auth.uid()
    )
  );
