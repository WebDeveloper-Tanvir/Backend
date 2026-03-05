-- Create quiz options/answers table
create table if not exists public.quiz_options (
  id uuid primary key default gen_random_uuid(),
  question_id uuid not null references public.quiz_questions(id) on delete cascade,
  option_text text not null,
  is_correct boolean not null default false,
  order_index integer not null,
  created_at timestamp with time zone default now()
);

alter table public.quiz_options enable row level security;

-- Allow anyone to view options
create policy "quiz_options_select_all"
  on public.quiz_options for select
  using (true);
