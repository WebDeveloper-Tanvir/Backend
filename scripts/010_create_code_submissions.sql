-- Create code submissions table
create table if not exists public.code_submissions (
  id uuid primary key default gen_random_uuid(),
  user_id uuid not null references auth.users(id) on delete cascade,
  lesson_id uuid references public.lessons(id) on delete set null,
  language text not null,
  code_content text not null,
  output text,
  error_message text,
  execution_time_ms integer,
  is_correct boolean,
  created_at timestamp with time zone default now(),
  updated_at timestamp with time zone default now()
);

alter table public.code_submissions enable row level security;

create policy "code_submissions_select_own"
  on public.code_submissions for select
  using (auth.uid() = user_id);

create policy "code_submissions_insert_own"
  on public.code_submissions for insert
  with check (auth.uid() = user_id);
