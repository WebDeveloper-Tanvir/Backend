-- Create lessons table
create table if not exists public.lessons (
  id uuid primary key default gen_random_uuid(),
  course_id uuid not null references public.courses(id) on delete cascade,
  title text not null,
  description text,
  content text not null,
  code_example text,
  order_index integer not null,
  created_at timestamp with time zone default now(),
  updated_at timestamp with time zone default now()
);

alter table public.lessons enable row level security;

-- Allow anyone to view lessons
create policy "lessons_select_all"
  on public.lessons for select
  using (true);
