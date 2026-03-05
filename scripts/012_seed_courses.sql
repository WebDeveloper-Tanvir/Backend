-- Seed initial courses
insert into public.courses (title, description, language, difficulty_level, icon, color) values
('Python Basics', 'Learn the fundamentals of Python programming', 'python', 'beginner', '🐍', '#3776ab'),
('Python Advanced', 'Master advanced Python concepts and patterns', 'python', 'advanced', '🐍', '#3776ab'),
('C Fundamentals', 'Introduction to C programming language', 'c', 'beginner', '⚙️', '#a8b9cc'),
('C++ Essentials', 'Learn C++ from basics to advanced concepts', 'cpp', 'beginner', '⚙️', '#00599c'),
('Java Basics', 'Get started with Java programming', 'java', 'beginner', '☕', '#007396'),
('Swift for iOS', 'Learn Swift programming for iOS development', 'swift', 'intermediate', '🍎', '#fa7343'),
('Machine Learning', 'Introduction to ML with Python', 'python', 'intermediate', '🤖', '#ff6b6b'),
('Data Structures', 'Master data structures and algorithms', 'python', 'intermediate', '📊', '#4ecdc4'),
('OOP Concepts', 'Object-Oriented Programming fundamentals', 'java', 'intermediate', '🏗️', '#95e1d3'),
('Deep Learning', 'Neural networks and deep learning basics', 'python', 'advanced', '🧠', '#a29bfe')
on conflict do nothing;
