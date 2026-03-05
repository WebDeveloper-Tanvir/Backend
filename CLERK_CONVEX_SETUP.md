# Clerk & Convex Integration Setup Guide

This guide explains how to complete the Clerk authentication and Convex backend integration for the Errorless IDE application.

## Prerequisites

- Node.js 18+ installed
- Git repository connected
- Vercel account (optional, for deployment)

## Step 1: Clerk Authentication Setup

### 1.1 Create Clerk Application

1. Go to [clerk.com](https://clerk.com)
2. Sign up or log in to your Clerk account
3. Create a new application
4. Choose "Next.js" as your framework
5. Copy your API keys

### 1.2 Add Environment Variables

In your Vercel project or local `.env.local` file, add:

```
NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY=your_publishable_key_here
CLERK_SECRET_KEY=your_secret_key_here
```

### 1.3 Configure Clerk URLs (Optional)

In your Clerk dashboard, set the following redirect URLs:

- **Sign-in URL**: `http://localhost:3000/sign-in` (development)
- **Sign-up URL**: `http://localhost:3000/sign-up` (development)
- **After sign-in URL**: `http://localhost:3000/ide`
- **After sign-up URL**: `http://localhost:3000/ide`

## Step 2: Convex Backend Setup

### 2.1 Create Convex Project

1. Go to [convex.dev](https://convex.dev)
2. Sign up or log in
3. Create a new project
4. Select your deployment region
5. Get your deployment URL

### 2.2 Add Environment Variables

Add the Convex URL to your environment:

```
NEXT_PUBLIC_CONVEX_URL=your_convex_deployment_url
```

### 2.3 Deploy Convex Schema

1. Install Convex CLI globally (if not already installed):
   ```bash
   npm install -g convex
   ```

2. Deploy the schema and functions:
   ```bash
   npx convex deploy
   ```

3. Follow the prompts to authenticate with your Convex account

## Step 3: Environment Variables Summary

Create a `.env.local` file with all required variables:

```
# Clerk
NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY=pk_test_...
CLERK_SECRET_KEY=sk_test_...

# Convex
NEXT_PUBLIC_CONVEX_URL=https://your-project.convex.cloud
```

## Step 4: Running the Application

1. Install dependencies:
   ```bash
   npm install
   ```

2. Run the development server:
   ```bash
   npm run dev
   ```

3. Open [http://localhost:3000](http://localhost:3000)

## Testing the Integration

### Test Clerk Authentication

1. Visit `/sign-up` to create an account
2. Sign up with email and password
3. You'll be redirected to `/ide` after successful signup
4. Visit `/sign-in` to test login
5. Visit `/ide` (protected route) - should redirect to sign-in if not authenticated

### Test Convex Integration

1. Open the IDE page (you must be signed in)
2. Create a new project (if UI is connected)
3. Check your Convex dashboard to see the data synced
4. User information should be created in the users table
5. Projects should appear in the projects table

## File Structure

New/Modified Files:

```
app/
├── layout.tsx (Updated with Clerk & Convex providers)
├── sign-in/[[...index]]/page.tsx (Clerk sign-in page)
├── sign-up/[[...index]]/page.tsx (Clerk sign-up page)
└── ide/page.tsx (Updated with Clerk-Convex sync)

components/
└── convex-provider.tsx (Convex client provider)

convex/
├── schema.ts (Database schema)
├── functions.ts (API functions)
└── _generated/ (Auto-generated files)

hooks/
└── use-clerk-convex-sync.ts (Clerk-Convex sync hook)

middleware.ts (Updated with Clerk middleware)
```

## Key Features Implemented

### File Explorer Improvements
- Folders (projects, algorithms) are collapsed by default
- Users can expand/collapse folders with chevron icons

### Clerk Authentication
- Sign-up and sign-in pages
- Protected routes (IDE requires authentication)
- User profile management
- Automatic session handling

### Convex Backend
- User management with Clerk integration
- Project CRUD operations
- File management with hierarchy support
- Algorithm storage and retrieval
- Automatic timestamp tracking

### Clerk-Convex Sync
- Automatic user syncing from Clerk to Convex
- User profile updates (name, email, profile image)
- User lookup by Clerk ID

## Troubleshooting

### "Module not found: '@clerk/nextjs'"
- Run `npm install` to install all dependencies
- Clear node_modules and reinstall: `rm -rf node_modules && npm install`

### "NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY is missing"
- Add environment variables to your `.env.local` file
- Restart the development server

### "ConvexProvider not found"
- Ensure the ConvexClientProvider is properly imported and wrapped in layout.tsx
- Check that NEXT_PUBLIC_CONVEX_URL environment variable is set

### "Clerk user not syncing to Convex"
- Check browser console for errors
- Verify Clerk keys are correct
- Ensure Convex deployment is successful
- Check Convex dashboard for database records

## Next Steps

1. Connect the IDE file system to use Convex instead of in-memory storage
2. Implement real-time file synchronization
3. Add team collaboration features
4. Implement sharing and permissions
5. Add version history and file recovery

## Support

- Clerk Documentation: https://clerk.com/docs
- Convex Documentation: https://docs.convex.dev
- Next.js Documentation: https://nextjs.org/docs
