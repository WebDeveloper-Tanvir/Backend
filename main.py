"""
FastAPI Backend - Rule-based UI Generator
"""

import sys
import subprocess
import os

def install_package(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package, "--quiet"])

# Install dependencies
try:
    import uvicorn
except ImportError:
    install_package("uvicorn")
    import uvicorn

try:
    import nest_asyncio
except ImportError:
    install_package("nest_asyncio")
    import nest_asyncio

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional

# Apply nest_asyncio
nest_asyncio.apply()

# Import local modules
try:
    from intent_parser import IntentParser
    from planner import Planner
    from code_generator import CodeGenerator
    from code_validator import CodeValidator
except ImportError as e:
    print(f"Warning: Could not import modules: {e}")
    print("Make sure all Python files are in the same directory")
    sys.exit(1)

# Initialize FastAPI app
app = FastAPI(
    title="Rule-Based UI Generator",
    description="Generate React UIs without LLM",
    version="1.0.0"
)

# CRITICAL: CORS Configuration - Allow ALL origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize pipeline components
intent_parser = IntentParser()
planner = Planner()
code_generator = CodeGenerator()
code_validator = CodeValidator()

# Request/Response models
class GenerateRequest(BaseModel):
    prompt: str
    current_code: Optional[str] = None

class GenerateResponse(BaseModel):
    code: str
    explanation: str
    plan: dict
    validation: dict

@app.get("/")
async def root():
    return {
        "message": "Rule-Based UI Generator API",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "health": "/health",
            "generate": "/api/generate",
            "docs": "/docs"
        }
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "pipeline": "rule-based",
        "components": {
            "intent_parser": "active",
            "planner": "active",
            "code_generator": "active",
            "code_validator": "active"
        }
    }

@app.post("/api/generate", response_model=GenerateResponse)
async def generate_ui(request: GenerateRequest):
    try:
        prompt = request.prompt.strip()
        if not prompt:
            raise HTTPException(status_code=400, detail="Prompt cannot be empty")

        print(f"📝 Received prompt: {prompt[:50]}...")

        # Step 1: Parse intent
        intent = intent_parser.parse(prompt)
        print(f"✅ Intent: {intent.ui_type}")
        
        # Step 2: Create plan
        plan = planner.create_plan(intent)
        print(f"✅ Plan: {len(plan.components)} components")
        
        # Step 3: Generate code
        code = code_generator.generate(plan)
        print(f"✅ Code: {len(code)} chars")
        
        # Step 4: Validate code
        validation_result = code_validator.validate(code)
        print(f"✅ Validation: {'PASS' if validation_result.is_valid else 'FAIL'}")
        
        if not validation_result.is_valid:
            code = code_validator.fix_common_issues(code)
            validation_result = code_validator.validate(code)

        response = GenerateResponse(
            code=code,
            explanation=plan.reasoning,
            plan={
                "layout": plan.layout_type,
                "components": [
                    {"type": c.type, "props": c.props}
                    for c in plan.components
                ],
                "imports": plan.imports
            },
            validation={
                "is_valid": validation_result.is_valid,
                "errors": validation_result.errors,
                "warnings": validation_result.warnings,
                "suggestions": validation_result.suggestions
            }
        )

        print("🎉 Generation complete!")
        return response

    except Exception as e:
        print(f"❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

def run_server():
    """Run the FastAPI server"""
    port = int(os.environ.get("PORT", 8000))
    
    print("=" * 50)
    print("🚀 Starting Rule-Based UI Generator")
    print("=" * 50)
    print(f"🌐 Port: {port}")
    print(f"📚 Pipeline: Rule-based (No LLM)")
    print(f"🔧 CORS: Enabled for all origins")
    print("=" * 50)
    
    config = uvicorn.Config(
        app,
        host="0.0.0.0",
        port=port,
        log_level="info",
        access_log=True
    )
    server = uvicorn.Server(config)
    
    import asyncio
    loop = asyncio.get_event_loop()
    loop.create_task(server.serve())
    
    print("✅ Server started!")
    print(f"📖 API Docs: http://0.0.0.0:{port}/docs")
    print(f"🏥 Health: http://0.0.0.0:{port}/health")
    print("=" * 50)
    
    return server

if __name__ == "__main__":
    server = run_server()
