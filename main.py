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

# Import your modules
from intent_parser import IntentParser
from planner import Planner
from code_generator import CodeGenerator
from code_validator import CodeValidator

# Initialize FastAPI app
app = FastAPI(
    title="Rule-Based UI Generator",
    description="Generate React UIs without LLM",
    version="1.0.0"
)

# Add CORS middleware
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
        "endpoints": {
            "health": "/health",
            "generate": "/api/generate"
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

        # Step 1: Parse intent
        intent = intent_parser.parse(prompt)
        
        # Step 2: Create plan
        plan = planner.create_plan(intent)
        
        # Step 3: Generate code
        code = code_generator.generate(plan)
        
        # Step 4: Validate code
        validation_result = code_validator.validate(code)
        
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

        return response

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def run_server():
    port = int(os.environ.get("PORT", 8000))
    print(f"🚀 Starting on port {port}...")
    
    config = uvicorn.Config(
        app,
        host="0.0.0.0",
        port=port,
        log_level="info"
    )
    server = uvicorn.Server(config)
    
    import asyncio
    loop = asyncio.get_event_loop()
    loop.create_task(server.serve())
    
    return server

if __name__ == "__main__":
    server = run_server()
