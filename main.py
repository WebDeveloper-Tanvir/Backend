"""
FastAPI Backend - Rule-based UI Generator
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import os

# Import local modules
from intent_parser import IntentParser
from planner import Planner
from code_generator import CodeGenerator
from code_validator import CodeValidator

# Initialize FastAPI app
app = FastAPI(
    title="Rule-Based UI Generator",
    version="1.0.0"
)

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize components
intent_parser = IntentParser()
planner = Planner()
code_generator = CodeGenerator()
code_validator = CodeValidator()

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
        "status": "online"
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "pipeline": "rule-based"
    }

@app.post("/api/generate", response_model=GenerateResponse)
async def generate_ui(request: GenerateRequest):
    try:
        prompt = request.prompt.strip()
        if not prompt:
            raise HTTPException(status_code=400, detail="Prompt required")

        intent = intent_parser.parse(prompt)
        plan = planner.create_plan(intent)
        code = code_generator.generate(plan)
        validation_result = code_validator.validate(code)
        
        if not validation_result.is_valid:
            code = code_validator.fix_common_issues(code)
            validation_result = code_validator.validate(code)

        return GenerateResponse(
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

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    
    # ✅ FIX: Use Railway's PORT environment variable
    port = int(os.environ.get("PORT", 8000))
    print(f"🚀 Starting on port {port}")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port,  # ✅ Dynamic port from Railway
        log_level="info"
    )
