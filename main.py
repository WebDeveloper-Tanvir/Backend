"""
FastAPI Backend - Rule-based UI Generator
CORS-enabled for Railway deployment
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
    description="Generate React UIs without LLM",
    version="1.0.0"
)

# ===== CRITICAL: CORS CONFIGURATION =====
# This MUST come before any routes
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# =======================================

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
    """Root endpoint"""
    return {
        "message": "Rule-Based UI Generator API",
        "version": "1.0.0",
        "status": "online",
        "cors": "enabled"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
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
    """Generate UI from prompt"""
    try:
        prompt = request.prompt.strip()
        if not prompt:
            raise HTTPException(status_code=400, detail="Prompt cannot be empty")

        print(f"📝 Processing: {prompt[:50]}...")

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
        print(f"✅ Valid: {validation_result.is_valid}")
        
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

        print("🎉 Success!")
        return response

    except Exception as e:
        print(f"❌ Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# This is for Railway - it will run automatically
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    print(f"🚀 Starting on port {port}")
    print("🔓 CORS: Enabled for all origins")
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port,
        log_level="info"
    )
