"""
AI Content Generator Service
Generates AI-powered content for tool descriptions, tags, and metadata
"""

import os
import asyncio
from typing import Optional, List, Dict, Any
from datetime import datetime
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from dotenv import load_dotenv
import uvicorn

from generators.description_generator import DescriptionGenerator
from generators.tag_generator import TagGenerator
from processors.batch_processor import BatchProcessor
from utils.rate_limiter import RateLimiter
from utils.cost_tracker import CostTracker

# Load environment variables
load_dotenv()

# Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
SERVICE_API_KEY = os.getenv("AI_SERVICE_API_KEY", "dev-key")
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")

# Pydantic models
class GenerateDescriptionRequest(BaseModel):
    tool_id: str = Field(..., description="Tool UUID")
    tool_name: str = Field(..., min_length=1, max_length=255)
    website_url: str = Field(..., description="Tool website URL")
    existing_description: Optional[str] = Field(None, max_length=5000)
    tone: str = Field(default="professional", pattern="^(professional|casual|technical|marketing)$")
    max_length: int = Field(default=500, ge=100, le=2000)
    include_features: bool = Field(default=True)
    include_use_cases: bool = Field(default=True)


class GenerateTagsRequest(BaseModel):
    tool_id: str = Field(..., description="Tool UUID")
    tool_name: str = Field(..., min_length=1)
    description: str = Field(..., min_length=10)
    current_tags: Optional[List[str]] = Field(default_factory=list)
    max_tags: int = Field(default=10, ge=1, le=20)


class BatchGenerateRequest(BaseModel):
    tool_ids: List[str] = Field(..., min_length=1, max_length=20)
    generation_type: str = Field(..., pattern="^(description|tags|category|full_content)$")
    options: Optional[Dict[str, Any]] = Field(default_factory=dict)


class GenerationResponse(BaseModel):
    success: bool
    data: Optional[Dict[str, Any]] = None
    job_id: Optional[str] = None
    tokens_used: Optional[int] = None
    estimated_cost: Optional[float] = None
    processing_time_ms: Optional[int] = None


class BatchResponse(BaseModel):
    success: bool
    batch_id: str
    message: str
    total_jobs: int
    estimated_completion: Optional[datetime] = None


class JobStatusResponse(BaseModel):
    job_id: str
    status: str
    progress: Optional[float] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


class HealthResponse(BaseModel):
    status: str
    timestamp: datetime
    version: str = "1.0.0"
    services: Dict[str, str]


# API Key dependency
async def verify_api_key(x_service_api_key: str = Header(...)):
    if x_service_api_key != SERVICE_API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return x_service_api_key


# Global service instances
description_generator: Optional[DescriptionGenerator] = None
tag_generator: Optional[TagGenerator] = None
batch_processor: Optional[BatchProcessor] = None
rate_limiter: Optional[RateLimiter] = None
cost_tracker: Optional[CostTracker] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan events"""
    global description_generator, tag_generator, batch_processor, rate_limiter, cost_tracker
    
    print("🚀 AI Content Generator Service starting...")
    
    # Initialize services
    rate_limiter = RateLimiter(
        rpm=int(os.getenv("RATE_LIMIT_RPM", "60")),
        tpd=int(os.getenv("RATE_LIMIT_TPD", "100000")),
    )
    
    cost_tracker = CostTracker()
    
    description_generator = DescriptionGenerator(
        openai_api_key=OPENAI_API_KEY,
        anthropic_api_key=ANTHROPIC_API_KEY,
        rate_limiter=rate_limiter,
        cost_tracker=cost_tracker,
    )
    
    tag_generator = TagGenerator(
        openai_api_key=OPENAI_API_KEY,
        rate_limiter=rate_limiter,
        cost_tracker=cost_tracker,
    )
    
    batch_processor = BatchProcessor(
        description_generator=description_generator,
        tag_generator=tag_generator,
        supabase_url=SUPABASE_URL,
        supabase_key=SUPABASE_KEY,
    )
    
    print("✅ All services initialized")
    
    yield
    
    # Cleanup
    print("🛑 Shutting down AI Content Generator Service...")


# Create FastAPI app
app = FastAPI(
    title="AI Content Generator Service",
    description="Generates AI-powered content for AI Tools Directory",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", response_model=dict)
async def root():
    """Root endpoint"""
    return {
        "service": "AI Content Generator Service",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    services = {
        "openai": "available" if OPENAI_API_KEY else "not_configured",
        "anthropic": "available" if ANTHROPIC_API_KEY else "not_configured",
        "supabase": "available" if SUPABASE_URL else "not_configured",
    }
    
    return HealthResponse(
        status="healthy",
        timestamp=datetime.utcnow(),
        services=services,
    )


@app.post("/api/v1/generate/description", response_model=GenerationResponse)
async def generate_description(
    request: GenerateDescriptionRequest,
    background_tasks: BackgroundTasks,
    api_key: str = Depends(verify_api_key),
):
    """
    Generate AI-powered description for a tool.
    If tool_id is provided, automatically updates the database.
    """
    if not description_generator:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    start_time = datetime.utcnow()
    
    try:
        # Check rate limit
        allowed, retry_after = await rate_limiter.check_limit("description")
        if not allowed:
            raise HTTPException(
                status_code=429,
                detail=f"Rate limit exceeded. Retry after {retry_after} seconds",
            )
        
        # Generate description
        result = await description_generator.generate(
            tool_name=request.tool_name,
            website_url=request.website_url,
            existing_description=request.existing_description,
            tone=request.tone,
            max_length=request.max_length,
            include_features=request.include_features,
            include_use_cases=request.include_use_cases,
        )
        
        processing_time = int((datetime.utcnow() - start_time).total_seconds() * 1000)
        
        # Update database if tool_id provided (background task)
        if request.tool_id and SUPABASE_URL:
            background_tasks.add_task(
                update_tool_description,
                request.tool_id,
                result,
            )
        
        return GenerationResponse(
            success=True,
            data=result,
            tokens_used=result.get("tokens_used"),
            estimated_cost=result.get("estimated_cost"),
            processing_time_ms=processing_time,
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/generate/tags", response_model=GenerationResponse)
async def generate_tags(
    request: GenerateTagsRequest,
    background_tasks: BackgroundTasks,
    api_key: str = Depends(verify_api_key),
):
    """Generate AI-powered tags for a tool"""
    if not tag_generator:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    start_time = datetime.utcnow()
    
    try:
        # Check rate limit
        allowed, retry_after = await rate_limiter.check_limit("tags")
        if not allowed:
            raise HTTPException(
                status_code=429,
                detail=f"Rate limit exceeded. Retry after {retry_after} seconds",
            )
        
        # Generate tags
        result = await tag_generator.generate(
            tool_name=request.tool_name,
            description=request.description,
            current_tags=request.current_tags,
            max_tags=request.max_tags,
        )
        
        processing_time = int((datetime.utcnow() - start_time).total_seconds() * 1000)
        
        # Update database if tool_id provided
        if request.tool_id and SUPABASE_URL:
            background_tasks.add_task(
                update_tool_tags,
                request.tool_id,
                result.get("tags", []),
            )
        
        return GenerationResponse(
            success=True,
            data=result,
            tokens_used=result.get("tokens_used"),
            estimated_cost=result.get("estimated_cost"),
            processing_time_ms=processing_time,
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/generate/full-content", response_model=GenerationResponse)
async def generate_full_content(
    request: GenerateDescriptionRequest,
    background_tasks: BackgroundTasks,
    api_key: str = Depends(verify_api_key),
):
    """
    Generate complete content package: description, tags, features, and use cases.
    """
    if not description_generator or not tag_generator:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    start_time = datetime.utcnow()
    
    try:
        # Check rate limit
        allowed, retry_after = await rate_limiter.check_limit("full_content")
        if not allowed:
            raise HTTPException(
                status_code=429,
                detail=f"Rate limit exceeded. Retry after {retry_after} seconds",
            )
        
        # Generate description
        desc_result = await description_generator.generate(
            tool_name=request.tool_name,
            website_url=request.website_url,
            existing_description=request.existing_description,
            tone=request.tone,
            max_length=request.max_length,
            include_features=True,
            include_use_cases=True,
        )
        
        # Generate tags based on the new description
        tags_result = await tag_generator.generate(
            tool_name=request.tool_name,
            description=desc_result.get("description", ""),
            max_tags=10,
        )
        
        # Combine results
        result = {
            "description": desc_result.get("description"),
            "short_description": desc_result.get("short_description"),
            "features": desc_result.get("features", []),
            "use_cases": desc_result.get("use_cases", []),
            "tags": tags_result.get("tags", []),
            "target_audience": desc_result.get("target_audience"),
            "value_proposition": desc_result.get("value_proposition"),
            "tokens_used": (desc_result.get("tokens_used", 0) + tags_result.get("tokens_used", 0)),
            "estimated_cost": (desc_result.get("estimated_cost", 0) + tags_result.get("estimated_cost", 0)),
        }
        
        processing_time = int((datetime.utcnow() - start_time).total_seconds() * 1000)
        
        # Update database if tool_id provided
        if request.tool_id and SUPABASE_URL:
            background_tasks.add_task(
                update_tool_full_content,
                request.tool_id,
                result,
            )
        
        return GenerationResponse(
            success=True,
            data=result,
            tokens_used=result.get("tokens_used"),
            estimated_cost=result.get("estimated_cost"),
            processing_time_ms=processing_time,
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/generate/batch", response_model=BatchResponse)
async def generate_batch(
    request: BatchGenerateRequest,
    background_tasks: BackgroundTasks,
    api_key: str = Depends(verify_api_key),
):
    """
    Process multiple tools in batch.
    Returns immediately with batch ID; processing happens in background.
    """
    if not batch_processor:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    batch_id = f"batch_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{len(request.tool_ids)}"
    
    # Start batch processing in background
    background_tasks.add_task(
        batch_processor.process_batch,
        batch_id,
        request.tool_ids,
        request.generation_type,
        request.options,
    )
    
    estimated_seconds = len(request.tool_ids) * 15  # Rough estimate
    estimated_completion = datetime.utcnow()
    estimated_completion = estimated_completion.replace(second=estimated_completion.second + estimated_seconds)
    
    return BatchResponse(
        success=True,
        batch_id=batch_id,
        message=f"Batch processing started for {len(request.tool_ids)} tools",
        total_jobs=len(request.tool_ids),
        estimated_completion=estimated_completion,
    )


@app.get("/api/v1/jobs/{job_id}/status", response_model=JobStatusResponse)
async def get_job_status(
    job_id: str,
    api_key: str = Depends(verify_api_key),
):
    """Get status of a generation job"""
    if not batch_processor:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    status = await batch_processor.get_job_status(job_id)
    
    if not status:
        raise HTTPException(status_code=404, detail="Job not found")
    
    return JobStatusResponse(
        job_id=job_id,
        status=status.get("status", "unknown"),
        progress=status.get("progress"),
        result=status.get("result"),
        error=status.get("error"),
    )


@app.get("/api/v1/stats")
async def get_stats(
    api_key: str = Depends(verify_api_key),
):
    """Get service statistics"""
    if not cost_tracker:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    return {
        "costs": cost_tracker.get_stats(),
        "rate_limits": rate_limiter.get_stats() if rate_limiter else {},
    }


# Background task functions
async def update_tool_description(tool_id: str, result: Dict[str, Any]):
    """Update tool description in database"""
    try:
        from supabase import create_client
        
        if not SUPABASE_URL or not SUPABASE_KEY:
            return
        
        supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
        
        supabase.table("tools").update({
            "ai_generated_description": result.get("description"),
            "short_description": result.get("short_description"),
            "features": result.get("features"),
            "content_generation_status": "completed",
            "updated_at": datetime.utcnow().isoformat(),
        }).eq("id", tool_id).execute()
        
        print(f"✅ Updated tool {tool_id} with generated description")
        
    except Exception as e:
        print(f"❌ Failed to update tool {tool_id}: {e}")


async def update_tool_tags(tool_id: str, tags: List[str]):
    """Update tool tags in database"""
    try:
        from supabase import create_client
        
        if not SUPABASE_URL or not SUPABASE_KEY:
            return
        
        supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
        
        supabase.table("tools").update({
            "ai_generated_tags": tags,
            "updated_at": datetime.utcnow().isoformat(),
        }).eq("id", tool_id).execute()
        
        print(f"✅ Updated tool {tool_id} with generated tags")
        
    except Exception as e:
        print(f"❌ Failed to update tool tags {tool_id}: {e}")


async def update_tool_full_content(tool_id: str, result: Dict[str, Any]):
    """Update tool with full generated content"""
    try:
        from supabase import create_client
        
        if not SUPABASE_URL or not SUPABASE_KEY:
            return
        
        supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
        
        supabase.table("tools").update({
            "ai_generated_description": result.get("description"),
            "short_description": result.get("short_description"),
            "features": result.get("features"),
            "tags": result.get("tags"),
            "content_generation_status": "completed",
            "updated_at": datetime.utcnow().isoformat(),
        }).eq("id", tool_id).execute()
        
        # Create analytics event
        supabase.table("ai_generation_jobs").insert({
            "job_type": "full_content",
            "tool_id": tool_id,
            "status": "completed",
            "output_data": result,
            "tokens_used": result.get("tokens_used"),
            "cost_estimate": result.get("estimated_cost"),
            "completed_at": datetime.utcnow().isoformat(),
        }).execute()
        
        print(f"✅ Updated tool {tool_id} with full generated content")
        
    except Exception as e:
        print(f"❌ Failed to update tool full content {tool_id}: {e}")


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8001, reload=True)
