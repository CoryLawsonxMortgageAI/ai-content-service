"""
Batch Processor - Handle batch generation jobs
"""

import asyncio
from typing import List, Dict, Any, Optional
from datetime import datetime
import json

from supabase import create_client, Client

from generators.description_generator import DescriptionGenerator
from generators.tag_generator import TagGenerator


class BatchProcessor:
    """Process batch generation jobs"""
    
    def __init__(
        self,
        description_generator: DescriptionGenerator,
        tag_generator: TagGenerator,
        supabase_url: Optional[str] = None,
        supabase_key: Optional[str] = None,
    ):
        self.description_generator = description_generator
        self.tag_generator = tag_generator
        
        self.supabase: Optional[Client] = None
        if supabase_url and supabase_key:
            self.supabase = create_client(supabase_url, supabase_key)
        
        # In-memory job storage (use Redis/DB in production)
        self.jobs: Dict[str, Dict[str, Any]] = {}
    
    async def process_batch(
        self,
        batch_id: str,
        tool_ids: List[str],
        generation_type: str,
        options: Dict[str, Any],
    ):
        """Process a batch of tools"""
        
        print(f"🔄 Starting batch {batch_id} with {len(tool_ids)} tools")
        
        # Initialize batch status
        self.jobs[batch_id] = {
            "batch_id": batch_id,
            "status": "processing",
            "total": len(tool_ids),
            "completed": 0,
            "failed": 0,
            "started_at": datetime.utcnow().isoformat(),
            "results": [],
        }
        
        # Process each tool
        semaphore = asyncio.Semaphore(3)  # Limit concurrent operations
        
        async def process_with_limit(tool_id: str):
            async with semaphore:
                return await self._process_single_tool(
                    batch_id,
                    tool_id,
                    generation_type,
                    options,
                )
        
        # Run all tasks
        tasks = [process_with_limit(tool_id) for tool_id in tool_ids]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Update final status
        completed = sum(1 for r in results if isinstance(r, dict) and r.get("success"))
        failed = len(results) - completed
        
        self.jobs[batch_id].update({
            "status": "completed",
            "completed": completed,
            "failed": failed,
            "completed_at": datetime.utcnow().isoformat(),
        })
        
        print(f"✅ Batch {batch_id} completed: {completed} succeeded, {failed} failed")
    
    async def _process_single_tool(
        self,
        batch_id: str,
        tool_id: str,
        generation_type: str,
        options: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Process a single tool in a batch"""
        
        job_id = f"{batch_id}_{tool_id}"
        
        try:
            # Fetch tool data
            tool_data = await self._fetch_tool_data(tool_id)
            
            if not tool_data:
                raise ValueError(f"Tool {tool_id} not found")
            
            # Create job record
            self.jobs[job_id] = {
                "job_id": job_id,
                "batch_id": batch_id,
                "tool_id": tool_id,
                "status": "processing",
                "started_at": datetime.utcnow().isoformat(),
            }
            
            result = None
            
            # Generate based on type
            if generation_type == "description":
                result = await self.description_generator.generate(
                    tool_name=tool_data["name"],
                    website_url=tool_data["website_url"],
                    existing_description=tool_data.get("description"),
                    tone=options.get("tone", "professional"),
                    max_length=options.get("max_length", 500),
                )
            
            elif generation_type == "tags":
                result = await self.tag_generator.generate(
                    tool_name=tool_data["name"],
                    description=tool_data.get("description", ""),
                    current_tags=tool_data.get("tags", []),
                    max_tags=options.get("max_tags", 10),
                )
            
            elif generation_type == "full_content":
                # Generate both description and tags
                desc_result = await self.description_generator.generate(
                    tool_name=tool_data["name"],
                    website_url=tool_data["website_url"],
                    existing_description=tool_data.get("description"),
                    tone=options.get("tone", "professional"),
                )
                
                tags_result = await self.tag_generator.generate(
                    tool_name=tool_data["name"],
                    description=desc_result.get("description", ""),
                )
                
                result = {
                    **desc_result,
                    "tags": tags_result.get("tags", []),
                    "tokens_used": desc_result.get("tokens_used", 0) + tags_result.get("tokens_used", 0),
                    "estimated_cost": desc_result.get("estimated_cost", 0) + tags_result.get("estimated_cost", 0),
                }
            
            elif generation_type == "category":
                result = await self.tag_generator.suggest_category(
                    tool_name=tool_data["name"],
                    description=tool_data.get("description", ""),
                )
            
            # Update database
            await self._update_tool_data(tool_id, generation_type, result)
            
            # Update job status
            self.jobs[job_id].update({
                "status": "completed",
                "completed_at": datetime.utcnow().isoformat(),
                "result": result,
            })
            
            # Update batch progress
            self.jobs[batch_id]["completed"] += 1
            self.jobs[batch_id]["results"].append({
                "tool_id": tool_id,
                "success": True,
            })
            
            return {"success": True, "tool_id": tool_id}
            
        except Exception as e:
            print(f"❌ Error processing tool {tool_id}: {e}")
            
            # Update job status
            self.jobs[job_id] = {
                "job_id": job_id,
                "batch_id": batch_id,
                "tool_id": tool_id,
                "status": "failed",
                "error": str(e),
                "completed_at": datetime.utcnow().isoformat(),
            }
            
            # Update batch progress
            self.jobs[batch_id]["failed"] += 1
            self.jobs[batch_id]["results"].append({
                "tool_id": tool_id,
                "success": False,
                "error": str(e),
            })
            
            return {"success": False, "tool_id": tool_id, "error": str(e)}
    
    async def _fetch_tool_data(self, tool_id: str) -> Optional[Dict[str, Any]]:
        """Fetch tool data from database"""
        
        if not self.supabase:
            # Return mock data for testing
            return {
                "id": tool_id,
                "name": "Sample AI Tool",
                "description": "A sample AI tool for testing.",
                "website_url": "https://example.com",
                "tags": ["ai", "testing"],
            }
        
        result = self.supabase.table("tools") \
            .select("*") \
            .eq("id", tool_id) \
            .single() \
            .execute()
        
        return result.data if result.data else None
    
    async def _update_tool_data(
        self,
        tool_id: str,
        generation_type: str,
        result: Dict[str, Any],
    ):
        """Update tool data in database"""
        
        if not self.supabase:
            return
        
        updates = {}
        
        if generation_type == "description":
            updates = {
                "ai_generated_description": result.get("description"),
                "short_description": result.get("short_description"),
                "content_generation_status": "completed",
            }
        elif generation_type == "tags":
            updates = {
                "ai_generated_tags": result.get("tags"),
            }
        elif generation_type == "full_content":
            updates = {
                "ai_generated_description": result.get("description"),
                "short_description": result.get("short_description"),
                "features": result.get("features"),
                "ai_generated_tags": result.get("tags"),
                "content_generation_status": "completed",
            }
        elif generation_type == "category":
            # Would need to map category slug to ID
            pass
        
        if updates:
            self.supabase.table("tools") \
                .update(updates) \
                .eq("id", tool_id) \
                .execute()
        
        # Record in AI generation jobs table
        self.supabase.table("ai_generation_jobs").insert({
            "job_type": generation_type,
            "tool_id": tool_id,
            "status": "completed",
            "output_data": result,
            "tokens_used": result.get("tokens_used"),
            "cost_estimate": result.get("estimated_cost"),
            "completed_at": datetime.utcnow().isoformat(),
        }).execute()
    
    async def get_job_status(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a job"""
        
        if job_id in self.jobs:
            job = self.jobs[job_id]
            
            # Calculate progress for batch jobs
            if "total" in job:
                total = job["total"]
                completed = job.get("completed", 0)
                failed = job.get("failed", 0)
                progress = (completed + failed) / total if total > 0 else 0
                
                return {
                    **job,
                    "progress": round(progress * 100, 2),
                }
            
            return job
        
        return None
    
    async def cancel_batch(self, batch_id: str) -> bool:
        """Cancel a running batch"""
        
        if batch_id in self.jobs:
            self.jobs[batch_id]["status"] = "cancelled"
            self.jobs[batch_id]["cancelled_at"] = datetime.utcnow().isoformat()
            return True
        
        return False
