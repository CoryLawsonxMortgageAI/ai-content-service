"""
Cost Tracker - Track AI API usage and costs
"""

import json
import time
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict
from datetime import datetime


@dataclass
class CostEntry:
    """Single cost tracking entry"""
    timestamp: float
    operation: str
    provider: str
    model: str
    tokens: int
    cost: float
    user_id: Optional[str] = None
    tool_id: Optional[str] = None


class CostTracker:
    """Track and report AI API costs"""
    
    # Pricing per 1K tokens (approximate)
    PRICING = {
        "openai": {
            "gpt-4": {"input": 0.03, "output": 0.06},
            "gpt-4-turbo": {"input": 0.01, "output": 0.03},
            "gpt-3.5-turbo": {"input": 0.0015, "output": 0.002},
        },
        "anthropic": {
            "claude-2": {"input": 0.008, "output": 0.024},
            "claude-instant": {"input": 0.00163, "output": 0.00551},
        },
    }
    
    def __init__(self, storage_path: Optional[str] = None):
        self.storage_path = storage_path
        self.entries: List[CostEntry] = []
        self._daily_cache: Dict[str, Dict[str, Any]] = {}
    
    async def track_cost(
        self,
        operation: str,
        provider: str,
        model: str,
        tokens: int,
        cost: Optional[float] = None,
        user_id: Optional[str] = None,
        tool_id: Optional[str] = None,
    ):
        """Track a cost entry"""
        
        # Calculate cost if not provided
        if cost is None:
            cost = self.calculate_cost(provider, model, tokens)
        
        entry = CostEntry(
            timestamp=time.time(),
            operation=operation,
            provider=provider,
            model=model,
            tokens=tokens,
            cost=cost,
            user_id=user_id,
            tool_id=tool_id,
        )
        
        self.entries.append(entry)
        
        # Persist if path provided
        if self.storage_path:
            await self._persist_entry(entry)
    
    def calculate_cost(self, provider: str, model: str, tokens: int, is_output: bool = False) -> float:
        """Calculate cost based on provider and model"""
        
        provider_pricing = self.PRICING.get(provider, {})
        model_pricing = provider_pricing.get(model, provider_pricing.get("default", {"input": 0, "output": 0}))
        
        rate_type = "output" if is_output else "input"
        rate = model_pricing.get(rate_type, 0)
        
        return (tokens / 1000) * rate
    
    def get_stats(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Get cost statistics"""
        
        # Filter by date if provided
        entries = self.entries
        if start_date or end_date:
            start_ts = datetime.fromisoformat(start_date).timestamp() if start_date else 0
            end_ts = datetime.fromisoformat(end_date).timestamp() if end_date else float('inf')
            entries = [e for e in entries if start_ts <= e.timestamp <= end_ts]
        
        if not entries:
            return {
                "total_cost": 0,
                "total_tokens": 0,
                "total_requests": 0,
                "by_provider": {},
                "by_operation": {},
                "by_day": {},
            }
        
        total_cost = sum(e.cost for e in entries)
        total_tokens = sum(e.tokens for e in entries)
        
        # Group by provider
        by_provider: Dict[str, Dict[str, Any]] = {}
        for e in entries:
            if e.provider not in by_provider:
                by_provider[e.provider] = {"cost": 0, "tokens": 0, "requests": 0}
            by_provider[e.provider]["cost"] += e.cost
            by_provider[e.provider]["tokens"] += e.tokens
            by_provider[e.provider]["requests"] += 1
        
        # Group by operation
        by_operation: Dict[str, Dict[str, Any]] = {}
        for e in entries:
            if e.operation not in by_operation:
                by_operation[e.operation] = {"cost": 0, "tokens": 0, "requests": 0}
            by_operation[e.operation]["cost"] += e.cost
            by_operation[e.operation]["tokens"] += e.tokens
            by_operation[e.operation]["requests"] += 1
        
        # Group by day
        by_day: Dict[str, Dict[str, Any]] = {}
        for e in entries:
            day = datetime.fromtimestamp(e.timestamp).strftime("%Y-%m-%d")
            if day not in by_day:
                by_day[day] = {"cost": 0, "tokens": 0, "requests": 0}
            by_day[day]["cost"] += e.cost
            by_day[day]["tokens"] += e.tokens
            by_day[day]["requests"] += 1
        
        return {
            "total_cost": round(total_cost, 4),
            "total_tokens": total_tokens,
            "total_requests": len(entries),
            "average_cost_per_request": round(total_cost / len(entries), 6),
            "average_tokens_per_request": round(total_tokens / len(entries), 2),
            "by_provider": {k: {**v, "cost": round(v["cost"], 4)} for k, v in by_provider.items()},
            "by_operation": {k: {**v, "cost": round(v["cost"], 4)} for k, v in by_operation.items()},
            "by_day": {k: {**v, "cost": round(v["cost"], 4)} for k, v in by_day.items()},
        }
    
    def get_daily_summary(self, days: int = 7) -> List[Dict[str, Any]]:
        """Get daily summary for the last N days"""
        
        now = time.time()
        cutoff = now - (days * 86400)
        
        recent_entries = [e for e in self.entries if e.timestamp >= cutoff]
        
        # Group by day
        daily: Dict[str, List[CostEntry]] = {}
        for e in recent_entries:
            day = datetime.fromtimestamp(e.timestamp).strftime("%Y-%m-%d")
            if day not in daily:
                daily[day] = []
            daily[day].append(e)
        
        # Build summary
        summary = []
        for day in sorted(daily.keys(), reverse=True):
            entries = daily[day]
            summary.append({
                "date": day,
                "total_cost": round(sum(e.cost for e in entries), 4),
                "total_tokens": sum(e.tokens for e in entries),
                "total_requests": len(entries),
            })
        
        return summary
    
    def get_budget_alert(self, daily_budget: float = 50.0) -> Optional[Dict[str, Any]]:
        """Check if approaching budget limit"""
        
        today = datetime.now().strftime("%Y-%m-%d")
        today_entries = [
            e for e in self.entries
            if datetime.fromtimestamp(e.timestamp).strftime("%Y-%m-%d") == today
        ]
        
        today_cost = sum(e.cost for e in today_entries)
        utilization = today_cost / daily_budget
        
        if utilization >= 0.9:
            return {
                "alert": True,
                "severity": "critical" if utilization >= 1.0 else "warning",
                "today_cost": round(today_cost, 4),
                "daily_budget": daily_budget,
                "utilization_percent": round(utilization * 100, 2),
                "message": f"Daily budget {'exceeded' if utilization >= 1.0 else 'at risk'}: ${today_cost:.2f} / ${daily_budget:.2f}",
            }
        
        return {
            "alert": False,
            "today_cost": round(today_cost, 4),
            "daily_budget": daily_budget,
            "utilization_percent": round(utilization * 100, 2),
        }
    
    async def _persist_entry(self, entry: CostEntry):
        """Persist entry to storage"""
        if not self.storage_path:
            return
        
        try:
            with open(self.storage_path, "a") as f:
                f.write(json.dumps(asdict(entry)) + "\n")
        except Exception as e:
            print(f"❌ Failed to persist cost entry: {e}")
    
    def load_from_file(self, filepath: str):
        """Load cost entries from file"""
        try:
            with open(filepath, "r") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        data = json.loads(line)
                        self.entries.append(CostEntry(**data))
            print(f"✅ Loaded {len(self.entries)} cost entries from {filepath}")
        except FileNotFoundError:
            print(f"⚠️ Cost file not found: {filepath}")
        except Exception as e:
            print(f"❌ Error loading cost entries: {e}")


# Simple file-based persistence
class FileCostTracker(CostTracker):
    """Cost tracker with automatic file persistence"""
    
    def __init__(self, storage_path: str = "costs.jsonl"):
        super().__init__(storage_path)
        self.load_from_file(storage_path)
