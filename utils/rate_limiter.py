"""
Rate Limiter - Manage API rate limits for AI services
"""

import asyncio
import time
from typing import Dict, Tuple, Optional
from dataclasses import dataclass


@dataclass
class RateLimitBucket:
    """Token bucket for rate limiting"""
    tokens: float
    last_update: float
    rpm_limit: int
    tpm_limit: int
    current_minute: int
    tokens_this_minute: int


class RateLimiter:
    """Rate limiter for OpenAI and other AI APIs"""
    
    def __init__(
        self,
        rpm: int = 60,  # Requests per minute
        tpm: int = 60000,  # Tokens per minute
        tpd: int = 1000000,  # Tokens per day
    ):
        self.rpm = rpm
        self.tpm = tpm
        self.tpd = tpd
        
        # In-memory storage (use Redis in production)
        self.buckets: Dict[str, RateLimitBucket] = {}
        self.daily_tokens: Dict[str, Dict[str, int]] = {}  # user_id -> {date: tokens}
        
        # Lock for thread safety
        self._lock = asyncio.Lock()
    
    async def check_limit(self, user_id: str = "default") -> Tuple[bool, int]:
        """
        Check if request is within rate limits.
        Returns (allowed, retry_after_seconds)
        """
        async with self._lock:
            now = time.time()
            current_minute = int(now // 60)
            today = time.strftime("%Y-%m-%d")
            
            # Get or create bucket
            if user_id not in self.buckets:
                self.buckets[user_id] = RateLimitBucket(
                    tokens=self.rpm,
                    last_update=now,
                    rpm_limit=self.rpm,
                    tpm_limit=self.tpm,
                    current_minute=current_minute,
                    tokens_this_minute=0,
                )
            
            bucket = self.buckets[user_id]
            
            # Reset minute counter if new minute
            if bucket.current_minute != current_minute:
                bucket.current_minute = current_minute
                bucket.tokens_this_minute = 0
                bucket.tokens = self.rpm
            
            # Check daily limit
            if user_id not in self.daily_tokens:
                self.daily_tokens[user_id] = {}
            
            daily_usage = self.daily_tokens[user_id].get(today, 0)
            if daily_usage >= self.tpd:
                retry_after = 86400 - (now % 86400)  # Seconds until midnight
                return False, int(retry_after)
            
            # Check token bucket
            time_passed = now - bucket.last_update
            bucket.tokens = min(
                self.rpm,
                bucket.tokens + time_passed * (self.rpm / 60)
            )
            bucket.last_update = now
            
            if bucket.tokens < 1:
                retry_after = int(60 / self.rpm) + 1
                return False, retry_after
            
            # Consume token
            bucket.tokens -= 1
            
            return True, 0
    
    async def record_tokens(self, user_id: str, tokens: int):
        """Record token usage for daily limit tracking"""
        async with self._lock:
            today = time.strftime("%Y-%m-%d")
            
            if user_id not in self.daily_tokens:
                self.daily_tokens[user_id] = {}
            
            self.daily_tokens[user_id][today] = self.daily_tokens[user_id].get(today, 0) + tokens
            
            # Update bucket
            if user_id in self.buckets:
                self.buckets[user_id].tokens_this_minute += tokens
    
    async def get_remaining(self, user_id: str = "default") -> Dict[str, any]:
        """Get remaining quota for a user"""
        async with self._lock:
            now = time.time()
            current_minute = int(now // 60)
            today = time.strftime("%Y-%m-%d")
            
            if user_id not in self.buckets:
                return {
                    "rpm_remaining": self.rpm,
                    "tpm_remaining": self.tpm,
                    "tpd_remaining": self.tpd,
                }
            
            bucket = self.buckets[user_id]
            
            # Recalculate tokens
            time_passed = now - bucket.last_update
            rpm_remaining = min(
                self.rpm,
                bucket.tokens + time_passed * (self.rpm / 60)
            )
            
            daily_usage = self.daily_tokens.get(user_id, {}).get(today, 0)
            tpd_remaining = max(0, self.tpd - daily_usage)
            
            return {
                "rpm_remaining": int(rpm_remaining),
                "tpm_remaining": self.tpm - bucket.tokens_this_minute,
                "tpd_remaining": tpd_remaining,
                "resets_at": (bucket.current_minute + 1) * 60,
            }
    
    def get_stats(self) -> Dict[str, any]:
        """Get rate limiter statistics"""
        today = time.strftime("%Y-%m-%d")
        total_daily_usage = sum(
            user_data.get(today, 0)
            for user_data in self.daily_tokens.values()
        )
        
        return {
            "active_buckets": len(self.buckets),
            "unique_users": len(self.daily_tokens),
            "total_tokens_today": total_daily_usage,
            "daily_limit": self.tpd,
            "utilization_percent": round((total_daily_usage / self.tpd) * 100, 2),
        }


# Redis-based rate limiter for production
class RedisRateLimiter(RateLimiter):
    """Redis-backed rate limiter for distributed systems"""
    
    def __init__(
        self,
        redis_url: str,
        rpm: int = 60,
        tpm: int = 60000,
        tpd: int = 1000000,
    ):
        super().__init__(rpm, tpm, tpd)
        
        try:
            import redis.asyncio as redis
            self.redis = redis.from_url(redis_url)
            self.use_redis = True
        except ImportError:
            print("⚠️ Redis not available, using in-memory storage")
            self.use_redis = False
    
    async def check_limit(self, user_id: str = "default") -> Tuple[bool, int]:
        """Check rate limit using Redis"""
        if not self.use_redis:
            return await super().check_limit(user_id)
        
        now = time.time()
        minute_key = f"ratelimit:{user_id}:minute"
        day_key = f"ratelimit:{user_id}:day:{time.strftime('%Y-%m-%d')}"
        
        # Use Redis transactions for atomic operations
        pipe = self.redis.pipeline()
        
        # Check minute limit
        pipe.get(minute_key)
        pipe.ttl(minute_key)
        
        # Check daily limit
        pipe.get(day_key)
        
        results = await pipe.execute()
        minute_count = int(results[0] or 0)
        minute_ttl = results[1]
        daily_count = int(results[2] or 0)
        
        # Check limits
        if minute_count >= self.rpm:
            retry_after = max(1, minute_ttl)
            return False, retry_after
        
        if daily_count >= self.tpd:
            retry_after = 86400 - (int(now) % 86400)
            return False, retry_after
        
        # Increment counters
        pipe = self.redis.pipeline()
        pipe.incr(minute_key)
        pipe.expire(minute_key, 60)
        pipe.incr(day_key)
        pipe.expire(day_key, 86400)
        await pipe.execute()
        
        return True, 0
    
    async def record_tokens(self, user_id: str, tokens: int):
        """Record token usage in Redis"""
        if not self.use_redis:
            await super().record_tokens(user_id, tokens)
            return
        
        key = f"ratelimit:{user_id}:tokens:{time.strftime('%Y-%m-%d')}"
        await self.redis.incrby(key, tokens)
        await self.redis.expire(key, 86400)
