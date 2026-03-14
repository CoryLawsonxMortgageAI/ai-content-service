"""
Tag Generator - Generate AI-powered tags for tools
"""

import json
from typing import Optional, Dict, Any, List

import openai
from tenacity import retry, stop_after_attempt, wait_exponential

from utils.cost_tracker import CostTracker
from utils.rate_limiter import RateLimiter


class TagGenerator:
    """Generate relevant tags for AI tools"""
    
    # Predefined tag categories
    TAG_CATEGORIES = {
        "technology": [
            "ai", "machine-learning", "deep-learning", "neural-networks",
            "nlp", "computer-vision", "generative-ai", "llm",
            "automation", "api", "saas", "cloud",
        ],
        "function": [
            "generator", "assistant", "analyzer", "optimizer",
            "automation", "chatbot", "writer", "designer",
            "transcriber", "translator", "summarizer", "classifier",
        ],
        "content_type": [
            "text", "image", "video", "audio", "voice",
            "code", "data", "document", "presentation",
        ],
        "use_case": [
            "marketing", "seo", "writing", "design", "development",
            "business", "productivity", "research", "education",
            "customer-support", "sales", "analytics",
        ],
        "pricing": [
            "free", "freemium", "paid", "enterprise", "open-source",
        ],
    }
    
    def __init__(
        self,
        openai_api_key: Optional[str] = None,
        rate_limiter: Optional[RateLimiter] = None,
        cost_tracker: Optional[CostTracker] = None,
    ):
        self.openai_api_key = openai_api_key
        self.rate_limiter = rate_limiter
        self.cost_tracker = cost_tracker
        
        if openai_api_key:
            self.openai_client = openai.AsyncOpenAI(api_key=openai_api_key)
        else:
            self.openai_client = None
    
    async def generate(
        self,
        tool_name: str,
        description: str,
        current_tags: Optional[List[str]] = None,
        max_tags: int = 10,
    ) -> Dict[str, Any]:
        """Generate tags for a tool"""
        
        if not self.openai_client:
            raise ValueError("OpenAI client not initialized")
        
        # Combine rule-based and AI-generated tags
        rule_based_tags = self._generate_rule_based_tags(tool_name, description)
        
        # Get AI-generated tags
        ai_tags = await self._generate_ai_tags(
            tool_name,
            description,
            current_tags or [],
            max_tags - len(rule_based_tags),
        )
        
        # Combine and deduplicate
        all_tags = list(dict.fromkeys(rule_based_tags + ai_tags))
        
        # Sort by relevance (AI tags first, then rule-based)
        all_tags = all_tags[:max_tags]
        
        # Calculate metadata
        tokens_used = 200  # Approximate
        estimated_cost = (tokens_used / 1000) * 0.002  # GPT-3.5 rate
        
        # Track costs
        if self.cost_tracker:
            await self.cost_tracker.track_cost(
                operation="tag_generation",
                provider="openai",
                model="gpt-3.5-turbo",
                tokens=tokens_used,
                cost=estimated_cost,
            )
        
        return {
            "tags": all_tags,
            "rule_based_tags": rule_based_tags,
            "ai_suggested_tags": ai_tags,
            "tokens_used": tokens_used,
            "estimated_cost": round(estimated_cost, 6),
            "model": "gpt-3.5-turbo",
        }
    
    def _generate_rule_based_tags(self, tool_name: str, description: str) -> List[str]:
        """Generate tags based on rules and keyword matching"""
        
        text = f"{tool_name} {description}".lower()
        tags = set()
        
        # Match against predefined categories
        for category, keywords in self.TAG_CATEGORIES.items():
            for keyword in keywords:
                if keyword.replace('-', ' ') in text or keyword in text:
                    tags.add(keyword)
        
        # Add specific patterns
        patterns = {
            "chatbot": ["chat", "chatbot", "conversation", "messaging"],
            "writing": ["write", "writing", "content", "copywriting"],
            "image-generation": ["image", "generate image", "ai art", "image creation"],
            "code-assistant": ["code", "coding", "developer", "programming"],
            "voice": ["voice", "speech", "audio", "sound"],
            "video": ["video", "video editing", "animation"],
            "automation": ["automation", "automate", "workflow"],
            "analytics": ["analytics", "analysis", "insights", "metrics"],
            "seo": ["seo", "search engine", "ranking"],
            "marketing": ["marketing", "advertising", "campaign"],
        }
        
        for tag, keywords in patterns.items():
            if any(kw in text for kw in keywords):
                tags.add(tag)
        
        return sorted(list(tags))[:5]
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def _generate_ai_tags(
        self,
        tool_name: str,
        description: str,
        current_tags: List[str],
        max_tags: int,
    ) -> List[str]:
        """Generate tags using AI"""
        
        current_tags_text = ", ".join(current_tags) if current_tags else "None"
        
        prompt = f"""Given the following AI tool, suggest relevant tags for categorization.

Tool Name: {tool_name}
Description: {description}
Current Tags: {current_tags_text}

Available tag categories:
- Technology: ai, machine-learning, deep-learning, nlp, computer-vision, generative-ai
- Function: generator, assistant, analyzer, optimizer, automation, chatbot, writer
- Content Type: text, image, video, audio, code, data
- Use Case: marketing, seo, writing, design, development, business, productivity
- Pricing: free, freemium, paid, enterprise, open-source

Generate {max_tags} relevant tags that:
1. Are lowercase and hyphenated (e.g., "machine-learning")
2. Cover the tool's primary functionality
3. Include relevant industry/sector tags
4. Match common search terms users might use

Respond with ONLY a JSON array of strings:
["tag1", "tag2", "tag3", ...]"""
        
        try:
            response = await self.openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a taxonomy expert. Suggest relevant tags. Respond with valid JSON only."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                max_tokens=200,
                temperature=0.3,
                response_format={"type": "json_object"}
            )
            
            content = response.choices[0].message.content
            result = json.loads(content)
            
            # Handle different response formats
            if isinstance(result, list):
                return result[:max_tags]
            elif isinstance(result, dict):
                # Look for tags in common keys
                for key in ["tags", "suggested_tags", "result"]:
                    if key in result and isinstance(result[key], list):
                        return result[key][:max_tags]
            
            return []
            
        except json.JSONDecodeError:
            print("❌ Failed to parse tag generation response")
            return []
        except Exception as e:
            print(f"❌ Error generating AI tags: {e}")
            return []
    
    async def suggest_category(
        self,
        tool_name: str,
        description: str,
    ) -> Dict[str, Any]:
        """Suggest the best category for a tool"""
        
        categories = [
            "text-writing",
            "image-design",
            "audio-voice",
            "video-animation",
            "code-development",
            "business-productivity",
            "marketing-seo",
            "chatbots-assistants",
            "data-analytics",
            "education-learning",
        ]
        
        prompt = f"""Categorize this AI tool into the most appropriate category.

Tool Name: {tool_name}
Description: {description}

Available Categories:
{chr(10).join(f"- {c}" for c in categories)}

Respond with JSON:
{{
    "primary_category": "category-slug",
    "confidence": 0.95,
    "reasoning": "Brief explanation",
    "secondary_categories": ["category1", "category2"]
}}"""
        
        try:
            response = await self.openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a categorization expert."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                max_tokens=300,
                temperature=0.2,
                response_format={"type": "json_object"}
            )
            
            content = response.choices[0].message.content
            return json.loads(content)
            
        except Exception as e:
            print(f"❌ Error suggesting category: {e}")
            return {
                "primary_category": "business-productivity",
                "confidence": 0.5,
                "reasoning": "Default category",
                "secondary_categories": [],
            }
    
    async def extract_keywords(self, description: str, max_keywords: int = 10) -> List[str]:
        """Extract SEO keywords from description"""
        
        prompt = f"""Extract SEO keywords from this tool description.

Description: {description}

Extract up to {max_keywords} relevant keywords that:
1. Are commonly searched terms
2. Represent the tool's functionality
3. Include both short and long-tail keywords

Respond with JSON array: ["keyword1", "keyword2", ...]"""
        
        try:
            response = await self.openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {
                        "role": "system",
                        "content": "You are an SEO expert."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                max_tokens=200,
                temperature=0.3,
                response_format={"type": "json_object"}
            )
            
            content = response.choices[0].message.content
            result = json.loads(content)
            
            if isinstance(result, list):
                return result
            elif isinstance(result, dict):
                return result.get("keywords", [])
            
            return []
            
        except Exception as e:
            print(f"❌ Error extracting keywords: {e}")
            return []
