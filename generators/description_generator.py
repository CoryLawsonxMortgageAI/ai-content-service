"""
Description Generator - Generate AI-powered tool descriptions
"""

import json
from typing import Optional, Dict, Any, List
from string import Template

import openai
from tenacity import retry, stop_after_attempt, wait_exponential

from utils.cost_tracker import CostTracker
from utils.rate_limiter import RateLimiter


class DescriptionGenerator:
    """Generate AI-powered descriptions for AI tools"""
    
    # Prompt templates
    DESCRIPTION_PROMPT = Template("""You are an expert AI tools directory curator. Write a compelling, SEO-optimized description for the following AI tool.

Tool Name: $tool_name
Website: $website_url
Tone: $tone
Maximum Length: $max_length characters

$existing_description

Requirements:
1. Write a comprehensive description ($min_desc_length-$max_length characters)
2. Include key features and benefits
3. Highlight what makes this tool unique
4. Target audience: professionals looking for AI solutions
5. Avoid buzzwords and be specific about capabilities
6. Use $tone tone throughout

Generate the following:
1. Main description (comprehensive, SEO-friendly)
2. Short description (one sentence, under 120 characters)
3. Key features (5-7 bullet points)
4. Target audience description
5. Primary use cases (3-4 examples)
6. Value proposition (what makes it unique)

Respond ONLY with valid JSON in this exact format:
{
    "description": "Main description text...",
    "short_description": "One sentence summary",
    "features": ["Feature 1", "Feature 2", "Feature 3", "Feature 4", "Feature 5"],
    "target_audience": "Description of ideal users",
    "use_cases": ["Use case 1", "Use case 2", "Use case 3"],
    "value_proposition": "What makes this tool unique"
}""")

    # Tone modifiers
    TONE_MODIFIERS = {
        "professional": "professional, business-focused, authoritative",
        "casual": "conversational, approachable, friendly",
        "technical": "technical, detailed, precise, developer-focused",
        "marketing": "persuasive, benefit-focused, engaging, conversion-optimized",
    }

    def __init__(
        self,
        openai_api_key: Optional[str] = None,
        anthropic_api_key: Optional[str] = None,
        rate_limiter: Optional[RateLimiter] = None,
        cost_tracker: Optional[CostTracker] = None,
    ):
        self.openai_api_key = openai_api_key
        self.anthropic_api_key = anthropic_api_key
        self.rate_limiter = rate_limiter
        self.cost_tracker = cost_tracker
        
        if openai_api_key:
            self.openai_client = openai.AsyncOpenAI(api_key=openai_api_key)
        else:
            self.openai_client = None
    
    async def generate(
        self,
        tool_name: str,
        website_url: str,
        existing_description: Optional[str] = None,
        tone: str = "professional",
        max_length: int = 500,
        include_features: bool = True,
        include_use_cases: bool = True,
    ) -> Dict[str, Any]:
        """Generate description for a tool"""
        
        if not self.openai_client:
            raise ValueError("OpenAI client not initialized")
        
        # Build prompt
        existing_desc_text = ""
        if existing_description:
            existing_desc_text = f"Existing Description (for reference):\n{existing_description}\n\n"
        
        prompt = self.DESCRIPTION_PROMPT.substitute(
            tool_name=tool_name,
            website_url=website_url,
            tone=self.TONE_MODIFIERS.get(tone, tone),
            max_length=max_length,
            min_desc_length=max(200, max_length // 3),
            existing_description=existing_desc_text,
        )
        
        # Call OpenAI API
        result = await self._call_openai(prompt, max_length)
        
        # Add metadata
        result["tool_name"] = tool_name
        result["tone"] = tone
        
        # Filter based on include flags
        if not include_features:
            result.pop("features", None)
        if not include_use_cases:
            result.pop("use_cases", None)
        
        return result
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def _call_openai(self, prompt: str, max_length: int) -> Dict[str, Any]:
        """Call OpenAI API with retry logic"""
        
        try:
            response = await self.openai_client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert AI tools directory curator. Generate accurate, engaging content. Always respond with valid JSON."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                max_tokens=1500,
                temperature=0.7,
                response_format={"type": "json_object"}
            )
            
            # Parse response
            content = response.choices[0].message.content
            result = json.loads(content)
            
            # Calculate cost
            tokens_used = response.usage.total_tokens if response.usage else 0
            # GPT-4 pricing: $0.03 per 1K prompt tokens, $0.06 per 1K completion tokens
            estimated_cost = (tokens_used / 1000) * 0.045  # Average rate
            
            # Track costs
            if self.cost_tracker:
                await self.cost_tracker.track_cost(
                    operation="description_generation",
                    provider="openai",
                    model="gpt-4",
                    tokens=tokens_used,
                    cost=estimated_cost,
                )
            
            # Add metadata to result
            result["tokens_used"] = tokens_used
            result["estimated_cost"] = round(estimated_cost, 6)
            result["model"] = "gpt-4"
            
            # Validate and clean result
            result = self._validate_result(result, max_length)
            
            return result
            
        except json.JSONDecodeError as e:
            print(f"❌ Failed to parse OpenAI response: {e}")
            raise
        except Exception as e:
            print(f"❌ OpenAI API error: {e}")
            raise
    
    def _validate_result(self, result: Dict[str, Any], max_length: int) -> Dict[str, Any]:
        """Validate and clean generated result"""
        
        # Ensure required fields exist
        if "description" not in result or not result["description"]:
            result["description"] = result.get("short_description", "AI-powered tool.")
        
        if "short_description" not in result or not result["short_description"]:
            # Generate from description
            desc = result["description"]
            sentences = desc.split('.')
            result["short_description"] = sentences[0][:120] if sentences else desc[:120]
        
        # Truncate if too long
        if len(result["description"]) > max_length:
            result["description"] = result["description"][:max_length-3] + "..."
        
        # Ensure features is a list
        if "features" not in result or not isinstance(result["features"], list):
            result["features"] = []
        
        # Ensure use_cases is a list
        if "use_cases" not in result or not isinstance(result["use_cases"], list):
            result["use_cases"] = []
        
        # Clean up features
        result["features"] = [
            f.strip() for f in result["features"]
            if f and len(f.strip()) > 5
        ][:7]  # Max 7 features
        
        # Clean up use_cases
        result["use_cases"] = [
            uc.strip() for uc in result["use_cases"]
            if uc and len(uc.strip()) > 5
        ][:4]  # Max 4 use cases
        
        return result
    
    async def generate_variations(
        self,
        tool_name: str,
        website_url: str,
        existing_description: Optional[str] = None,
        num_variations: int = 3,
    ) -> List[Dict[str, Any]]:
        """Generate multiple variations for A/B testing"""
        
        variations = []
        tones = ["professional", "casual", "marketing"]
        
        for i in range(min(num_variations, len(tones))):
            result = await self.generate(
                tool_name=tool_name,
                website_url=website_url,
                existing_description=existing_description,
                tone=tones[i],
            )
            result["variation_id"] = i + 1
            result["tone_used"] = tones[i]
            variations.append(result)
        
        return variations
    
    async def improve_description(
        self,
        current_description: str,
        improvement_type: str = "seo",
    ) -> Dict[str, Any]:
        """Improve an existing description"""
        
        improvement_prompts = {
            "seo": "Optimize this description for search engines. Include relevant keywords naturally.",
            "clarity": "Make this description clearer and more concise. Remove jargon and buzzwords.",
            "engagement": "Make this description more engaging and persuasive. Highlight benefits.",
            "technical": "Add technical details and specifications to this description.",
        }
        
        prompt = f"""{improvement_prompts.get(improvement_type, improvement_prompts['clarity'])}

Current Description:
{current_description}

Provide the improved description in JSON format:
{{
    "improved_description": "The improved description",
    "changes_made": ["Change 1", "Change 2"],
    "keywords_added": ["keyword1", "keyword2"]
}}"""
        
        try:
            response = await self.openai_client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert copywriter. Improve the given description."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                max_tokens=800,
                temperature=0.5,
                response_format={"type": "json_object"}
            )
            
            content = response.choices[0].message.content
            return json.loads(content)
            
        except Exception as e:
            print(f"❌ Error improving description: {e}")
            return {
                "improved_description": current_description,
                "changes_made": [],
                "keywords_added": [],
            }
