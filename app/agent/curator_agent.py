import os
import json
import re
from typing import List
from groq import Groq
from pydantic import BaseModel, Field, ValidationError
from dotenv import load_dotenv

load_dotenv()


# -------------------- Output Schemas --------------------

class RankedArticle(BaseModel):
    digest_id: str = Field(description="The ID of the digest (article_type:article_id)")
    relevance_score: float = Field(ge=0.0, le=10.0)
    rank: int = Field(ge=1)
    reasoning: str


class RankedDigestList(BaseModel):
    articles: List[RankedArticle]


# -------------------- System Prompt --------------------

BASE_PROMPT = """You are an AI system that ranks AI-related content for a specific user.

STRICT RULES:
- Output MUST be valid JSON only
- Do NOT include markdown, explanations, or extra text
- Do NOT wrap output in code blocks

SCORING RULES (MANDATORY):
- relevance_score is an INTEGER from 1 to 10
- 10 = highest relevance
- 1 = lowest relevance
- Rank 1 MUST have relevance_score = 10
- Scores MUST strictly decrease as rank increases
- No two articles may share the same relevance_score

JSON schema:
{
  "articles": [
    {
      "digest_id": "string",
      "relevance_score": number,
      "rank": number,
      "reasoning": "string"
    }
  ]
}

Rank articles from most relevant (rank 1) to least relevant.
Each article must have a UNIQUE rank.




"""


# -------------------- Utilities --------------------

def extract_json(text: str) -> dict:
    """Extract first JSON object from model output."""
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError("No JSON object found in model output")
    return json.loads(match.group())


# -------------------- Curator Agent --------------------

class CuratorAgent:
    def __init__(self, user_profile: dict):
        self.client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        self.model = "llama-3.1-8b-instant"
        self.user_profile = user_profile
        self.system_prompt = self._build_system_prompt()

    def _build_system_prompt(self) -> str:
        interests = "\n".join(f"- {i}" for i in self.user_profile["interests"])
        preferences = "\n".join(
            f"- {k}: {v}" for k, v in self.user_profile["preferences"].items()
        )

        return f"""{BASE_PROMPT}

User Profile:
Name: {self.user_profile["name"]}
Background: {self.user_profile["background"]}
Expertise Level: {self.user_profile["expertise_level"]}

Interests:
{interests}

Preferences:
{preferences}
"""

    def rank_digests(self, digests: List[dict]) -> List[RankedArticle]:
        if not digests:
            return []

        digest_text = "\n\n".join(
            f"ID: {d['id']}\nTitle: {d['title']}\nSummary: {d['summary']}\nType: {d['article_type']}"
            for d in digests
        )

        user_prompt = f"""
Rank the following {len(digests)} AI digests based on the user profile.

{digest_text}
"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.3,
            )

            raw_text = response.choices[0].message.content

            if not raw_text or not raw_text.strip():
                raise ValueError("Empty response from model")

            data = extract_json(raw_text)
            ranked = RankedDigestList(**data)

            return ranked.articles

        except (json.JSONDecodeError, ValidationError, ValueError) as e:
            print("❌ Failed to parse ranking output")
            print("RAW OUTPUT:\n", raw_text)
            print("ERROR:", e)
            return []

        except Exception as e:
            print("❌ Groq API error:", e)
            return []
