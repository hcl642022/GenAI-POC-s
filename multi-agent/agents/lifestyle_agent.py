from .gemini_client import call_gemini
from utils import render_prompt
 
PROMPT_TEMPLATE = """You are a friendly certified lifestyle coach.
User context:
{context}
 
Provide a concise, actionable 3-step lifestyle plan tailored for the user.
Keep it short (max ~200 words). Use bulleted steps.
"""
 
def get_lifestyle_advice(context: str) -> str:
    prompt = render_prompt(PROMPT_TEMPLATE, {'context': context})
    return call_gemini(prompt)