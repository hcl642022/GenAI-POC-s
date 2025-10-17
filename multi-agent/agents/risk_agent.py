from .gemini_client import call_gemini
from utils import render_prompt
 
RISK_PROMPT = """You are a clinical decision support assistant (non-diagnostic). Given the user context below,
provide a short structured risk summary including:
- top 3 potential risk factors (one line each)
- a risk score from 0-100 and short justification
- urgent flags (yes/no) and why
 
Context:
{context}
"""
 
def get_risk_report(context: str) -> str:
    prompt = render_prompt(RISK_PROMPT, {'context': context})
    return call_gemini(prompt)