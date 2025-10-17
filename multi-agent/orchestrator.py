#orchastor.py
from agents.lifestyle_agent import get_lifestyle_advice

from agents.risk_agent import get_risk_report
 
def orchestrate(context: str) -> dict:

    """Synchronous orchestration: call risk agent then lifestyle agent and merge outputs."""

    risk = get_risk_report(context)

    lifestyle = get_lifestyle_advice(context)
 
    final = f"""Final Recommendation:
 
Risk Report:

{risk}
 
Lifestyle Advice:

{lifestyle}
 
Combined Plan:

- If any urgent flags are present in the risk report, recommend clinical follow-up.

- Otherwise follow the 3-step lifestyle plan and re-assess in 4 weeks.

""".strip()
 
    return {

        "risk_report": risk,

        "lifestyle_advice": lifestyle,

        "final_recommendation": final,

    }

 