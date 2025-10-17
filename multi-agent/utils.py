from typing import Dict
 
def render_prompt(template: str, vars: Dict[str, str]) -> str:
    return template.format(**vars)