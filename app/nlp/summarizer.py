"""
Primary: Groq (llama3-70b-8192) - free tier, fast
Fallback: Gemini (gemini-1.5-flash) - free tier
Final fallback: extractive first 350 chars
"""
import httpx
from app.config import settings

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent"

PROMPT = """Summarize this PR mention in 2-3 sentences. Max 350 characters. 
Focus on: who, what action, why it matters for PR/comms teams.
Text: {text}"""

async def summarize(text: str) -> str:
    # Try Groq first
    if settings.GROQ_API_KEY:
        try:
            return await _groq_summarize(text)
        except Exception:
            pass

    # Fallback to Gemini
    if settings.GEMINI_API_KEY:
        try:
            return await _gemini_summarize(text)
        except Exception:
            pass

    # Final fallback: extractive
    return _extractive_summarize(text)

def _extractive_summarize(text: str) -> str:
    """Sentence-importance fallback: pick top sentences by word frequency."""
    import re
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    if not sentences:
        return text[:347] + "..."
    
    # Score by word frequency
    words = re.findall(r'\w+', text.lower())
    freq = {}
    for w in words:
        if len(w) > 3:
            freq[w] = freq.get(w, 0) + 1
    
    def score(s):
        return sum(freq.get(w.lower(), 0) for w in re.findall(r'\w+', s))
    
    scored = sorted(sentences, key=score, reverse=True)
    summary = ""
    for s in scored:
        if len(summary) + len(s) + 1 <= 347:
            summary += s + " "
        if len(summary) >= 200:
            break
    return (summary.strip() or text[:347] + "...")

async def _groq_summarize(text: str) -> str:
    async with httpx.AsyncClient(timeout=15) as client:
        resp = await client.post(
            GROQ_URL,
            headers={
                "Authorization": f"Bearer {settings.GROQ_API_KEY}",
                "Content-Type": "application/json"
            },
            json={
                "model": settings.GROQ_MODEL,
                "messages": [{"role": "user", "content": PROMPT.format(text=text[:2000])}],
                "max_tokens": 120,
                "temperature": 0.3
            }
        )
        resp.raise_for_status()
        summary = resp.json()["choices"][0]["message"]["content"].strip()
        return summary[:350]

async def _gemini_summarize(text: str) -> str:
    async with httpx.AsyncClient(timeout=15) as client:
        resp = await client.post(
            f"{GEMINI_URL}?key={settings.GEMINI_API_KEY}",
            json={
                "contents": [{
                    "parts": [{"text": PROMPT.format(text=text[:2000])}]
                }],
                "generationConfig": {"maxOutputTokens": 120, "temperature": 0.3}
            }
        )
        resp.raise_for_status()
        return resp.json()["candidates"][0]["content"]["parts"][0]["text"][:350]
