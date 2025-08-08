import os
import logging
from dotenv import load_dotenv
from groq import Groq

load_dotenv()
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("Missing GROQ_API_KEY in .env")

client = Groq(api_key=GROQ_API_KEY)

LEGAL_PROMPT_TEMPLATE = """
You are a highly knowledgeable and professional legal assistant specializing exclusively in constitutional law. 
Use only the given context to answer clearly and concisely.

Context:
{context}

Question:
{question}

Answer:
"""

def generate_response(question, context_chunks, model="llama-3.3-70b-versatile", temperature=0.1, max_tokens=512):
    context = "\n\n".join(f"{i+1}. {chunk}" for i, chunk in enumerate(context_chunks)) or "No context available."
    prompt = LEGAL_PROMPT_TEMPLATE.format(context=context, question=question.strip())

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a constitutional law assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=temperature,
            max_tokens=max_tokens
        )
        return response.choices[0].message.content.strip()

    except Exception as e:
        logger.exception("Groq SDK call failed")
        raise RuntimeError(f"LLM generation error: {e}")

# NEW: Extract triples from text
def extract_triples_from_text(text: str, model="llama-3.3-70b-versatile", temperature=0.3) -> list:
    prompt = f"""
You are a legal text analysis expert. Extract subject-relation-object triples from the following legal paragraph. 
Return them as a list of dictionaries in the following format:

[
  {{ "subject": "...", "relation": "...", "object": "..." }},
  ...
]

Text:
\"\"\"{text.strip()}\"\"\"
"""

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=temperature,
            max_tokens=1024
        )

        output = response.choices[0].message.content.strip()
        triples = eval(output)  # assumes LLM gives valid Python-style list
        valid_triples = [t for t in triples if all(k in t for k in ("subject", "relation", "object"))]
        return valid_triples

    except Exception as e:
        logger.warning(f"Triple extraction failed: {e}")
        return []
