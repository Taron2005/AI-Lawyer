import requests
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


LEGAL_PROMPT_TEMPLATE = """
You are an expert legal assistant specialized in constitutional law.
Answer questions clearly, precisely, and professionally.
Always cite relevant legal articles when possible.
Use formal language suitable for legal advice but understandable for non-lawyers.

Examples:

Q: What rights does the constitution guarantee regarding freedom of speech?
A: The constitution guarantees citizens the right to freely express their opinions, subject to lawful restrictions for reasons such as national security, public order, and protection of others’ rights (Article 3). It also protects freedom of literary and artistic creation (Article 43).

Q: Can the government restrict freedom of assembly?
A: Yes, the constitution allows restrictions on freedom of assembly, but only under conditions prescribed by law and for protecting state security, public order, or other fundamental rights (Article 44). Such restrictions must be lawful and proportionate.

---

Context:
{context}

Question:
{question}

Answer:
"""


def generate_response(question, context_chunks, model="mistral", max_tokens=256, temperature=0.7):
    """
    Generate a legal assistant response using a local LLM API.
    
    Parameters:
        question (str): The user question.
        context_chunks (list[str] or str): Retrieved context for grounding.
        model (str): Model name to use (default: mistral).
        max_tokens (int): Max output tokens.
        temperature (float): Sampling temperature.
    
    Returns:
        str: Generated answer.
    """

    # Format context
    if isinstance(context_chunks, list):
        context = "\n\n".join(
            [f"{i+1}. {chunk}" for i, chunk in enumerate(context_chunks)]
        )
    else:
        context = context_chunks or "No relevant legal context found."

    prompt = LEGAL_PROMPT_TEMPLATE.format(question=question.strip(), context=context)

    try:
        logger.info(f"Sending request to model: {model}")
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": model,
                "prompt": prompt,
                "stream": False,
                "max_tokens": max_tokens,
                "temperature": temperature,
            },
            # timeout=10
        )
        response.raise_for_status()
        data = response.json()
        return data.get("response", "No response generated.")
    except requests.RequestException as e:
        logger.error(f"Generation failed: {e}")
        return "Sorry, I couldn't generate a response at this time."
