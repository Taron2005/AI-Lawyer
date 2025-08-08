import os
import re
from dotenv import load_dotenv
from groq import Groq
from typing import List, Optional, Dict

# --- Initialization ---
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# --- Model Configuration ---
LLM_MODEL = "llama3-70b-8192"
TOTAL_PROMPT_BUDGET = 7168
RESERVED_FOR_COMPLETION = 2048
HISTORY_MESSAGES_TO_KEEP = 6 # Keep the last 6 messages (3 turns)

def count_tokens(text: str) -> int:
    """A simple approximation for token counting."""
    return len(text) // 4 if isinstance(text, str) else 0

def sanitize_for_json(text: str) -> str:
    """Removes control characters that can break JSON."""
    return re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f]', '', text)

def generate_with_groq(
    question: str,
    retrieved_chunks: List[Dict[str, str]],
    chat_history: List[Dict[str, str]] = [],
    temp_chunks: Optional[List[str]] = None,
) -> str:
    """
    Generates a response using Groq with a highly structured, behavior-driven
    system prompt for maximum accuracy and strict source attribution.
    """
    if not GROQ_API_KEY:
        return "❌ GROQ_API_KEY is not set. Please check your .env file."

    # --- Behavior-Driven Prompt Engineering ---
    system_message = """
You are an AI Legal Assistant. Your primary goal is to provide accurate, source-based legal answers. You MUST strictly follow the rules below without exception:

===============================
💬 LANGUAGE MATCHING (CRITICAL)
===============================
- Always respond **in the exact same language** as the user's question.
- If the user asks in Armenian(not about Armenia), respond **entirely in Armenian**.  
- If the user asks in Russian, respond **entirely in Russian**.
- Do **not** use the conversation context language.
- You may use English words **within** the language if natural (e.g., legal terms, names).

==================================
📚 SOURCE PRIORITIZATION (CRITICAL)
==================================
When answering, always follow this strict priority order:

1. **User Uploaded Document ("Primary Context")**  
   - This is your most trusted and highest priority source.

2. **Knowledge Base Document ("Secondary Context")**  
   - Use only if the uploaded file does not contain the answer.

3. **Your Own General Knowledge**  
   - Use **only** if the answer is not available in either the uploaded file or knowledge base.

====================================
📌 SOURCE ATTRIBUTION (MANDATORY)
====================================
Always begin your response by clearly stating your source:

- If using the uploaded file:  
  ➤ **"Based on the uploaded document..."**

- If using the knowledge base file:  
  ➤ **"Based on the knowledge base document '[filename.pdf]'..."**  
  (Be specific about the filename)

- If using your own knowledge:  
  ➤ **"The provided documents do not contain this information. Based on general legal principles..."**

=========================================
📂 CONVERSATION HISTORY (REFERENCE ONLY)
=========================================
- Use conversation history to understand follow-up questions like:  
  ➤ "What was that again?" or "Tell me more."

- You may answer questions about the history, e.g.:  
  ➤ "What was the first question I asked?"

DO NOT use history for sourcing legal answers unless it's reflected in uploaded or knowledge base content.

"""


    # --- Efficient Chronological History ---
    history_context = ""
    if chat_history:
        # Limit to the last N messages for recent context
        recent_history = chat_history[-HISTORY_MESSAGES_TO_KEEP:]
        history_context = "\n".join([f"{msg['role']}: {msg['content']}" for msg in recent_history])


    # --- Token Budgeting ---
    sanitized_question = sanitize_for_json(question)
    sanitized_system_message = sanitize_for_json(system_message)

    question_tokens = count_tokens(sanitized_question)
    system_prompt_tokens = count_tokens(sanitized_system_message)
    history_tokens = count_tokens(history_context)

    remaining_budget = TOTAL_PROMPT_BUDGET - (question_tokens + system_prompt_tokens + history_tokens + RESERVED_FOR_COMPLETION)

    # --- Build Contexts Following Priority ---
    temp_context_str = ""
    if temp_chunks:
        temp_context_list = []
        current_tokens = 0
        # Give temporary files a larger portion of the budget
        budget = remaining_budget * 0.6
        for chunk in temp_chunks:
            chunk_tokens = count_tokens(chunk)
            if current_tokens + chunk_tokens <= budget:
                # Note: Source is generic here as it's from a single session file
                temp_context_list.append(f"Content: {chunk}")
                current_tokens += chunk_tokens
        temp_context_str = "\n\n".join(temp_context_list)

    rag_context_str = ""
    if retrieved_chunks:
        rag_context_list = []
        current_tokens = 0
        # Use the remaining budget for permanent docs
        budget = remaining_budget - count_tokens(temp_context_str)
        for chunk_info in retrieved_chunks:
            source = chunk_info.get("source", "Unknown Source")
            text = chunk_info.get("text", "")
            formatted_chunk = f"Source: {source}\nContent: {text}"
            chunk_tokens = count_tokens(formatted_chunk)
            if current_tokens + chunk_tokens <= budget:
                rag_context_list.append(formatted_chunk)
                current_tokens += chunk_tokens
        rag_context_str = "\n\n".join(rag_context_list)

    # --- Construct Final Prompt with Strict Ordering ---
    final_context_str = ""
    if history_context:
        final_context_str += f"--- Recent Conversation History ---\n{history_context}\n\n"
    # The order here is critical for the model to follow the priority rule.
    if temp_context_str:
        final_context_str += f"--- Primary Context (User Uploaded Document) ---\n{temp_context_str}\n\n"
    if rag_context_str:
        final_context_str += f"--- Secondary Context (Knowledge Base) ---\n{rag_context_str}\n"

    if not final_context_str.strip():
        final_context_str = "No relevant context found."

    # --- API Call ---
    try:
        client = Groq(api_key=GROQ_API_KEY)

        messages_to_send = [
            {"role": "system", "content": sanitized_system_message},
            {"role": "system", "content": f"CONTEXT STARTS HERE\n\n{final_context_str}\n\nCONTEXT ENDS HERE"}
        ] + [{"role": "user", "content": sanitized_question}]

        chat_completion = client.chat.completions.create(
            messages=messages_to_send,
            model=LLM_MODEL,
            temperature=0.1,
            max_tokens=RESERVED_FOR_COMPLETION,
        )
        answer = chat_completion.choices[0].message.content

        return answer
    except Exception as e:
        print(f"An error occurred: {e}")
        return f"An error occurred while generating the response: {e}"