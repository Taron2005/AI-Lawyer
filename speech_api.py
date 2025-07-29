# speech_api.py
from pathlib import Path
from groq import Groq
from dotenv import load_dotenv
import os
from pathlib import Path
import tempfile
import os






load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))


def tts_long_text(text: str, base_filename: str = "speech", max_chunk_length: int = 800, max_chunks: int = 5):
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))

    chunks = []
    start = 0
    text_len = len(text)
    while start < text_len:
        end = min(start + max_chunk_length, text_len)
        if end < text_len:
            last_space = text.rfind(" ", start, end)
            if last_space > start:
                end = last_space
        chunks.append(text[start:end].strip())
        start = end

    audio_paths = []
    # Limit number of chunks to avoid excessive token usage
    if len(chunks) > max_chunks:
        return False, (
            f"Text is too long for speech synthesis. "
            f"Please shorten your text or try again with a smaller input."
        )

    for i, chunk in enumerate(chunks):
        speech_file_path = Path(__file__).parent / f"{base_filename}_{i}.wav"
        try:
            response = client.audio.speech.create(
                model="playai-tts",
                voice="Aaliyah-PlayAI",
                response_format="wav",
                input=chunk,
            )
            response.write_to_file(speech_file_path)
            audio_paths.append(speech_file_path)
        except Exception as e:
            err_msg = str(e)
            if "rate_limit_exceeded" in err_msg or "Limit" in err_msg:
                return False, (
                    "Rate limit reached for speech synthesis API. "
                    "Please wait and try again later or upgrade your plan."
                )
            return False, f"TTS failed at chunk {i+1}: {err_msg}"

    return True, audio_paths