import threading
from typing import Dict, List

class SessionManager:
    """
    A thread-safe singleton to manage temporary session data, such as
    document chunks for single-session analysis.
    """
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if not cls._instance:
            with cls._lock:
                if not cls._instance:
                    cls._instance = super(SessionManager, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        if not hasattr(self, 'initialized'):
            with self._lock:
                if not hasattr(self, 'initialized'):
                    self.sessions: Dict[str, List[str]] = {}
                    self.initialized = True
                    print("🚀 Session Manager Initialized.")

    def add_temp_chunks(self, session_id: str, chunks: List[str]):
        """Adds document chunks to a temporary session."""
        with self._lock:
            self.sessions[session_id] = chunks
            print(f"Added {len(chunks)} chunks to session {session_id}")

    def get_temp_chunks(self, session_id: str) -> List[str]:
        """Retrieves chunks from a temporary session."""
        with self._lock:
            return self.sessions.get(session_id, [])

    def clear_session(self, session_id: str):
        """Clears all data associated with a session."""
        with self._lock:
            if session_id in self.sessions:
                del self.sessions[session_id]
                print(f"Cleared session {session_id}")

def get_session_manager():
    """Factory function to get the SessionManager instance."""
    return SessionManager()
