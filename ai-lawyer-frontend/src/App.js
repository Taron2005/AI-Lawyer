import React, { useState, useRef, useEffect, useMemo } from 'react';
import { marked } from 'marked';
import './App.css';

// --- Configuration ---
const API_URL = "http://127.0.0.1:8000";

// --- Global Setup for Markdown Parsing ---
marked.setOptions({
  breaks: true,
});

// --- Helper function to wrap citations in a styled span ---
const formatCitations = (text) => {
  if (!text) return '';
  // Looks for (Source: ...) and wraps it in a span for styling
  const citationRegex = /\(Source: ([^)]+)\)/g;
  return text.replace(citationRegex, '<span class="citation">(Source: $1)</span>');
};


// --- UI & Icon Components ---
const PaperclipIcon = () => <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="m21.44 11.05-9.19 9.19a6 6 0 0 1-8.49-8.49l8.57-8.57A4 4 0 1 1 18 8.84l-8.59 8.59a2 2 0 0 1-2.83-2.83l8.49-8.48"/></svg>;
const SendIcon = () => <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="m22 2-7 20-4-9-9-4Z"/><path d="M22 2 11 13"/></svg>;
const FileTextIcon = () => <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M15 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V7Z"/><path d="M14 2v4a2 2 0 0 0 2 2h4"/><path d="M10 9H8"/><path d="M16 13H8"/><path d="M16 17H8"/></svg>;
const Trash2Icon = () => <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M3 6h18"/><path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6"/><path d="M8 6V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"/><line x1="10" x2="10" y1="11" y2="17"/><line x1="14" x2="14" y1="11" y2="17"/></svg>;
const Spinner = () => <div className="spinner"></div>;

// Component for the "..." typing animation
const TypingIndicator = () => (
    <div className="message-wrapper assistant">
        <div className="message-bubble assistant typing-indicator">
            <div className="dot"></div>
            <div className="dot"></div>
            <div className="dot"></div>
        </div>
    </div>
);

// Component for a single chat message
const Message = React.memo(({ role, text }) => {
    const isUser = role === 'user';
    const sanitizedHtml = useMemo(() => {
        const formattedText = isUser ? text : formatCitations(text);
        return marked.parse(formattedText);
    }, [text, isUser]);

    return (
        <div className={`message-wrapper ${isUser ? 'user' : 'assistant'}`}>
            <div 
                className={`message-bubble ${isUser ? 'user' : 'assistant'}`} 
                dangerouslySetInnerHTML={{ __html: sanitizedHtml }} 
            />
        </div>
    );
});


// --- Main App Component ---
export default function App() {
    const [messages, setMessages] = useState([
        { role: 'assistant', text: 'Hello! I am your AI Legal Assistant. How can I help you today?' }
    ]);
    const [input, setInput] = useState('');
    const [isLoading, setIsLoading] = useState(false);
    const [notification, setNotification] = useState({ message: null, type: 'info' });
    const chatEndRef = useRef(null);
    
    const [fileToUpload, setFileToUpload] = useState(null);
    const [isUploading, setIsUploading] = useState(false);
    const [uploadType, setUploadType] = useState('temp');
    const [sessionId, setSessionId] = useState(null);
    const fileInputRef = useRef(null);

    useEffect(() => {
        chatEndRef.current?.scrollIntoView({ behavior: 'smooth' });
    }, [messages, isLoading]);

    const showNotification = (message, type = 'info') => {
        setNotification({ message, type });
        setTimeout(() => setNotification({ message: null, type: 'info' }), 5000);
    };

    const handleSendMessage = async (e) => {
        e.preventDefault();
        if (!input.trim() || isLoading) return;

        const userMessage = { role: 'user', text: input };
        setMessages(prev => [...prev, userMessage]);
        const currentChatHistory = [...messages, userMessage].slice(1); // Exclude initial message
        setInput('');
        setIsLoading(true);

        const bodyPayload = { 
            question: input,
            chat_history: currentChatHistory.map(msg => ({ role: msg.role, content: msg.text }))
        };
        if (sessionId) {
            bodyPayload.session_id = sessionId;
        }

        try {
            const response = await fetch(`${API_URL}/ask`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(bodyPayload),
            });

            if (!response.ok) {
                let detailMessage = `An error occurred on the server (status: ${response.status}).`;
                try {
                    const errData = await response.json();
                    detailMessage = (typeof errData.detail === 'string') 
                        ? errData.detail 
                        : JSON.stringify(errData.detail);
                } catch (jsonError) {
                    console.error("Could not parse error response as JSON.", jsonError);
                }
                throw new Error(detailMessage);
            }
            const data = await response.json();
            const assistantMessage = { role: 'assistant', text: data.answer };
            setMessages(prev => [...prev, assistantMessage]);
        } catch (error) {
            const message = error.message || String(error);
            const errorMessage = { role: 'assistant', text: `❌ **Error:** ${message}. Please ensure the backend server is running.` };
            setMessages(prev => [...prev, errorMessage]);
        } finally {
            setIsLoading(false);
        }
    };
    
    const handleFileChange = (e) => {
        const selectedFile = e.target.files[0];
        if (selectedFile && (selectedFile.type === "application/pdf" || selectedFile.type === "text/plain")) {
            setFileToUpload(selectedFile);
        } else if (selectedFile) {
            showNotification("Invalid file type. Please select a PDF or TXT file.", "error");
        }
    };

    const handleRemoveFile = () => {
        setFileToUpload(null);
        if(fileInputRef.current) fileInputRef.current.value = "";
    };

    const handleUpload = async () => {
        if (!fileToUpload) return;
        setIsUploading(true);

        const formData = new FormData();
        formData.append('file', fileToUpload);
        const endpoint = uploadType === 'temp' ? '/upload-temp' : '/upload-permanent';

        try {
            const response = await fetch(`${API_URL}${endpoint}`, { method: 'POST', body: formData });
            if (!response.ok) {
                let detailMessage = `HTTP error! status: ${response.status}`;
                try {
                    const errData = await response.json();
                    detailMessage = (typeof errData.detail === 'string') 
                        ? errData.detail 
                        : JSON.stringify(errData.detail);
                } catch (jsonError) {
                     console.error("Could not parse upload error response as JSON.", jsonError);
                }
                throw new Error(detailMessage);
            }
            const result = await response.json();
            
            if(uploadType === 'temp' && result.session_id) {
                setSessionId(result.session_id);
                showNotification(`✅ File ready for this session. Session ID: ${result.session_id.substring(0, 8)}...`, 'success');
            } else {
                showNotification(`✅ "${result.filename}" permanently added. Vectors: ${result.vector_count}.`, 'success');
            }
            handleRemoveFile();
        } catch (err) {
            const message = err.message || String(err);
            showNotification(`Upload failed: ${message}`, 'error');
        } finally {
            setIsUploading(false);
        }
    };

    const handleNewSession = () => {
        setMessages([{ role: 'assistant', text: 'New session started. Previous temporary files are cleared.' }]);
        setSessionId(null);
        handleRemoveFile();
        showNotification("New session started.", "info");
    };

    return (
        <div className="app-container">
            <header className="app-header">
                <div>
                    <h1>AI Lawyer</h1>
                    <p>Session ID: {sessionId ? `${sessionId.substring(0, 8)}...` : 'None'}</p>
                </div>
                <button onClick={handleNewSession} className="button button-new-session">
                    New Session
                </button>
            </header>
            
            {notification.message && (
                <div className={`notification ${notification.type}`}>
                    {notification.message}
                </div>
            )}

            <main className="messages-area">
                {messages.map((msg, index) => <Message key={index} role={msg.role} text={msg.text} />)}
                {isLoading && <TypingIndicator />}
                <div ref={chatEndRef} />
            </main>

            <footer className="app-footer">
                {fileToUpload && (
                    <div className="file-upload-info">
                        <div className="file-upload-info-header">
                            <div className="file-info">
                                <FileTextIcon />
                                <span>{fileToUpload.name}</span>
                            </div>
                            <button onClick={handleRemoveFile} className="button-icon"><Trash2Icon /></button>
                        </div>
                        <div className="file-upload-info-footer">
                            <div className="upload-options">
                                <label><input type="radio" name="uploadType" value="temp" checked={uploadType === 'temp'} onChange={() => setUploadType('temp')} /> Temp</label>
                                <label><input type="radio" name="uploadType" value="permanent" checked={uploadType === 'permanent'} onChange={() => setUploadType('permanent')} /> Permanent</label>
                            </div>
                            <button onClick={handleUpload} disabled={isUploading} className="button button-upload">
                                {isUploading ? <Spinner /> : 'Upload File'}
                            </button>
                        </div>
                    </div>
                )}
                <form onSubmit={handleSendMessage} className="message-form">
                    <input type="file" ref={fileInputRef} onChange={handleFileChange} style={{ display: 'none' }} accept=".pdf,.txt" />
                    <button type="button" onClick={() => fileInputRef.current.click()} className="button-icon" title="Attach File">
                        <PaperclipIcon />
                    </button>
                    <input
                        type="text"
                        value={input}
                        onChange={(e) => setInput(e.target.value)}
                        placeholder="Ask a legal question or upload a file..."
                        className="text-input"
                        disabled={isLoading}
                    />
                    <button type="submit" disabled={!input.trim() || isLoading} className="button-send" title="Send Message">
                        <SendIcon />
                    </button>
                </form>
            </footer>
        </div>
    );
}