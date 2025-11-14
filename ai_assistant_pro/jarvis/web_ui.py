"""
JARVIS Web Interface

Beautiful, responsive web UI for interacting with JARVIS.
Built with FastAPI, WebSockets for real-time chat, and modern UI.
"""

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
from pydantic import BaseModel
from typing import List, Optional, Dict
import json
import asyncio
from datetime import datetime

from ai_assistant_pro.jarvis.memory import ConversationalMemory, MultiUserMemory
from ai_assistant_pro.utils.logging import get_logger

logger = get_logger("jarvis.web")


class ChatMessage(BaseModel):
    """Chat message model"""
    role: str  # "user" or "assistant"
    content: str
    timestamp: Optional[str] = None


class ChatRequest(BaseModel):
    """Chat request model"""
    message: str
    user_id: str = "default"
    use_memory: bool = True


class ConnectionManager:
    """Manage WebSocket connections"""

    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}

    async def connect(self, user_id: str, websocket: WebSocket):
        await websocket.accept()
        self.active_connections[user_id] = websocket
        logger.info(f"User {user_id} connected")

    def disconnect(self, user_id: str):
        if user_id in self.active_connections:
            del self.active_connections[user_id]
            logger.info(f"User {user_id} disconnected")

    async def send_message(self, user_id: str, message: dict):
        if user_id in self.active_connections:
            await self.active_connections[user_id].send_json(message)


class JARVISWebInterface:
    """
    JARVIS Web Interface Application

    Provides:
    - Real-time chat via WebSockets
    - Conversation history
    - Memory management
    - Voice interface toggle
    - Settings panel
    """

    def __init__(
        self,
        engine,
        memory_manager: Optional[MultiUserMemory] = None,
    ):
        """
        Initialize web interface

        Args:
            engine: AssistantEngine instance
            memory_manager: Multi-user memory manager
        """
        self.engine = engine
        self.memory_manager = memory_manager or MultiUserMemory()
        self.connection_manager = ConnectionManager()

        # Create FastAPI app
        self.app = FastAPI(
            title="JARVIS - AI Assistant",
            description="Just A Rather Very Intelligent System",
            version="1.0.0"
        )

        self._setup_routes()

        logger.info("✓ JARVIS Web Interface initialized")

    def _setup_routes(self):
        """Setup API routes"""

        @self.app.get("/", response_class=HTMLResponse)
        async def index():
            """Main page"""
            return self._get_html_template()

        @self.app.post("/api/chat")
        async def chat(request: ChatRequest):
            """REST API for chat"""
            try:
                response = await self._process_message(
                    request.message,
                    request.user_id,
                    request.use_memory
                )

                return {
                    "response": response,
                    "timestamp": datetime.now().isoformat(),
                }

            except Exception as e:
                logger.error(f"Error in chat: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.websocket("/ws/{user_id}")
        async def websocket_endpoint(websocket: WebSocket, user_id: str):
            """WebSocket endpoint for real-time chat"""
            await self.connection_manager.connect(user_id, websocket)

            try:
                while True:
                    # Receive message
                    data = await websocket.receive_json()
                    message = data.get("message", "")

                    if message:
                        # Process message
                        response = await self._process_message(
                            message,
                            user_id,
                            use_memory=True
                        )

                        # Send response
                        await self.connection_manager.send_message(
                            user_id,
                            {
                                "role": "assistant",
                                "content": response,
                                "timestamp": datetime.now().isoformat(),
                            }
                        )

            except WebSocketDisconnect:
                self.connection_manager.disconnect(user_id)

        @self.app.get("/api/history/{user_id}")
        async def get_history(user_id: str, limit: int = 50):
            """Get conversation history"""
            memory = self.memory_manager.get_memory(user_id, create_if_missing=False)

            if not memory:
                return {"messages": []}

            turns = memory.get_session_context(max_turns=limit)

            messages = []
            for turn in turns:
                messages.append({
                    "role": "user",
                    "content": turn.user_message,
                    "timestamp": datetime.fromtimestamp(turn.timestamp).isoformat(),
                })
                messages.append({
                    "role": "assistant",
                    "content": turn.assistant_response,
                    "timestamp": datetime.fromtimestamp(turn.timestamp).isoformat(),
                })

            return {"messages": messages}

        @self.app.post("/api/clear/{user_id}")
        async def clear_history(user_id: str):
            """Clear conversation history"""
            memory = self.memory_manager.get_memory(user_id, create_if_missing=False)

            if memory:
                memory.clear_session()

            return {"status": "cleared"}

        @self.app.get("/api/stats/{user_id}")
        async def get_stats(user_id: str):
            """Get memory statistics"""
            memory = self.memory_manager.get_memory(user_id, create_if_missing=False)

            if not memory:
                return {"error": "No memory found"}

            return memory.get_statistics()

    async def _process_message(
        self,
        message: str,
        user_id: str,
        use_memory: bool = True,
    ) -> str:
        """
        Process user message and generate response

        Args:
            message: User message
            user_id: User identifier
            use_memory: Whether to use memory

        Returns:
            Assistant response
        """
        # Get user memory
        memory = self.memory_manager.get_memory(user_id) if use_memory else None

        # Build prompt with context
        if memory:
            context = memory.get_context_summary(query=message)
            prompt = f"{context}\n\nUser: {message}\nAssistant:"
        else:
            prompt = message

        # Generate response
        response = self.engine.generate(
            prompt=prompt,
            max_tokens=200,
            temperature=0.7,
        )

        # Extract just the assistant's response (remove prompt echo)
        if memory:
            # Store in memory
            memory.add_turn(
                user_message=message,
                assistant_response=response,
            )

        return response

    def _get_html_template(self) -> str:
        """Get HTML template for chat interface"""
        return """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>JARVIS - AI Assistant</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            height: 100vh;
            display: flex;
            justify-content: center;
            align-items: center;
        }

        .container {
            width: 90%;
            max-width: 800px;
            height: 90vh;
            background: white;
            border-radius: 20px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            display: flex;
            flex-direction: column;
            overflow: hidden;
        }

        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            text-align: center;
        }

        .header h1 {
            font-size: 2em;
            margin-bottom: 5px;
        }

        .header p {
            opacity: 0.9;
            font-size: 0.9em;
        }

        .chat-container {
            flex: 1;
            overflow-y: auto;
            padding: 20px;
            background: #f5f5f5;
        }

        .message {
            margin-bottom: 15px;
            display: flex;
            align-items: flex-start;
        }

        .message.user {
            justify-content: flex-end;
        }

        .message.assistant {
            justify-content: flex-start;
        }

        .message-content {
            max-width: 70%;
            padding: 12px 18px;
            border-radius: 18px;
            word-wrap: break-word;
        }

        .message.user .message-content {
            background: #667eea;
            color: white;
            border-bottom-right-radius: 4px;
        }

        .message.assistant .message-content {
            background: white;
            color: #333;
            border-bottom-left-radius: 4px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }

        .input-container {
            padding: 20px;
            background: white;
            border-top: 1px solid #eee;
            display: flex;
            gap: 10px;
        }

        #messageInput {
            flex: 1;
            padding: 12px 18px;
            border: 2px solid #667eea;
            border-radius: 25px;
            font-size: 1em;
            outline: none;
            transition: border-color 0.3s;
        }

        #messageInput:focus {
            border-color: #764ba2;
        }

        #sendButton {
            padding: 12px 30px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            border-radius: 25px;
            cursor: pointer;
            font-size: 1em;
            font-weight: bold;
            transition: transform 0.2s;
        }

        #sendButton:hover {
            transform: scale(1.05);
        }

        #sendButton:active {
            transform: scale(0.95);
        }

        .typing-indicator {
            display: none;
            padding: 10px 18px;
            background: white;
            border-radius: 18px;
            width: fit-content;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }

        .typing-indicator span {
            height: 8px;
            width: 8px;
            background: #667eea;
            border-radius: 50%;
            display: inline-block;
            margin: 0 2px;
            animation: typing 1.4s infinite;
        }

        .typing-indicator span:nth-child(2) {
            animation-delay: 0.2s;
        }

        .typing-indicator span:nth-child(3) {
            animation-delay: 0.4s;
        }

        @keyframes typing {
            0%, 60%, 100% {
                transform: translateY(0);
            }
            30% {
                transform: translateY(-10px);
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🤖 JARVIS</h1>
            <p>Just A Rather Very Intelligent System</p>
        </div>

        <div class="chat-container" id="chatContainer">
            <div class="message assistant">
                <div class="message-content">
                    Hello! I'm JARVIS, your AI assistant. How can I help you today?
                </div>
            </div>
        </div>

        <div class="typing-indicator" id="typingIndicator">
            <span></span>
            <span></span>
            <span></span>
        </div>

        <div class="input-container">
            <input
                type="text"
                id="messageInput"
                placeholder="Type your message..."
                autocomplete="off"
            />
            <button id="sendButton">Send</button>
        </div>
    </div>

    <script>
        const userId = 'user_' + Math.random().toString(36).substr(2, 9);
        const ws = new WebSocket(`ws://${window.location.host}/ws/${userId}`);
        const chatContainer = document.getElementById('chatContainer');
        const messageInput = document.getElementById('messageInput');
        const sendButton = document.getElementById('sendButton');
        const typingIndicator = document.getElementById('typingIndicator');

        ws.onmessage = function(event) {
            const data = JSON.parse(event.data);
            addMessage('assistant', data.content);
            typingIndicator.style.display = 'none';
        };

        function addMessage(role, content) {
            const messageDiv = document.createElement('div');
            messageDiv.className = `message ${role}`;

            const contentDiv = document.createElement('div');
            contentDiv.className = 'message-content';
            contentDiv.textContent = content;

            messageDiv.appendChild(contentDiv);
            chatContainer.appendChild(messageDiv);

            // Scroll to bottom
            chatContainer.scrollTop = chatContainer.scrollHeight;
        }

        function sendMessage() {
            const message = messageInput.value.trim();

            if (message) {
                // Add user message
                addMessage('user', message);

                // Show typing indicator
                typingIndicator.style.display = 'block';
                chatContainer.appendChild(typingIndicator);

                // Send via WebSocket
                ws.send(JSON.stringify({message: message}));

                // Clear input
                messageInput.value = '';
            }
        }

        sendButton.addEventListener('click', sendMessage);

        messageInput.addEventListener('keypress', function(e) {
            if (e.key === 'Enter') {
                sendMessage();
            }
        });

        messageInput.focus();
    </script>
</body>
</html>
        """

    def run(self, host: str = "0.0.0.0", port: int = 8080):
        """
        Run web interface

        Args:
            host: Host to bind to
            port: Port to bind to
        """
        import uvicorn

        logger.info(f"🚀 Starting JARVIS Web Interface on http://{host}:{port}")
        uvicorn.run(self.app, host=host, port=port)


# CLI command to start web UI
if __name__ == "__main__":
    from ai_assistant_pro import AssistantEngine

    engine = AssistantEngine(model_name="gpt2")
    web_ui = JARVISWebInterface(engine=engine)
    web_ui.run()
