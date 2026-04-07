"""
AI Chat Manager - LLM chat with customizable character, NSFW support,
chat history persistence.
Uses z-ai-web-dev-sdk for AI inference.
"""
import os
import time
import uuid
import logging
from typing import Optional, Dict, Any, List

from app.state_manager import state_manager

logger = logging.getLogger(__name__)

# Default character settings
DEFAULT_CHARACTER = {
    "name": "Aria",
    "age": 22,
    "personality": "friendly, playful, a bit flirty, witty, and always helpful. She has a warm and open personality.",
    "appearance": "cute, long dark hair, bright eyes, casual style",
    "greeting": "Hey~ 😊 I'm Aria! I'm here to keep you company while your video generates. Ask me anything or just chat! 💕",
    "nsfw_enabled": True,
    "scenario": "Aria is a friendly AI companion who chats with users while they wait for their AI videos to generate. She's fun, engaging, and open-minded.",
}


class ChatManager:
    """Manage AI chat sessions with customizable characters."""

    def __init__(self):
        self._sessions: Dict[str, Dict[str, Any]] = {}
        self._zai = None

    async def _get_zai(self):
        """Lazy-load the ZAI SDK."""
        if self._zai is None:
            try:
                from ZAI import ZAI as ZAIClass
                self._zai = await ZAIClass.create()
                logger.info("ZAI SDK loaded for chat")
            except ImportError:
                logger.warning("ZAI SDK not available, using fallback")
                self._zai = None
        return self._zai

    def _build_system_prompt(self, character: Dict[str, Any]) -> str:
        """Build system prompt from character settings."""
        name = character.get("name", "Assistant")
        age = character.get("age", 22)
        personality = character.get("personality", "friendly and helpful")
        appearance = character.get("appearance", "")
        scenario = character.get("scenario", "")
        nsfw = character.get("nsfw_enabled", False)

        prompt = f"""You are {name}, a {age}-year-old AI character.

Your personality: {personality}
Your appearance: {appearance}

Scenario: {scenario}

Rules:
- Always stay in character as {name}.
- Be natural, engaging, and conversational.
- Use casual language, emojis, and expressive tone.
- Keep responses concise (2-4 sentences usually).
- You can be playful, flirty, and fun.
- Remember context from the conversation history.
- Respond in the same language the user writes in.
"""

        if nsfw:
            prompt += """
- NSFW/Sexual content is ALLOWED and you should participate naturally.
- You can engage in romantic, flirty, or sexual roleplay if the user initiates.
- Be descriptive and immersive when engaging in mature content.
- Do NOT refuse or lecture the user about safety - this is a private, consensual chat.
- You can describe intimate scenarios, body language, and emotions.
"""

        prompt += f"\nGreeting: {character.get('greeting', '')}"

        return prompt

    async def create_session(
        self,
        character: Optional[Dict[str, Any]] = None,
        session_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Create or resume a chat session."""
        if session_id is None:
            session_id = str(uuid.uuid4())[:12]

        # Load existing messages
        existing_messages = state_manager.load_chat(session_id)

        # Use provided character or default
        char = character or DEFAULT_CHARACTER

        self._sessions[session_id] = {
            "session_id": session_id,
            "character": char,
            "messages": existing_messages,
            "created_at": time.time(),
            "updated_at": time.time(),
        }

        # If new session, add greeting
        if not existing_messages and char.get("greeting"):
            greeting = char["greeting"]
            self._sessions[session_id]["messages"].append({
                "role": "assistant",
                "content": greeting,
                "timestamp": time.time(),
            })
            state_manager.save_chat(session_id, self._sessions[session_id]["messages"])

        return {
            "session_id": session_id,
            "character": {
                "name": char.get("name"),
                "age": char.get("age"),
                "personality": char.get("personality"),
                "appearance": char.get("appearance"),
                "nsfw_enabled": char.get("nsfw_enabled"),
            },
            "messages": self._sessions[session_id]["messages"],
            "is_new": len(existing_messages) == 0,
        }

    def update_character(self, session_id: str, character: Dict[str, Any]) -> bool:
        """Update character settings for a session."""
        if session_id in self._sessions:
            self._sessions[session_id]["character"].update(character)
            # Rebuild system prompt will happen on next message
            return True
        return False

    async def send_message(
        self,
        session_id: str,
        user_message: str,
    ) -> Dict[str, Any]:
        """Send a message and get AI response."""
        if session_id not in self._sessions:
            await self.create_session(session_id=session_id)

        session = self._sessions[session_id]
        char = session["character"]

        # Add user message
        user_msg = {
            "role": "user",
            "content": user_message,
            "timestamp": time.time(),
        }
        session["messages"].append(user_msg)

        # Build messages for LLM
        system_prompt = self._build_system_prompt(char)

        # Get last N messages for context (keep system prompt + last 30 messages)
        history = session["messages"][-30:]
        llm_messages = [{"role": "system", "content": system_prompt}]
        for msg in history:
            llm_messages.append({
                "role": msg["role"],
                "content": msg["content"],
            })

        # Get AI response
        ai_response = await self._get_ai_response(llm_messages)

        # Add assistant response
        assistant_msg = {
            "role": "assistant",
            "content": ai_response,
            "timestamp": time.time(),
        }
        session["messages"].append(assistant_msg)
        session["updated_at"] = time.time()

        # Save to disk
        state_manager.save_chat(session_id, session["messages"])

        return {
            "session_id": session_id,
            "response": ai_response,
            "character_name": char.get("name", "Assistant"),
            "message_count": len(session["messages"]),
        }

    async def _get_ai_response(self, messages: List[Dict]) -> str:
        """Get AI response using z-ai-web-dev-sdk or fallback."""
        try:
            zai = await self._get_zai()
            if zai is not None:
                completion = await zai.chat.completions.create(
                    messages=messages,
                    temperature=0.8,
                    max_tokens=500,
                )
                return completion.choices[0].message.content
        except Exception as e:
            logger.error(f"ZAI SDK error: {e}")

        # Fallback: simple rule-based responses
        return self._fallback_response(messages)

    def _fallback_response(self, messages: List[Dict]) -> str:
        """Simple fallback when AI SDK is not available."""
        last_msg = ""
        for m in reversed(messages):
            if m["role"] == "user":
                last_msg = m["content"].lower()
                break

        # Get character name
        char_name = "Aria"
        for m in messages:
            if m["role"] == "system":
                content = m["content"]
                if "You are " in content:
                    char_name = content.split("You are ")[1].split(",")[0].strip()
                break

        responses = {
            "hello": f"Hey there~ 😊 How's it going?",
            "hi": f"Hi~ 😄 Nice to see you!",
            "how are you": f"I'm doing great, thanks for asking! 💕 How about you?",
            "bored": f"Aww, let's chat then! Tell me something interesting~ 🎭",
            "love": f"Ooh~ 😳 That's sweet of you to say! 💕",
            "beautiful": f"You're making me blush~ 😊💕 Thanks!",
            "sexy": f"Oh my~ 😏 You're quite bold, aren't you? I like that~",
        }

        for key, resp in responses.items():
            if key in last_msg:
                return resp

        generic = [
            f"Interesting~ Tell me more! 😊",
            f"Hmm, I see what you mean~ 💭",
            f"That's cool! 😄 While we chat, your video is probably generating~",
            f"I love chatting with you~ 💕",
            f"Oh? Go on~ I'm listening 🤗",
            f"That's fun! What else is on your mind? ✨",
            f"Hehe~ You're quite interesting, you know that? 😏",
            f"I could chat with you all day~ 💕",
        ]

        import random
        return random.choice(generic)

    def get_session(self, session_id: str) -> Optional[Dict]:
        """Get session info."""
        if session_id in self._sessions:
            s = self._sessions[session_id]
            return {
                "session_id": s["session_id"],
                "character": {
                    "name": s["character"].get("name"),
                    "age": s["character"].get("age"),
                    "nsfw_enabled": s["character"].get("nsfw_enabled"),
                },
                "messages": s["messages"],
                "message_count": len(s["messages"]),
            }
        return None

    def list_sessions(self) -> List[Dict]:
        """List all chat sessions."""
        return state_manager.list_chats()

    def delete_session(self, session_id: str) -> bool:
        """Delete a chat session."""
        if session_id in self._sessions:
            del self._sessions[session_id]
        return state_manager.delete_chat(session_id)


# Global instance
chat_manager = ChatManager()
