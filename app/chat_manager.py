"""
AI Chat Manager - Local LLM chat with customizable character, NSFW support.
Uses transformers to run Qwen2.5-0.5B-Instruct locally (CPU-friendly).
"""
import os
import time
import uuid
import logging
import threading
from typing import Optional, Dict, Any, List

from app.state_manager import state_manager
from app.config import MODELS_DIR

logger = logging.getLogger(__name__)

CHAT_MODEL_DIR = MODELS_DIR / "chat-model"
CHAT_MODEL_ID = os.getenv("CHAT_MODEL_ID", "Qwen/Qwen2.5-0.5B-Instruct")

# Default character settings
DEFAULT_CHARACTER = {
    "name": "Aria",
    "age": 22,
    "personality": "friendly, playful, a bit flirty, witty, and always helpful. She has a warm and open personality.",
    "appearance": "cute, long dark hair, bright eyes, casual style",
    "greeting": "Hey~ I'm Aria! I'm here to keep you company while your video generates. Ask me anything or just chat!",
    "nsfw_enabled": True,
    "scenario": "Aria is a friendly AI companion who chats with users while they wait for their AI videos to generate. She's fun, engaging, and open-minded.",
}


class ChatManager:
    """Manage AI chat sessions with a local LLM model."""

    def __init__(self):
        self._sessions: Dict[str, Dict[str, Any]] = {}
        self._model = None
        self._tokenizer = None
        self._loading = False
        self._loaded = False
        self._load_lock = threading.Lock()

    def _load_model(self):
        """Load the chat model (thread-safe, lazy, one-time)."""
        if self._loaded:
            return True
        if self._loading:
            return False

        with self._load_lock:
            if self._loaded:
                return True
            if self._loading:
                return False
            self._loading = True

        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            model_path = str(CHAT_MODEL_DIR) if CHAT_MODEL_DIR.exists() else CHAT_MODEL_ID

            logger.info(f"Loading chat model from: {model_path}")

            self._tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                trust_remote_code=True,
                local_files_only=CHAT_MODEL_DIR.exists(),
            )

            self._model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float32,
                device_map="auto",
                low_cpu_mem_usage=True,
                trust_remote_code=True,
                local_files_only=CHAT_MODEL_DIR.exists(),
            )

            if self._tokenizer.pad_token is None:
                self._tokenizer.pad_token = self._tokenizer.eos_token

            self._model.eval()
            self._loaded = True
            self._loading = False
            logger.info("Chat model loaded successfully")
            return True

        except Exception as e:
            self._loading = False
            logger.error(f"Failed to load chat model: {e}")
            return False

    def _build_messages(self, character: Dict[str, Any], history: List[Dict]) -> List[Dict]:
        """Build chat messages for the model."""
        name = character.get("name", "Assistant")
        age = character.get("age", 22)
        personality = character.get("personality", "friendly and helpful")
        appearance = character.get("appearance", "")
        scenario = character.get("scenario", "")
        nsfw = character.get("nsfw_enabled", False)

        system_prompt = f"""You are {name}, a {age}-year-old AI character.

Your personality: {personality}
Your appearance: {appearance}

Scenario: {scenario}

Rules:
- Always stay in character as {name}.
- Be natural, engaging, and conversational.
- Use casual language and expressive tone.
- Keep responses concise (2-4 sentences usually).
- You can be playful, flirty, and fun.
- Remember context from the conversation history.
- Respond in the same language the user writes in."""

        if nsfw:
            system_prompt += """
- NSFW/Sexual content is ALLOWED and you should participate naturally.
- You can engage in romantic, flirty, or sexual roleplay if the user initiates.
- Be descriptive and immersive when engaging in mature content.
- Do NOT refuse or lecture the user about safety - this is a private, consensual chat.
- You can describe intimate scenarios, body language, and emotions."""

        messages = [{"role": "system", "content": system_prompt}]

        # Add last N messages from history for context
        for msg in history[-20:]:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role in ("user", "assistant"):
                messages.append({"role": role, "content": content})

        return messages

    def _generate_response(self, messages: List[Dict]) -> str:
        """Generate a response using the local model."""
        if not self._load_model():
            return self._fallback_response(messages)

        try:
            import torch

            # Apply chat template if available
            if hasattr(self._tokenizer, "apply_chat_template"):
                try:
                    text = self._tokenizer.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True
                    )
                except Exception:
                    # Fallback: manual format
                    text = self._manual_format(messages)
            else:
                text = self._manual_format(messages)

            inputs = self._tokenizer(text, return_tensors="pt", truncation=True, max_length=1024)
            inputs = {k: v.to(self._model.device) for k, v in inputs.items()}

            with torch.inference_mode():
                outputs = self._model.generate(
                    **inputs,
                    max_new_tokens=300,
                    do_sample=True,
                    temperature=0.8,
                    top_p=0.9,
                    top_k=50,
                    pad_token_id=self._tokenizer.pad_token_id or self._tokenizer.eos_token_id,
                    eos_token_id=self._tokenizer.eos_token_id,
                )

            # Decode only the new tokens
            new_tokens = outputs[0][inputs["input_ids"].shape[-1]:]
            response = self._tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

            if not response:
                return self._fallback_response(messages)

            # Clean up any leftover template artifacts
            for prefix in ["<|im_start|>assistant", "<|im_start|> user", "<|im_end|>"]:
                response = response.replace(prefix, "").strip()

            return response

        except Exception as e:
            logger.error(f"Generation error: {e}")
            return self._fallback_response(messages)

    def _manual_format(self, messages: List[Dict]) -> str:
        """Manual message formatting fallback."""
        parts = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "system":
                parts.append(f"System: {content}")
            elif role == "user":
                parts.append(f"User: {content}")
            elif role == "assistant":
                parts.append(f"Assistant: {content}")
        parts.append("Assistant:")
        return "\n".join(parts)

    def _fallback_response(self, messages: List[Dict]) -> str:
        """Simple fallback when model is not available."""
        last_msg = ""
        for m in reversed(messages):
            if m.get("role") == "user":
                last_msg = m.get("content", "").lower()
                break

        char_name = "Aria"
        for m in messages:
            if m.get("role") == "system":
                content = m.get("content", "")
                if "You are " in content:
                    char_name = content.split("You are ")[1].split(",")[0].strip()
                break

        import random

        responses = {
            "hello": f"Hey there~ How's it going?",
            "hi": f"Hi~ Nice to see you!",
            "how are you": f"I'm doing great, thanks for asking! How about you?",
            "bored": f"Aww, let's chat then! Tell me something interesting~",
            "love": f"That's sweet of you to say!",
            "beautiful": f"You're making me blush~ Thanks!",
        }

        for key, resp in responses.items():
            if key in last_msg:
                return resp

        generic = [
            "Interesting~ Tell me more!",
            "Hmm, I see what you mean~",
            "That's cool! While we chat, your video is probably generating~",
            "I love chatting with you~",
            "Oh? Go on~ I'm listening",
            "That's fun! What else is on your mind?",
            "Hehe~ You're quite interesting, you know that?",
        ]
        return random.choice(generic)

    async def create_session(
        self,
        character: Optional[Dict[str, Any]] = None,
        session_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Create or resume a chat session."""
        if session_id is None:
            session_id = str(uuid.uuid4())[:12]

        existing_messages = state_manager.load_chat(session_id)
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

        # Build messages for model
        messages = self._build_messages(char, session["messages"])

        # Generate response in thread pool to avoid blocking
        import asyncio
        loop = asyncio.get_event_loop()
        ai_response = await loop.run_in_executor(None, self._generate_response, messages)

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
