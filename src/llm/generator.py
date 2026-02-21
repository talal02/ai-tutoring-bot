import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List, Dict, Optional
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))

from utils.config import LLMConfig, PromptsConfig
from utils.logger import get_logger

logger = get_logger(__name__)


class LLMGenerator:
    def __init__(self, model: AutoModelForCausalLM, tokenizer: AutoTokenizer,
                 llm_config: LLMConfig, prompts_config: PromptsConfig):
        self.model = model
        self.tokenizer = tokenizer
        self.llm_config = llm_config
        self.prompts_config = prompts_config
        self.conversation_history: List[Dict[str, str]] = []
        logger.info("LLMGenerator initialized")

    def format_chat_prompt(self, user_message: str, system_prompt: Optional[str] = None,
                           context: Optional[str] = None,
                           history: Optional[List[Dict[str, str]]] = None) -> str:
        system_prompt = system_prompt or self.prompts_config.system_prompt
        messages = [{"role": "system", "content": system_prompt}]
        if history:
            messages.extend(history)
        user_content = f"Context:\n{context}\n\nQuestion: {user_message}" if context else user_message
        messages.append({"role": "user", "content": user_content})

        if hasattr(self.tokenizer, 'apply_chat_template'):
            try:
                return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            except Exception as e:
                logger.warning(f"Chat template failed: {e}. Using fallback.")

        # Fallback for tokenizers without a chat template
        parts = []
        for msg in messages:
            role, content = msg["role"], msg["content"]
            parts.append(f"<|{role}|>\n{content}\n")
        parts.append("<|assistant|>\n")
        return "".join(parts)

    def generate(self, prompt: str, max_new_tokens: Optional[int] = None,
                 temperature: Optional[float] = None, top_p: Optional[float] = None,
                 top_k: Optional[int] = None, do_sample: Optional[bool] = None, **kwargs) -> str:
        gen_config = self.llm_config.generation.copy()
        overrides = {"max_new_tokens": max_new_tokens, "temperature": temperature,
                     "top_p": top_p, "top_k": top_k, "do_sample": do_sample}
        gen_config.update({k: v for k, v in overrides.items() if v is not None})
        gen_config.update(kwargs)

        try:
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(self.model.device)
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs, **gen_config,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )
            # Slice off the prompt tokens — only decode the newly generated portion
            generated_text = self.tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
            return generated_text.strip()
        except Exception as e:
            logger.error(f"Generation failed: {e}", exc_info=True)
            raise

    def generate_with_rag(self, question: str, context: str, **kwargs) -> str:
        prompt = self.prompts_config.rag_prompt_template.format(context=context, question=question)
        full_prompt = self.format_chat_prompt(
            user_message=prompt,
            history=self.conversation_history[-6:] if self.conversation_history else None,
        )
        response = self.generate(full_prompt, **kwargs)
        self.add_to_history(question, response)
        return response

    def generate_hint(self, question: str, student_response: str, hint_level: str = "nudge", **kwargs) -> str:
        if hint_level not in ("nudge", "partial", "full"):
            logger.warning(f"Invalid hint level '{hint_level}', using 'nudge'")
            hint_level = "nudge"
        prompt = self.prompts_config.hint_prompt_template.format(
            question=question, student_response=student_response, hint_level=hint_level,
        )
        # Lower temperature for more consistent hints
        kwargs.setdefault("temperature", 0.5)
        return self.generate(self.format_chat_prompt(user_message=prompt), **kwargs)

    def add_to_history(self, user_message: str, assistant_message: str) -> None:
        self.conversation_history.append({"role": "user", "content": user_message})
        self.conversation_history.append({"role": "assistant", "content": assistant_message})
        # Keep last 10 exchanges (20 messages)
        if len(self.conversation_history) > 20:
            self.conversation_history = self.conversation_history[-20:]

    def clear_history(self) -> None:
        self.conversation_history = []

    def get_history(self) -> List[Dict[str, str]]:
        return self.conversation_history.copy()
