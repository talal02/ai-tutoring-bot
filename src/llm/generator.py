"""Text generation module for LLM-based tutoring responses."""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List, Dict, Optional, Union
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))

from utils.config import LLMConfig, PromptsConfig
from utils.logger import get_logger

logger = get_logger(__name__)


class LLMGenerator:
    """Handles text generation with prompt formatting."""

    def __init__(
        self,
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        llm_config: LLMConfig,
        prompts_config: PromptsConfig,
    ):
        """Initialize generator."""
        self.model = model
        self.tokenizer = tokenizer
        self.llm_config = llm_config
        self.prompts_config = prompts_config
        self.conversation_history: List[Dict[str, str]] = []

        logger.info("LLMGenerator initialized")

    def format_chat_prompt(
        self,
        user_message: str,
        system_prompt: Optional[str] = None,
        context: Optional[str] = None,
        history: Optional[List[Dict[str, str]]] = None,
    ) -> str:
        """Format messages into a chat prompt."""
        if system_prompt is None:
            system_prompt = self.prompts_config.system_prompt

        messages = []

        # Add system message
        messages.append({
            "role": "system",
            "content": system_prompt,
        })

        # Add conversation history
        if history:
            messages.extend(history)

        # Add context if provided
        user_content = user_message
        if context:
            user_content = f"Context:\n{context}\n\nQuestion: {user_message}"

        messages.append({
            "role": "user",
            "content": user_content,
        })

        # Use tokenizer's chat template if available
        if hasattr(self.tokenizer, 'apply_chat_template'):
            try:
                prompt = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
                return prompt
            except Exception as e:
                logger.warning(f"Chat template failed: {e}. Using fallback.")

        # Fallback formatting
        prompt_parts = []
        for msg in messages:
            role = msg["role"]
            content = msg["content"]

            if role == "system":
                prompt_parts.append(f"<|system|>\n{content}\n")
            elif role == "user":
                prompt_parts.append(f"<|user|>\n{content}\n")
            elif role == "assistant":
                prompt_parts.append(f"<|assistant|>\n{content}\n")

        prompt_parts.append("<|assistant|>\n")
        return "".join(prompt_parts)

    def generate(
        self,
        prompt: str,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        do_sample: Optional[bool] = None,
        **kwargs,
    ) -> str:
        """
        Generate text from a prompt.

        Args:
            prompt: Input prompt.
            max_new_tokens: Maximum tokens to generate.
            temperature: Sampling temperature.
            top_p: Nucleus sampling parameter.
            top_k: Top-k sampling parameter.
            do_sample: Whether to use sampling.
            **kwargs: Additional generation parameters.

        Returns:
            Generated text.
        """
        # Use config defaults if not specified
        gen_config = self.llm_config.generation.copy()
        if max_new_tokens is not None:
            gen_config["max_new_tokens"] = max_new_tokens
        if temperature is not None:
            gen_config["temperature"] = temperature
        if top_p is not None:
            gen_config["top_p"] = top_p
        if top_k is not None:
            gen_config["top_k"] = top_k
        if do_sample is not None:
            gen_config["do_sample"] = do_sample

        # Override with kwargs
        gen_config.update(kwargs)

        logger.debug(f"Generating with config: {gen_config}")

        try:
            # Tokenize input
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=2048,
            ).to(self.model.device)

            # Generate
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    **gen_config,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )

            # Decode output
            generated_text = self.tokenizer.decode(
                outputs[0][inputs['input_ids'].shape[1]:],
                skip_special_tokens=True,
            )

            logger.debug(f"Generated {len(generated_text)} characters")
            return generated_text.strip()

        except Exception as e:
            logger.error(f"Generation failed: {str(e)}", exc_info=True)
            raise

    def generate_with_rag(
        self,
        question: str,
        context: str,
        **kwargs,
    ) -> str:
        """
        Generate response using RAG context.

        Args:
            question: Student's question.
            context: Retrieved context from RAG.
            **kwargs: Additional generation parameters.

        Returns:
            Generated response.
        """
        prompt = self.prompts_config.rag_prompt_template.format(
            context=context,
            question=question,
        )

        full_prompt = self.format_chat_prompt(
            user_message=prompt,
            history=self.conversation_history[-6:] if self.conversation_history else None,
        )

        response = self.generate(full_prompt, **kwargs)

        # Update conversation history
        self.add_to_history(question, response)

        return response

    def generate_hint(
        self,
        question: str,
        student_response: str,
        hint_level: str = "nudge",
        **kwargs,
    ) -> str:
        """
        Generate a pedagogical hint.

        Args:
            question: The question student is working on.
            student_response: Student's current answer attempt.
            hint_level: Type of hint (nudge, partial, full).
            **kwargs: Additional generation parameters.

        Returns:
            Generated hint.
        """
        if hint_level not in ["nudge", "partial", "full"]:
            logger.warning(f"Invalid hint level: {hint_level}, using 'nudge'")
            hint_level = "nudge"

        prompt = self.prompts_config.hint_prompt_template.format(
            question=question,
            student_response=student_response,
            hint_level=hint_level,
        )

        full_prompt = self.format_chat_prompt(user_message=prompt)

        # Use lower temperature for hints to be more consistent
        kwargs.setdefault("temperature", 0.5)

        return self.generate(full_prompt, **kwargs)

    def generate_batch(
        self,
        prompts: List[str],
        batch_size: int = 4,
        **kwargs,
    ) -> List[str]:
        """
        Generate responses for multiple prompts.

        Args:
            prompts: List of input prompts.
            batch_size: Batch size for generation.
            **kwargs: Additional generation parameters.

        Returns:
            List of generated responses.
        """
        responses = []

        for i in range(0, len(prompts), batch_size):
            batch_prompts = prompts[i:i + batch_size]
            logger.debug(f"Processing batch {i // batch_size + 1}")

            for prompt in batch_prompts:
                response = self.generate(prompt, **kwargs)
                responses.append(response)

        return responses

    def add_to_history(
        self,
        user_message: str,
        assistant_message: str,
    ) -> None:
        """
        Add exchange to conversation history.

        Args:
            user_message: User's message.
            assistant_message: Assistant's response.
        """
        self.conversation_history.append({
            "role": "user",
            "content": user_message,
        })
        self.conversation_history.append({
            "role": "assistant",
            "content": assistant_message,
        })

        # Limit history length
        max_history = 20  # 10 exchanges
        if len(self.conversation_history) > max_history:
            self.conversation_history = self.conversation_history[-max_history:]

    def clear_history(self) -> None:
        """Clear conversation history."""
        self.conversation_history = []
        logger.debug("Cleared conversation history")

    def get_history(self) -> List[Dict[str, str]]:
        """Get current conversation history."""
        return self.conversation_history.copy()

    def count_tokens(self, text: str) -> int:
        """
        Count tokens in text.

        Args:
            text: Input text.

        Returns:
            Number of tokens.
        """
        return len(self.tokenizer.encode(text))

    def estimate_generation_cost(
        self,
        prompt: str,
        max_new_tokens: int = 512,
    ) -> Dict[str, int]:
        """
        Estimate token usage for generation.

        Args:
            prompt: Input prompt.
            max_new_tokens: Maximum tokens to generate.

        Returns:
            Dictionary with input/output token estimates.
        """
        input_tokens = self.count_tokens(prompt)

        return {
            "input_tokens": input_tokens,
            "max_output_tokens": max_new_tokens,
            "total_max_tokens": input_tokens + max_new_tokens,
        }
