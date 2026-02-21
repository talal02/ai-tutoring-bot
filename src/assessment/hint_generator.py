from typing import Optional
from enum import Enum
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from utils.logger import get_logger

logger = get_logger(__name__)


class HintLevel(Enum):
    NUDGE = "nudge"      # Very subtle — guiding question only
    PARTIAL = "partial"  # More direct — points to key concepts
    FULL = "full"        # Nearly complete — walks through reasoning


class HintGenerator:
    def __init__(self, llm_generator=None):
        self.llm_generator = llm_generator

    def set_llm_generator(self, llm_generator) -> None:
        self.llm_generator = llm_generator

    def generate_hint(self, question: str, student_response: Optional[str] = None,
                      hint_level: HintLevel = HintLevel.PARTIAL, context: Optional[str] = None,
                      previous_hints: Optional[list] = None) -> str:
        logger.info(f"Generating {hint_level.value} hint")
        if not self.llm_generator:
            return self._fallback_hint(hint_level)
        prompt = self._build_hint_prompt(question, student_response, hint_level, context, previous_hints)
        try:
            hint = self.llm_generator.generate(prompt=prompt, temperature=0.7, max_new_tokens=200)
            return hint.strip()
        except Exception as e:
            logger.error(f"Error generating hint: {e}")
            return self._fallback_hint(hint_level)

    def _build_hint_prompt(self, question: str, student_response: Optional[str],
                           hint_level: HintLevel, context: Optional[str],
                           previous_hints: Optional[list]) -> str:
        instructions = {
            HintLevel.NUDGE: (
                "Provide a VERY SUBTLE hint that nudges the student in the right direction. "
                "Don't reveal the answer. Ask a guiding question or point to something they should consider."
            ),
            HintLevel.PARTIAL: (
                "Provide a MODERATE hint that gives more direct guidance. "
                "Point to key concepts or steps they need to think about, but don't give the full answer."
            ),
            HintLevel.FULL: (
                "Provide a DETAILED hint that walks them through most of the solution. "
                "Explain the reasoning step-by-step, but let them complete the final step themselves."
            ),
        }
        parts = ["You are a patient history tutor helping a student.", f"\nQuestion: {question}"]
        if student_response:
            parts.append(f"\nStudent's attempt: {student_response}")
        if previous_hints:
            parts.append(f"\nPrevious hints given: {', '.join(previous_hints[-2:])}")
        if context:
            parts.append(f"\nRelevant information: {context[:500]}")
        parts.append(f"\n{instructions[hint_level]}\nHint:")
        return "\n".join(parts)

    def _fallback_hint(self, hint_level: HintLevel) -> str:
        fallbacks = {
            HintLevel.NUDGE: "Think about what you already know about this topic. What key concepts or time periods are relevant here?",
            HintLevel.PARTIAL: "Consider breaking this question into smaller parts. What are the main factors or causes you need to identify?",
            HintLevel.FULL: "Let's approach this step by step. First, identify the time period. Then think about the key events and their consequences. Finally, connect these to answer the question.",
        }
        return fallbacks.get(hint_level, "Think carefully about the question and try again.")

    def parse_hint_level(self, level_str: str) -> HintLevel:
        level_map = {
            'nudge': HintLevel.NUDGE, 'small': HintLevel.NUDGE,
            'partial': HintLevel.PARTIAL, 'medium': HintLevel.PARTIAL,
            'full': HintLevel.FULL, 'large': HintLevel.FULL, 'big': HintLevel.FULL,
        }
        return level_map.get(level_str.lower(), HintLevel.PARTIAL)
