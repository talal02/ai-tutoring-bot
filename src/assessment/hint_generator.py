import re
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
            max_tokens = {HintLevel.NUDGE: 40, HintLevel.PARTIAL: 60, HintLevel.FULL: 80}
            hint = self.llm_generator.generate(prompt=prompt, temperature=0.5, max_new_tokens=max_tokens[hint_level])

            hint = self._clean_hint_output(hint, hint_level)
            return hint.strip()
        except Exception as e:
            logger.error(f"Error generating hint: {e}")
            return self._fallback_hint(hint_level)

    def _build_hint_prompt(self, question: str, student_response: Optional[str],
                           hint_level: HintLevel, context: Optional[str],
                           previous_hints: Optional[list]) -> str:
        instructions = {
            HintLevel.NUDGE: (
                "CRITICAL RULES:\n"
                "- Write EXACTLY ONE sentence\n"
                "- Maximum 20 words\n"
                "- Ask a broad guiding question OR point to general time period/role\n"
                "- NEVER mention specific names or answers\n"
                "- NO conversational filler\n"
                "- Output ONLY the hint sentence\n\n"
                "GOOD: 'Think about Prussian political leaders in the 1860s-1870s.'\n"
                "BAD: 'Otto von Bismarck' or 'Let me help you...' or hallucinating responses"
            ),
            HintLevel.PARTIAL: (
                "CRITICAL RULES:\n"
                "- Write EXACTLY ONE sentence\n"
                "- Maximum 25 words\n"
                "- You MAY mention related events/concepts but NOT the final answer\n"
                "- NO conversational filler or meta-commentary\n"
                "- Output ONLY the hint sentence\n\n"
                "GOOD: 'Consider the role of Prussian leadership in the Franco-Prussian War.'\n"
                "BAD: Stating the person's name directly or 'Is this enough?'"
            ),
            HintLevel.FULL: (
                "CRITICAL RULES:\n"
                "- Write 2-3 sentences MAXIMUM\n"
                "- Total under 40 words\n"
                "- Explain the process but DO NOT state the person's name directly\n"
                "- NO meta-commentary, NO hallucinated conversations\n"
                "- Output ONLY the hint sentences\n\n"
                "GOOD: 'The Prussian Prime Minister orchestrated wars with Denmark, Austria, and France. "
                "These military victories led to unification proclaimed at Versailles in 1871.'\n"
                "BAD: Including 'Otto von Bismarck' directly or 'Student response: ...'"
            ),
        }
        parts = [
            "You are a hint generator. Output ONLY the hint text.",
            f"\nQuestion: {question}"
        ]
        if student_response:
            parts.append(f"\nStudent's wrong answer: {student_response}")
        if previous_hints:
            recent_hints = previous_hints[-2:]
            parts.append("\nPrevious hints (avoid repeating these):")
            for hint in recent_hints:
                parts.append(f"- {hint}")
        if context:
            parts.append(f"\nContext: {context[:200]}")
        parts.append(f"\n{instructions[hint_level]}\n\nHint:")
        return "\n".join(parts)

    def _clean_hint_output(self, raw_hint: str, hint_level: HintLevel) -> str:
        """Remove hallucinated student responses and meta-commentary"""
        raw_hint = re.sub(r"Student'?s? (response|attempt|answer):.*", "", raw_hint, flags=re.IGNORECASE | re.DOTALL)
        raw_hint = re.sub(r"(Please wait|Is this enough|Let me help you|Now that you|Should I).*", "", raw_hint, flags=re.IGNORECASE | re.DOTALL)
        raw_hint = re.sub(r"Answer:.*", "", raw_hint, flags=re.IGNORECASE | re.DOTALL)
        raw_hint = re.sub(r"\([^)]*(note|answer|part\s*\d+|student|hint)[^)]*\)", "", raw_hint, flags=re.IGNORECASE)
        raw_hint = re.sub(r"(Tutor:|Student:|Hint \d+:)", "", raw_hint, flags=re.IGNORECASE)
        raw_hint = re.sub(r"\s+", " ", raw_hint).strip()
        sentences = [s.strip() for s in re.split(r'[.!?]+', raw_hint) if s.strip()]
        if not sentences:
            return self._fallback_hint(hint_level)

        max_sentences = 1 if hint_level == HintLevel.NUDGE else (2 if hint_level == HintLevel.PARTIAL else 3)
        cleaned = '. '.join(sentences[:max_sentences])
        return cleaned + '.' if cleaned and not cleaned.endswith('.') else cleaned

    def _fallback_hint(self, hint_level: HintLevel) -> str:
        fallbacks = {
            HintLevel.NUDGE: "Think about the time period and key political figures involved.",
            HintLevel.PARTIAL: "Consider the main events, leaders, and their motivations during that era.",
            HintLevel.FULL: "Break this into steps: identify the time period, key leaders, major events, and their consequences.",
        }
        return fallbacks.get(hint_level, "Think carefully and try again.")

    def parse_hint_level(self, level_str: str) -> HintLevel:
        level_map = {
            'nudge': HintLevel.NUDGE, 'small': HintLevel.NUDGE,
            'partial': HintLevel.PARTIAL, 'medium': HintLevel.PARTIAL,
            'full': HintLevel.FULL, 'large': HintLevel.FULL, 'big': HintLevel.FULL,
        }
        return level_map.get(level_str.lower(), HintLevel.PARTIAL)
