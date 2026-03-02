import re
import random
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from assessment.hint_generator import HintGenerator, HintLevel
from assessment.error_analyzer import ErrorAnalyzer, ErrorType, ErrorDiagnosis
from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class AssessmentResult:
    response: str
    assessment_type: str  # 'hint', 'feedback', 'evaluation'
    is_correct: Optional[bool] = None
    errors: Optional[List[ErrorDiagnosis]] = None
    hint_level: Optional[HintLevel] = None


class AssessmentEngine:
    def __init__(self, llm_generator=None):
        self.hint_generator = HintGenerator(llm_generator)
        self.error_analyzer = ErrorAnalyzer(llm_generator)
        self.llm_generator = llm_generator
        logger.info("AssessmentEngine initialized")

    def set_llm_generator(self, llm_generator) -> None:
        self.llm_generator = llm_generator
        self.hint_generator.set_llm_generator(llm_generator)
        self.error_analyzer.set_llm_generator(llm_generator)

    def provide_hint(self, question: str, student_response: Optional[str] = None,
                     hint_level: str = "partial", context: Optional[str] = None,
                     previous_hints: Optional[List[str]] = None) -> AssessmentResult:
        logger.info(f"Providing {hint_level} hint")
        level = self.hint_generator.parse_hint_level(hint_level)
        hint = self.hint_generator.generate_hint(
            question=question, student_response=student_response,
            hint_level=level, context=context, previous_hints=previous_hints or [],
        )
        return AssessmentResult(response=hint, assessment_type='hint', hint_level=level)

    def assess_answer(self, question: str, student_answer: str,
                      context: Optional[str] = None, provide_feedback: bool = True) -> AssessmentResult:
        logger.info("Assessing student answer")
        errors = self.error_analyzer.analyze_answer(question=question, student_answer=student_answer, context=context)
        is_correct = self.error_analyzer.is_answer_correct(
            question=question,
            student_answer=student_answer,
            context=context,
            errors=errors,
        )
        if provide_feedback:
            feedback = self._generate_positive_feedback(student_answer, errors) if is_correct else \
                       self._generate_corrective_feedback(question, student_answer, errors, context)
        else:
            feedback = "Answer recorded."
        return AssessmentResult(response=feedback, assessment_type='evaluation', is_correct=is_correct, errors=errors)

    def ask_socratic_question(self, question: str, student_answer: str, context: Optional[str] = None) -> str:
        if not self.llm_generator:
            return self._fallback_socratic_question()
        prompt = self._build_socratic_prompt(question, student_answer, context)
        try:
            # Lower temperature for focused output
            raw = self.llm_generator.generate(prompt=prompt, temperature=0.5, max_new_tokens=30)
            # Clean hallucinated content before extracting question
            raw = self._clean_socratic_output(raw)
            # Extract just the question sentence — ignore preamble/formatting the model may produce
            return self._extract_question(raw) or self._fallback_socratic_question()
        except Exception as e:
            logger.error(f"Error generating Socratic question: {e}")
            return self._fallback_socratic_question()

    def _generate_positive_feedback(self, student_answer: str, errors: List[ErrorDiagnosis]) -> str:
        minor_issues = [e for e in errors if e.error_type != ErrorType.NONE and e.confidence < 0.7]
        if not minor_issues:
            return "Excellent work! Your answer demonstrates good understanding of the topic. You've provided clear reasoning and appropriate evidence."
        suggestion = self._clean_feedback_snippet(minor_issues[0].suggestion)
        return f"One small suggestion: {suggestion}"

    def _clean_feedback_snippet(self, text: str) -> str:
        snippet = (text or "").strip()
        # Remove common list-prefix artifacts from weak generations, e.g. "1.".
        snippet = re.sub(r'^\s*\d+\s*[.)-]?\s*', '', snippet)
        if len(snippet) < 8:
            return "Add one concrete historical detail (date, actor, or event) to strengthen your answer."
        return snippet

    def _generate_corrective_feedback(self, question: str, student_answer: str,
                                       errors: List[ErrorDiagnosis], context: Optional[str]) -> str:
        error_feedback = self.error_analyzer.provide_feedback(question=question, student_answer=student_answer, errors=errors)
        socratic_q = self.ask_socratic_question(question, student_answer, context)
        return f"{error_feedback}\n\n{socratic_q}\n\nWould you like a hint?"

    def _build_socratic_prompt(self, question: str, student_answer: str, context: Optional[str]) -> str:
        parts = [
            "You are a concise Socratic tutor.",
            f"\nQuestion: {question}",
            f"\nStudent's incorrect answer: {student_answer}",
        ]
        if context:
            parts.append(f"\nContext: {context[:200]}")
        parts.append(
            "\nCRITICAL RULES:\n"
            "- Write EXACTLY ONE short question (under 15 words)\n"
            "- Output ONLY the question — no preamble, no commentary\n"
            "- Do NOT give the answer\n"
            "- Do NOT hallucinate student responses\n"
            "- Do NOT add 'Student says...' or similar\n\n"
            "GOOD: 'What role did Prussian leadership play in the 1870s?'\n"
            "BAD: 'Student's response: ...' or 'Let me help you...'\n\n"
            "Guiding question:"
        )
        return "\n".join(parts)

    def _clean_socratic_output(self, raw: str) -> str:
        """Remove hallucinated student responses and meta-commentary from Socratic questions"""
        raw = re.sub(r"Student'?s? (response|attempt|answer):.*", "", raw, flags=re.IGNORECASE | re.DOTALL)
        raw = re.sub(r"(Please wait|Is this enough|Let me help|Now that you).*", "", raw, flags=re.IGNORECASE | re.DOTALL)
        raw = re.sub(r"(Tutor:|Student:|Question:)", "", raw, flags=re.IGNORECASE)
        return raw.strip()

    def _extract_question(self, raw: str) -> str:
        """Pick the shortest '?'-ending line from LLM output, discarding preamble."""
        lines = [l.strip() for l in raw.splitlines() if l.strip()]
        question_lines = [l for l in lines if l.endswith('?')]
        if question_lines:
            return min(question_lines, key=len)
        return lines[-1] if lines else ""

    def _fallback_socratic_question(self) -> str:
        return random.choice([
            "What evidence supports your reasoning?",
            "How would you explain this to someone unfamiliar with the topic?",
            "What are the key factors you should consider?",
            "Can you think of any counterexamples or alternative explanations?",
        ])

    def get_statistics(self) -> Dict[str, Any]:
        return {'llm_available': self.llm_generator is not None, 'components_active': {'hint_generator': True, 'error_analyzer': True}}
