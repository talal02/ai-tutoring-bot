import re
from typing import List, Optional
from enum import Enum
from dataclasses import dataclass
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from utils.logger import get_logger

logger = get_logger(__name__)


class ErrorType(Enum):
    PRESENTISM = "presentism"          # Judging past by present standards
    ANACHRONISM = "anachronism"        # Wrong time period
    OVERSIMPLIFICATION = "oversimplification"
    LACK_OF_EVIDENCE = "lack_of_evidence"
    FALSE_CAUSATION = "false_causation"
    BIAS = "bias"
    NONE = "none"


@dataclass
class ErrorDiagnosis:
    error_type: ErrorType
    confidence: float
    explanation: str
    suggestion: str


class ErrorAnalyzer:
    PRESENTISM_PATTERNS = [
        r'\b(should have|ought to have|why didn\'?t they just)\b',
        r'\b(obviously|clearly|simple|easy)\b.*\b(solution|answer)\b',
        r'\b(modern|today|nowadays|current)\b.*\b(standards|values|perspective)\b',
    ]

    # Post-1980 technology — anachronistic in any pre-1940 historical context
    _MODERN_TECH_RE = re.compile(
        r'\b(social\s+media|facebook|twitter|instagram|tiktok|youtube|snapchat|'
        r'smart\s*phone|cell\s*phone|mobile\s*phone|iphone|android|'
        r'internet|world\s+wide\s+web|website|e-?mail|online\s+communit|'
        r'laptop|tablet|ipad|microchip|microprocessor|'
        r'satellite\s+communication|gps\b|drone)\b',
        re.IGNORECASE,
    )
    # Post-1940 technology — anachronistic in WWI / early 20th-century contexts
    _POST_1940_TECH_RE = re.compile(
        r'\b(television|tv\s+(show|broadcast|news|program)|'
        r'atomic\s+bomb|nuclear\s+weapon|hydrogen\s+bomb|nuclear\s+power)\b',
        re.IGNORECASE,
    )
    # Detect pre-1940 historical context from question + RAG context
    _PRE_1940_PERIOD_RE = re.compile(
        r'\b(world\s+war\s+(i{1,3}|1|2|one|two)|wwi{1,2}|ww[12]|'
        r'first\s+world\s+war|second\s+world\s+war|'
        r'19th\s+century|1800s|18\d\ds|'
        r'ancient|medieval|middle\s+ages?|roman\s+empire|greek\s+(empire|city|civilization)|'
        r'ottoman\s+empire|byzantine|feudal|'
        r'civil\s+war|american\s+revolution|french\s+revolution|napoleonic|'
        r'victorian|edwardian)\b',
        re.IGNORECASE,
    )
    # Narrower WWI-era marker — triggers the post-1940 tech check as well
    _PRE_1920_PERIOD_RE = re.compile(
        r'\b(world\s+war\s+(i|1|one)|wwi\b|ww1\b|first\s+world\s+war|'
        r'191[0-9]|archduke|franz\s+ferdinand|'
        r'triple\s+(alliance|entente)|austro[-\s]hungarian|'
        r'trench\s+warfare|trench(es)?)\b',
        re.IGNORECASE,
    )

    OVERSIMPLIFICATION_PATTERNS = [
        r'\bthe\s+(only|sole)\s+(reason|cause|factor|explanation)\s+(was|is|were|are)\b',
        r'\b(simply|merely|just)\s+because\b',
        r'\ba\s+single\s+(cause|reason|factor)\b',
    ]

    LACK_OF_EVIDENCE_PATTERNS = [
        r'\b(must have|might have|could have)\b.*without',
    ]

    # Used by _check_oversimplification to skip negated sentences like
    _NEGATION_RE = re.compile(
        r'\b(no|not|never|nor|without|wasn\'t|isn\'t|aren\'t|weren\'t|didn\'t|don\'t|cannot|can\'t)\b',
        re.IGNORECASE,
    )
    # Hedge-only answers flagged as lacking evidence when short (< 20 words)
    _HEDGE_RE = re.compile(r'^(i\s+think|i\s+believe|i\s+feel|maybe|probably)\b', re.IGNORECASE)

    def __init__(self, llm_generator=None):
        self.llm_generator = llm_generator

    def set_llm_generator(self, llm_generator) -> None:
        self.llm_generator = llm_generator

    def analyze_answer(self, question: str, student_answer: str, context: Optional[str] = None) -> List[ErrorDiagnosis]:
        logger.info("Analyzing student answer for errors")
        errors = []
        for check in (
            self._check_presentism(student_answer),
            self._check_anachronism(question, student_answer, context),
            self._check_oversimplification(student_answer),
            self._check_lack_of_evidence(student_answer),
        ):
            if check:
                errors.append(check)

        if not errors and self.llm_generator:
            llm_diag = self._llm_error_check(question, student_answer, context)
            if llm_diag:
                errors.append(llm_diag)

        if not errors:
            errors.append(ErrorDiagnosis(
                error_type=ErrorType.NONE, confidence=0.8,
                explanation="No major errors detected in your reasoning.",
                suggestion="Good work! Consider adding more detail or evidence to strengthen your answer.",
            ))
        logger.debug(f"Found {len(errors)} error(s)")
        return errors

    def _check_presentism(self, answer: str) -> Optional[ErrorDiagnosis]:
        for pattern in self.PRESENTISM_PATTERNS:
            if re.search(pattern, answer, re.IGNORECASE):
                return ErrorDiagnosis(
                    error_type=ErrorType.PRESENTISM, confidence=0.7,
                    explanation="Your answer may be judging historical events by modern standards (presentism). Historical actors had different knowledge, values, and constraints.",
                    suggestion="Try to understand the historical context and the perspectives of people at that time. What did they know? What were their beliefs and options?",
                )
        return None

    def _check_oversimplification(self, answer: str) -> Optional[ErrorDiagnosis]:
        for pattern in self.OVERSIMPLIFICATION_PATTERNS:
            match = re.search(pattern, answer, re.IGNORECASE)
            if match:
                sent_start = answer.rfind('.', 0, match.start())
                sent_end = answer.find('.', match.end())
                sentence = answer[sent_start + 1: sent_end if sent_end >= 0 else None].strip()
                if self._NEGATION_RE.search(sentence):
                    continue
                return ErrorDiagnosis(
                    error_type=ErrorType.OVERSIMPLIFICATION, confidence=0.7,
                    explanation="Your answer may be oversimplifying a complex historical issue. Most historical events have multiple interconnected causes.",
                    suggestion="Consider multiple perspectives, causes, and consequences. What were the economic, political, social, and cultural factors involved?",
                )
        return None

    def _check_lack_of_evidence(self, answer: str) -> Optional[ErrorDiagnosis]:
        words = answer.split()
        if len(words) < 8:
            return ErrorDiagnosis(
                error_type=ErrorType.LACK_OF_EVIDENCE, confidence=0.6,
                explanation="Your answer seems brief and may lack sufficient evidence or detail.",
                suggestion="Support your claims with specific historical evidence: dates, events, sources, or examples.",
            )
        if len(words) < 20 and self._HEDGE_RE.search(answer):
            return ErrorDiagnosis(
                error_type=ErrorType.LACK_OF_EVIDENCE, confidence=0.65,
                explanation="Your answer relies on speculation without citing supporting evidence.",
                suggestion="Back up your reasoning with specific historical evidence from primary or secondary sources.",
            )
        for pattern in self.LACK_OF_EVIDENCE_PATTERNS:
            if re.search(pattern, answer, re.IGNORECASE):
                return ErrorDiagnosis(
                    error_type=ErrorType.LACK_OF_EVIDENCE, confidence=0.65,
                    explanation="Your answer relies on speculation without citing evidence.",
                    suggestion="Back up your reasoning with specific historical evidence.",
                )
        return None

    def _check_anachronism(self, question: str, answer: str, context: Optional[str] = None) -> Optional[ErrorDiagnosis]:
        """Detect modern technology applied to pre-modern historical periods."""
        period_text = question + " " + (context or "")
        is_pre_1940 = bool(self._PRE_1940_PERIOD_RE.search(period_text))
        is_pre_1920 = bool(self._PRE_1920_PERIOD_RE.search(period_text))
        if not is_pre_1940 and not is_pre_1920:
            return None

        anachronistic = list(self._MODERN_TECH_RE.findall(answer))
        if is_pre_1920:
            anachronistic.extend(self._POST_1940_TECH_RE.findall(answer))

        if not anachronistic:
            return None

        unique_terms = list(dict.fromkeys(t.strip() for t in anachronistic))[:3]
        terms_str = ", ".join(f"'{t}'" for t in unique_terms)
        logger.debug(f"Anachronism detected: {terms_str}")
        return ErrorDiagnosis(
            error_type=ErrorType.ANACHRONISM, confidence=0.85,
            explanation=f"Your answer contains anachronistic elements — {terms_str} did not exist during the historical period being discussed.",
            suggestion="Stick to technologies, institutions, and concepts that actually existed at the time.",
        )

    def _llm_error_check(self, question: str, student_answer: str, context: Optional[str]) -> Optional[ErrorDiagnosis]:
        if not self.llm_generator:
            return None
        try:
            parts = [
                "You are checking a history student's answer for errors in reasoning.",
                f"\nQuestion: {question}",
                f"\nStudent's answer: {student_answer}",
            ]
            if context:
                parts.append(f"\nCorrect information: {context[:300]}")
            parts.append(
                "\nIs there any error in the student's reasoning? "
                "Reply in ONE sentence only. Do not use numbered steps. "
                "If yes, state what is wrong. If no, say 'No major errors'."
            )
            response = self.llm_generator.generate(prompt="\n".join(parts), temperature=0.5, max_new_tokens=150)
            if any(w in response.lower() for w in ('error', 'mistake', 'incorrect', 'wrong')):
                raw = response.strip()
                period_pos = raw.find('.')
                suggestion = raw[:period_pos + 1] if period_pos > 0 else raw[:150]
                return ErrorDiagnosis(
                    error_type=ErrorType.BIAS, confidence=0.5,
                    explanation="There may be some issues with your reasoning or interpretation.",
                    suggestion=suggestion,
                )
        except Exception as e:
            logger.error(f"LLM error check failed: {e}")
        return None

    def provide_feedback(self, question: str, student_answer: str, errors: List[ErrorDiagnosis]) -> str:
        if not errors or (len(errors) == 1 and errors[0].error_type == ErrorType.NONE):
            return "Good effort! Your answer shows understanding. Consider adding more specific details or evidence to strengthen it."
        primary = errors[0]
        parts = [f"**Feedback**: {primary.explanation}", f"\n**Suggestion**: {primary.suggestion}"]
        if len(errors) > 1:
            others = [e.error_type.value for e in errors[1:] if e.error_type != ErrorType.NONE]
            if others:
                parts.append(f"\nAlso consider: {', '.join(others)}")
        return "\n".join(parts)

    def is_answer_correct(self, question: str, student_answer: str, context: Optional[str] = None) -> bool:
        errors = self.analyze_answer(question, student_answer, context)
        has_major_errors = any(e.error_type != ErrorType.NONE and e.confidence > 0.6 for e in errors)
        return not has_major_errors and len(student_answer.split()) >= 10
