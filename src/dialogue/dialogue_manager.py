from typing import ClassVar, Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import re
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from utils.logger import get_logger
from dialogue.intent_detector import IntentDetector, Intent, IntentResult

logger = get_logger(__name__)


class LessonPhase(Enum):
    LEARNING = "learning"
    PRACTICE = "practice"
    ASSESSMENT = "assessment"


@dataclass
class LessonState:
    topic: str = "general"
    phase: LessonPhase = LessonPhase.LEARNING
    current_question: Optional[str] = None
    attempts: int = 0
    hints_given: List[str] = field(default_factory=list)
    hints_used: int = 0
    correct_answers: int = 0
    incorrect_answers: int = 0
    topics_covered: List[str] = field(default_factory=list)
    start_time: datetime = field(default_factory=datetime.now)
    last_updated: datetime = field(default_factory=datetime.now)

    def reset(self) -> None:
        self.topic = "general"
        self.phase = LessonPhase.LEARNING
        self.current_question = None
        self.attempts = 0
        self.hints_given = []
        self.hints_used = 0
        self.start_time = datetime.now()
        self.last_updated = datetime.now()

    def update(self) -> None:
        self.last_updated = datetime.now()

    def to_dict(self) -> Dict[str, Any]:
        return {
            'topic': self.topic,
            'phase': self.phase.value,
            'current_question': self.current_question,
            'attempts': self.attempts,
            'hints_used': self.hints_used,
            'correct_answers': self.correct_answers,
            'incorrect_answers': self.incorrect_answers,
            'topics_covered': self.topics_covered,
        }


@dataclass
class ConversationTurn:
    timestamp: datetime
    user_message: str
    intent: Intent
    system_response: str
    lesson_state: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)


class ConversationState:
    def __init__(self, max_history: int = 20):
        self.max_history = max_history
        self.turns: List[ConversationTurn] = []
        self.context: Dict[str, Any] = {}
        self.session_start = datetime.now()
        logger.info("ConversationState initialized")

    def add_turn(self, user_message: str, intent: Intent, system_response: str,
                 lesson_state: Dict[str, Any], metadata: Optional[Dict[str, Any]] = None) -> None:
        turn = ConversationTurn(
            timestamp=datetime.now(),
            user_message=user_message,
            intent=intent,
            system_response=system_response,
            lesson_state=lesson_state,
            metadata=metadata or {},
        )
        self.turns.append(turn)
        if len(self.turns) > self.max_history:
            self.turns = self.turns[-self.max_history:]
        logger.debug(f"Added turn with intent: {intent.value}")

    def get_recent_turns(self, n: int = 5) -> List[ConversationTurn]:
        return self.turns[-n:] if self.turns else []

    def get_last_turn(self) -> Optional[ConversationTurn]:
        return self.turns[-1] if self.turns else None

    def update_context(self, key: str, value: Any) -> None:
        self.context[key] = value

    def get_context(self, key: str, default: Any = None) -> Any:
        return self.context.get(key, default)

    def clear_context(self) -> None:
        self.context = {}

    def reset(self) -> None:
        self.turns = []
        self.context = {}
        self.session_start = datetime.now()
        logger.info("Conversation state reset")

    def get_statistics(self) -> Dict[str, Any]:
        if not self.turns:
            return {'total_turns': 0, 'session_duration_minutes': 0}
        intent_counts = {}
        for turn in self.turns:
            intent_counts[turn.intent.value] = intent_counts.get(turn.intent.value, 0) + 1
        duration = (datetime.now() - self.session_start).total_seconds() / 60
        return {
            'total_turns': len(self.turns),
            'session_duration_minutes': round(duration, 2),
            'intent_distribution': intent_counts,
            'session_start': self.session_start.isoformat(),
        }


class DialogueManager:
    # Static response strategies — covers every intent that needs no dynamic fields.
    _STRATEGY_TABLE: ClassVar[Dict[Intent, Dict]] = {
        Intent.GREETING:      {"action": "greeting",            "use_rag": False, "assessment_needed": False},
        Intent.QUIT:          {"action": "farewell",            "use_rag": False, "assessment_needed": False},
        Intent.HELP:          {"action": "help_message",        "use_rag": False, "assessment_needed": False},
        Intent.THANKS:        {"action": "acknowledgment",      "use_rag": False, "assessment_needed": False},
        Intent.REPEAT:        {"action": "repeat_last",         "use_rag": False, "assessment_needed": False},
        Intent.CONTINUE:      {"action": "continue_lesson",     "use_rag": True,  "assessment_needed": False},
        Intent.QUESTION:      {"action": "answer_question",     "use_rag": True,  "assessment_needed": False},
        Intent.EXPLANATION:   {"action": "provide_explanation", "use_rag": True,  "assessment_needed": False, "detailed": True},
        Intent.EXAMPLE:       {"action": "provide_example",     "use_rag": True,  "assessment_needed": False},
        Intent.CLARIFICATION: {"action": "clarify",             "use_rag": True,  "assessment_needed": False, "refer_previous": True},
    }

    def __init__(self):
        self.intent_detector = IntentDetector()
        self.conversation = ConversationState()
        self.lesson = LessonState()
        logger.info("DialogueManager initialized")

    def process_message(self, message: str) -> Dict[str, Any]:
        logger.info(f"Processing message: '{message[:50]}...'")
        context = self._build_context_for_intent()
        intent_result = self.intent_detector.detect(message, context)
        self.lesson.update()
        response_strategy = self._determine_response_strategy(message, intent_result)
        self._update_context(message, intent_result)
        logger.debug(f"Response strategy: {response_strategy['action']}")
        return {
            'intent': intent_result.intent,
            'confidence': intent_result.confidence,
            'entities': intent_result.entities,
            'strategy': response_strategy,
            'lesson_state': self.lesson.to_dict(),
            'conversation_context': self.conversation.context,
        }

    def _build_context_for_intent(self) -> Dict[str, Any]:
        last_turn = self.conversation.get_last_turn()
        context = {
            'waiting_for_answer': self.lesson.current_question is not None,
            'last_action': self.conversation.get_context('last_action'),
            'in_explanation': self.conversation.get_context('in_explanation', False),
            'current_topic': self.lesson.topic,
            'lesson_phase': self.lesson.phase.value,
        }
        if last_turn:
            context['last_intent'] = last_turn.intent.value
        return context

    def _determine_response_strategy(self, message: str, intent_result: IntentResult) -> Dict[str, Any]:
        intent = intent_result.intent
        if intent == Intent.NEW_TOPIC:
            self.lesson.reset()
            # Extract and persist the new topic so downstream RAG queries stay on-topic.
            new_topic = self._extract_topic_from_message(message)
            self.lesson.topic = new_topic
            return {'action': 'new_topic', 'use_rag': True, 'assessment_needed': False, 'topic': new_topic}

        if intent == Intent.REQUEST_HINT:
            return {
                'action': 'provide_hint',
                'hint_level': intent_result.entities.get('hint_level', 'partial'),
                'question': self.lesson.current_question or self._get_last_topic_context(),
                'student_response': message,
                'use_rag': True,
                'assessment_needed': True,
            }

        if intent == Intent.ANSWER_SUBMISSION:
            return {
                'action': 'assess_answer',
                'question': self.lesson.current_question or "previous question",
                'student_answer': message,
                'use_rag': True,
                'assessment_needed': True,
            }

        strategy = dict(self._STRATEGY_TABLE.get(
            intent,
            {'action': 'fallback_question', 'use_rag': True, 'assessment_needed': False},
        ))

        if intent in {Intent.QUESTION, Intent.EXPLANATION, Intent.EXAMPLE, Intent.CLARIFICATION}:
            strategy['question'] = message
        if intent == Intent.QUESTION:
            strategy['topic'] = self.lesson.topic

        return strategy

    def _update_context(self, message: str, intent_result: IntentResult) -> None:
        intent = intent_result.intent
        if intent == Intent.REQUEST_HINT:
            self.conversation.update_context('last_action', 'hint_requested')
            self.lesson.hints_used += 1
        elif intent == Intent.ANSWER_SUBMISSION:
            self.conversation.update_context('last_action', 'answer_submitted')
            self.lesson.attempts += 1
        elif intent == Intent.QUESTION:
            self.conversation.update_context('last_action', 'question_asked')
            self.lesson.current_question = message
        if intent in [Intent.EXPLANATION, Intent.EXAMPLE]:
            self.conversation.update_context('in_explanation', True)
        else:
            self.conversation.update_context('in_explanation', False)

    def record_response(self, user_message: str, intent: Intent,
                        system_response: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        self.conversation.add_turn(
            user_message=user_message,
            intent=intent,
            system_response=system_response,
            lesson_state=self.lesson.to_dict(),
            metadata=metadata,
        )

    def update_lesson_result(self, correct: bool) -> None:
        if correct:
            self.lesson.correct_answers += 1
            self.conversation.update_context('last_action', 'correct_answer')
            if self.lesson.phase == LessonPhase.PRACTICE:
                if self.lesson.correct_answers >= 3:
                    self.lesson.phase = LessonPhase.ASSESSMENT
        else:
            self.lesson.incorrect_answers += 1
            self.conversation.update_context('last_action', 'incorrect_answer')
        logger.debug(
            f"Lesson updated: correct={self.lesson.correct_answers}, "
            f"incorrect={self.lesson.incorrect_answers}"
        )

    def set_current_question(self, question: str) -> None:
        self.lesson.current_question = question
        self.lesson.attempts = 0
        self.lesson.hints_given = []
        self.lesson.hints_used = 0

    def add_hint_given(self, hint: str, level: str) -> None:
        self.lesson.hints_given.append(f"{level}: {hint}")
        self.conversation.update_context('last_action', 'hint_given')

    _TOPIC_RE = re.compile(
        r'\b(?:about|learn|study|cover|switch\s+to|change\s+to|interested\s+in|know\s+about)\s+'
        r'(.{3,60}?)(?:\s+now\b|\s*[.!?]\s*$|\s*$)',
        re.IGNORECASE,
    )

    def _extract_topic_from_message(self, message: str) -> str:
        match = self._TOPIC_RE.search(message)
        if match:
            return match.group(1).strip().rstrip('.')
        return message[:80].strip()

    def _get_last_topic_context(self) -> str:
        """Return the last user message with meaningful content for RAG fallback."""
        content_intents = {
            Intent.QUESTION, Intent.EXPLANATION, Intent.ANSWER_SUBMISSION,
            Intent.EXAMPLE, Intent.CLARIFICATION, Intent.CONTINUE,
        }
        for turn in reversed(self.conversation.turns):
            if turn.intent in content_intents:
                return turn.user_message
        return self.lesson.topic

    def reset_conversation(self) -> None:
        self.conversation.reset()
        self.lesson.reset()
        logger.info("Dialogue manager reset")

    def get_statistics(self) -> Dict[str, Any]:
        return {
            'conversation': self.conversation.get_statistics(),
            'lesson': self.lesson.to_dict(),
        }
