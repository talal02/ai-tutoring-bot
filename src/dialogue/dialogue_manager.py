"""
Dialogue Manager for managing conversation flow and state.
Tracks conversation history, lesson progress, and ensures pedagogical coherence.
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from utils.logger import get_logger
from dialogue.intent_detector import IntentDetector, Intent, IntentResult

logger = get_logger(__name__)


class LessonPhase(Enum):
    """Current phase of the lesson."""
    INTRODUCTION = "introduction"      # Introducing topic
    EXPLORATION = "exploration"        # Exploring concepts
    PRACTICE = "practice"             # Practicing with questions
    ASSESSMENT = "assessment"         # Assessing understanding
    REVIEW = "review"                 # Reviewing learned material
    COMPLETED = "completed"           # Lesson completed


@dataclass
class LessonState:
    """Tracks state of current lesson."""
    topic: str = "general"
    phase: LessonPhase = LessonPhase.INTRODUCTION
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
        """Reset lesson state for new lesson."""
        self.topic = "general"
        self.phase = LessonPhase.INTRODUCTION
        self.current_question = None
        self.attempts = 0
        self.hints_given = []
        self.hints_used = 0
        self.start_time = datetime.now()
        self.last_updated = datetime.now()

    def update(self) -> None:
        """Update last_updated timestamp."""
        self.last_updated = datetime.now()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
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
    """Represents a single turn in conversation."""
    timestamp: datetime
    user_message: str
    intent: Intent
    system_response: str
    lesson_state: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)


class ConversationState:
    """
    Manages overall conversation state.
    Tracks history, context, and provides utilities for dialogue management.
    """

    def __init__(self, max_history: int = 20):
        """
        Initialize conversation state.

        Args:
            max_history: Maximum number of turns to keep in history.
        """
        self.max_history = max_history
        self.turns: List[ConversationTurn] = []
        self.context: Dict[str, Any] = {}
        self.session_start = datetime.now()

        logger.info("ConversationState initialized")

    def add_turn(
        self,
        user_message: str,
        intent: Intent,
        system_response: str,
        lesson_state: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Add a conversation turn.

        Args:
            user_message: User's message.
            intent: Detected intent.
            system_response: System's response.
            lesson_state: Current lesson state.
            metadata: Optional metadata.
        """
        turn = ConversationTurn(
            timestamp=datetime.now(),
            user_message=user_message,
            intent=intent,
            system_response=system_response,
            lesson_state=lesson_state,
            metadata=metadata or {},
        )

        self.turns.append(turn)

        # Limit history size
        if len(self.turns) > self.max_history:
            self.turns = self.turns[-self.max_history:]

        logger.debug(f"Added turn with intent: {intent.value}")

    def get_recent_turns(self, n: int = 5) -> List[ConversationTurn]:
        """
        Get recent conversation turns.

        Args:
            n: Number of recent turns to retrieve.

        Returns:
            List of recent turns.
        """
        return self.turns[-n:] if self.turns else []

    def get_last_turn(self) -> Optional[ConversationTurn]:
        """Get the last conversation turn."""
        return self.turns[-1] if self.turns else None

    def get_context_for_llm(self, n: int = 5) -> List[Dict[str, str]]:
        """
        Get conversation history formatted for LLM.

        Args:
            n: Number of recent turns to include.

        Returns:
            List of message dictionaries for LLM.
        """
        recent_turns = self.get_recent_turns(n)
        messages = []

        for turn in recent_turns:
            messages.append({
                "role": "user",
                "content": turn.user_message,
            })
            messages.append({
                "role": "assistant",
                "content": turn.system_response,
            })

        return messages

    def update_context(self, key: str, value: Any) -> None:
        """
        Update conversation context.

        Args:
            key: Context key.
            value: Context value.
        """
        self.context[key] = value

    def get_context(self, key: str, default: Any = None) -> Any:
        """
        Get value from conversation context.

        Args:
            key: Context key.
            default: Default value if key not found.

        Returns:
            Context value or default.
        """
        return self.context.get(key, default)

    def clear_context(self) -> None:
        """Clear conversation context."""
        self.context = {}

    def reset(self) -> None:
        """Reset conversation state."""
        self.turns = []
        self.context = {}
        self.session_start = datetime.now()
        logger.info("Conversation state reset")

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get conversation statistics.

        Returns:
            Dictionary with conversation stats.
        """
        if not self.turns:
            return {
                'total_turns': 0,
                'session_duration_minutes': 0,
            }

        intent_counts = {}
        for turn in self.turns:
            intent_name = turn.intent.value
            intent_counts[intent_name] = intent_counts.get(intent_name, 0) + 1

        duration = (datetime.now() - self.session_start).total_seconds() / 60

        return {
            'total_turns': len(self.turns),
            'session_duration_minutes': round(duration, 2),
            'intent_distribution': intent_counts,
            'session_start': self.session_start.isoformat(),
        }


class DialogueManager:
    """
    Manages dialogue flow and orchestrates conversation.
    Coordinates intent detection, state management, and response generation.
    """

    def __init__(self):
        """Initialize dialogue manager."""
        self.intent_detector = IntentDetector()
        self.conversation = ConversationState()
        self.lesson = LessonState()

        logger.info("DialogueManager initialized")

    def process_message(
        self,
        message: str,
    ) -> Dict[str, Any]:
        """
        Process user message and determine response strategy.

        Args:
            message: User's message.

        Returns:
            Dictionary with processing results and instructions.
        """
        logger.info(f"Processing message: '{message[:50]}...'")

        # Detect intent
        context = self._build_context_for_intent()
        intent_result = self.intent_detector.detect(message, context)

        # Update lesson state
        self.lesson.update()

        # Determine response strategy based on intent
        response_strategy = self._determine_response_strategy(
            message, intent_result
        )

        # Update context
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
        """Build context dictionary for intent detection."""
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

    def _determine_response_strategy(
        self,
        message: str,
        intent_result: IntentResult,
    ) -> Dict[str, Any]:
        """
        Determine how to respond based on intent and state.

        Args:
            message: User message.
            intent_result: Intent detection result.

        Returns:
            Response strategy dictionary.
        """
        intent = intent_result.intent

        # Handle each intent type
        if intent == Intent.GREETING:
            return {
                'action': 'greeting',
                'use_rag': False,
                'assessment_needed': False,
            }

        elif intent == Intent.QUIT:
            return {
                'action': 'farewell',
                'use_rag': False,
                'assessment_needed': False,
            }

        elif intent == Intent.HELP:
            return {
                'action': 'help_message',
                'use_rag': False,
                'assessment_needed': False,
            }

        elif intent == Intent.THANKS:
            return {
                'action': 'acknowledgment',
                'use_rag': False,
                'assessment_needed': False,
            }

        elif intent == Intent.REQUEST_HINT:
            hint_level = intent_result.entities.get('hint_level', 'partial')
            return {
                'action': 'provide_hint',
                'hint_level': hint_level,
                'question': self.lesson.current_question or message,
                'student_response': message,
                'use_rag': True,
                'assessment_needed': True,
            }

        elif intent == Intent.ANSWER_SUBMISSION:
            return {
                'action': 'assess_answer',
                'question': self.lesson.current_question or "previous question",
                'student_answer': message,
                'use_rag': True,
                'assessment_needed': True,
            }

        elif intent == Intent.QUESTION:
            # Extract topics if any
            topics = intent_result.entities.get('topics', [])
            if topics:
                self.lesson.topic = topics[0]

            return {
                'action': 'answer_question',
                'question': message,
                'topic': self.lesson.topic,
                'use_rag': True,
                'assessment_needed': False,
            }

        elif intent == Intent.EXPLANATION:
            return {
                'action': 'provide_explanation',
                'question': message,
                'use_rag': True,
                'assessment_needed': False,
                'detailed': True,
            }

        elif intent == Intent.EXAMPLE:
            return {
                'action': 'provide_example',
                'topic': message,
                'use_rag': True,
                'assessment_needed': False,
            }

        elif intent == Intent.CLARIFICATION:
            return {
                'action': 'clarify',
                'question': message,
                'use_rag': True,
                'assessment_needed': False,
                'refer_previous': True,
            }

        elif intent == Intent.NEW_TOPIC:
            self.lesson.reset()
            return {
                'action': 'new_topic',
                'use_rag': False,
                'assessment_needed': False,
            }

        elif intent == Intent.CONTINUE:
            return {
                'action': 'continue_lesson',
                'use_rag': True,
                'assessment_needed': False,
            }

        elif intent == Intent.REPEAT:
            return {
                'action': 'repeat_last',
                'use_rag': False,
                'assessment_needed': False,
            }

        else:  # UNKNOWN
            return {
                'action': 'fallback_question',
                'question': message,
                'use_rag': True,
                'assessment_needed': False,
            }

    def _update_context(self, message: str, intent_result: IntentResult) -> None:
        """Update conversation context after processing message."""
        intent = intent_result.intent

        # Update last action
        if intent == Intent.REQUEST_HINT:
            self.conversation.update_context('last_action', 'hint_requested')
            self.lesson.hints_used += 1
        elif intent == Intent.ANSWER_SUBMISSION:
            self.conversation.update_context('last_action', 'answer_submitted')
            self.lesson.attempts += 1
        elif intent == Intent.QUESTION:
            self.conversation.update_context('last_action', 'question_asked')
            self.lesson.current_question = message

        # Update explanation state
        if intent in [Intent.EXPLANATION, Intent.EXAMPLE]:
            self.conversation.update_context('in_explanation', True)
        else:
            self.conversation.update_context('in_explanation', False)

    def record_response(
        self,
        user_message: str,
        intent: Intent,
        system_response: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Record a conversation turn.

        Args:
            user_message: User's message.
            intent: Detected intent.
            system_response: System's response.
            metadata: Optional metadata.
        """
        self.conversation.add_turn(
            user_message=user_message,
            intent=intent,
            system_response=system_response,
            lesson_state=self.lesson.to_dict(),
            metadata=metadata,
        )

    def update_lesson_result(self, correct: bool) -> None:
        """
        Update lesson statistics after answer assessment.

        Args:
            correct: Whether answer was correct.
        """
        if correct:
            self.lesson.correct_answers += 1
            self.conversation.update_context('last_action', 'correct_answer')

            # Move to next phase if appropriate
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
        """
        Set the current question being worked on.

        Args:
            question: Question text.
        """
        self.lesson.current_question = question
        self.lesson.attempts = 0
        self.lesson.hints_given = []
        self.lesson.hints_used = 0

    def add_hint_given(self, hint: str, level: str) -> None:
        """
        Record that a hint was given.

        Args:
            hint: Hint text.
            level: Hint level (nudge, partial, full).
        """
        self.lesson.hints_given.append(f"{level}: {hint}")
        self.conversation.update_context('last_action', 'hint_given')

    def get_conversation_history(self, n: int = 5) -> List[Dict[str, str]]:
        """
        Get conversation history for LLM context.

        Args:
            n: Number of recent turns.

        Returns:
            List of message dictionaries.
        """
        return self.conversation.get_context_for_llm(n)

    def reset_conversation(self) -> None:
        """Reset entire conversation state."""
        self.conversation.reset()
        self.lesson.reset()
        logger.info("Dialogue manager reset")

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive statistics.

        Returns:
            Dictionary with all statistics.
        """
        return {
            'conversation': self.conversation.get_statistics(),
            'lesson': self.lesson.to_dict(),
        }
