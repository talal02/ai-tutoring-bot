"""
Dialogue management components for the tutoring system.
Handles intent detection, conversation state, and dialogue flow.
"""

from .intent_detector import IntentDetector, Intent
from .dialogue_manager import DialogueManager, ConversationState, LessonState

__all__ = [
    'IntentDetector',
    'Intent',
    'DialogueManager',
    'ConversationState',
    'LessonState',
]
