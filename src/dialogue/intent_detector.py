import re
from enum import Enum
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from utils.logger import get_logger

logger = get_logger(__name__)


class Intent(Enum):
    QUESTION = "question"
    CLARIFICATION = "clarification"
    EXAMPLE = "example"
    EXPLANATION = "explanation"
    ANSWER_SUBMISSION = "answer_submission"
    REQUEST_HINT = "request_hint"
    NEW_TOPIC = "new_topic"
    CONTINUE = "continue"
    REPEAT = "repeat"
    GREETING = "greeting"
    THANKS = "thanks"
    QUIT = "quit"
    HELP = "help"
    UNKNOWN = "unknown"


@dataclass
class IntentResult:
    intent: Intent
    confidence: float
    entities: Dict[str, any] = None

    def __post_init__(self):
        if self.entities is None:
            self.entities = {}


class IntentDetector:
    PATTERNS = {
        Intent.GREETING: [
            r'\b(hello|hi|hey|greetings|good\s+(morning|afternoon|evening))\b',
        ],
        Intent.THANKS: [
            r'\b(thank(s| you)|thx|appreciate)\b',
        ],
        Intent.QUIT: [
            r'\b(quit|exit|bye|goodbye|stop)\b',
        ],
        Intent.HELP: [
            r'\b(help|guide|how to|what can you|what do you)\b',
            r'^(help|info|information)$',
        ],
        Intent.REQUEST_HINT: [
            r'\b(hint|clue|help me|give me a hint|need a hint)\b',
            r'\b(stuck|don\'?t know|not sure|unsure)\b',
        ],
        Intent.EXAMPLE: [
            r'\b(example|instance|for example|such as|like what)\b',
            r'\bcan you (give|provide|show).*example\b',
        ],
        Intent.EXPLANATION: [
            r'\b(explain|elaborate|tell me more|more detail|in detail)\b',
            r'\b(why|how does|how do)\b.*\?',
        ],
        Intent.CLARIFICATION: [
            r'\b(what do you mean|clarify|confused|don\'?t understand)\b',
            r'\b(rephrase|say that again|didn\'?t get)\b',
        ],
        Intent.NEW_TOPIC: [
            r'\b(new topic|different topic|change topic|something else)\b',
            r'\b(next|move on|let\'?s talk about)\b',
        ],
        Intent.CONTINUE: [
            r'\b(continue|go on|keep going|next)\b',
            r'^(ok|okay|yes|yeah|sure)$',
        ],
        Intent.REPEAT: [
            r'\b(repeat|again|say that again|one more time)\b',
        ],
        Intent.ANSWER_SUBMISSION: [
            r'^(i think|i believe|my answer is|the answer is)',
            r'\b(because|since|therefore)\b.*\.',
        ],
    }

    QUESTION_WORDS = [
        'what', 'when', 'where', 'who', 'whom', 'which', 'why', 'how',
        'can', 'could', 'would', 'should', 'is', 'are', 'was', 'were',
        'do', 'does', 'did', 'has', 'have', 'had',
    ]

    def __init__(self):
        self.compiled_patterns = {}
        for intent, patterns in self.PATTERNS.items():
            self.compiled_patterns[intent] = [
                re.compile(pattern, re.IGNORECASE) for pattern in patterns
            ]
        logger.info("IntentDetector initialized with rule-based patterns")

    def detect(self, message: str, context: Optional[Dict] = None) -> IntentResult:
        message = message.strip()

        if not message:
            return IntentResult(Intent.UNKNOWN, 0.0)

        logger.debug(f"Detecting intent for: '{message[:50]}...'")

        matches = []
        for intent, patterns in self.compiled_patterns.items():
            for pattern in patterns:
                if pattern.search(message):
                    confidence = self._calculate_confidence(message, intent)
                    matches.append((intent, confidence))
                    break

        if self._is_question(message):
            if self._contains_keywords(message, ['example', 'instance']):
                matches.append((Intent.EXAMPLE, 0.7))
            elif self._contains_keywords(message, ['why', 'how', 'explain']):
                matches.append((Intent.EXPLANATION, 0.7))
            elif self._contains_keywords(message, ['what', 'when', 'where', 'who']):
                matches.append((Intent.QUESTION, 0.8))
            else:
                matches.append((Intent.QUESTION, 0.6))

        if context:
            matches = self._refine_with_context(matches, message, context)

        if matches:
            matches.sort(key=lambda x: x[1], reverse=True)
            best_intent, confidence = matches[0]
            logger.debug(f"Detected intent: {best_intent.value} (confidence: {confidence:.2f})")
            entities = self._extract_entities(message, best_intent)
            return IntentResult(best_intent, confidence, entities)

        logger.debug("No intent detected, defaulting to UNKNOWN")
        return IntentResult(Intent.UNKNOWN, 0.0)

    def _is_question(self, message: str) -> bool:
        if message.endswith('?'):
            return True
        first_word = message.split()[0].lower() if message.split() else ""
        return first_word in self.QUESTION_WORDS

    def _contains_keywords(self, message: str, keywords: List[str]) -> bool:
        message_lower = message.lower()
        return any(keyword in message_lower for keyword in keywords)

    def _calculate_confidence(self, message: str, intent: Intent) -> float:
        confidence = 0.7
        if len(message.split()) <= 2:
            confidence *= 0.8
        if self._is_question(message) and intent in [
            Intent.QUESTION, Intent.EXPLANATION, Intent.EXAMPLE, Intent.CLARIFICATION
        ]:
            confidence *= 1.2
        pattern_matches = sum(
            1 for pattern in self.compiled_patterns.get(intent, [])
            if pattern.search(message)
        )
        if pattern_matches > 1:
            confidence *= 1.1
        return min(confidence, 1.0)

    def _refine_with_context(self, matches: List[Tuple[Intent, float]],
                            message: str, context: Dict) -> List[Tuple[Intent, float]]:
        if context.get('waiting_for_answer', False):
            if not self._is_question(message):
                matches.append((Intent.ANSWER_SUBMISSION, 0.8))

        if context.get('last_action') == 'hint_given':
            if not self._is_question(message):
                matches.append((Intent.ANSWER_SUBMISSION, 0.75))

        if context.get('in_explanation', False):
            for intent, confidence in matches:
                if intent == Intent.CONTINUE:
                    matches = [(Intent.CONTINUE, min(confidence * 1.2, 1.0))]
                    break
        return matches

    def _extract_entities(self, message: str, intent: Intent) -> Dict[str, any]:
        entities = {}
        topics = self._extract_topics(message)
        if topics:
            entities['topics'] = topics

        if intent == Intent.REQUEST_HINT:
            if re.search(r'\b(big|major|more)\s+hint\b', message, re.IGNORECASE):
                entities['hint_level'] = 'full'
            elif re.search(r'\b(small|little|subtle)\s+hint\b', message, re.IGNORECASE):
                entities['hint_level'] = 'nudge'
            else:
                entities['hint_level'] = 'partial'
        return entities

    def _extract_topics(self, message: str) -> List[str]:
        topic_keywords = [
            'primary source', 'secondary source', 'evidence',
            'chronology', 'periodization', 'timeline',
            'cause', 'consequence', 'effect',
            'continuity', 'change', 'transformation',
            'interpretation', 'perspective', 'viewpoint',
            'document', 'artifact', 'photograph',
        ]
        message_lower = message.lower()
        return [topic for topic in topic_keywords if topic in message_lower]

    def get_intent_description(self, intent: Intent) -> str:
        descriptions = {
            Intent.QUESTION: "Asking a new question",
            Intent.CLARIFICATION: "Requesting clarification",
            Intent.EXAMPLE: "Requesting an example",
            Intent.EXPLANATION: "Requesting detailed explanation",
            Intent.ANSWER_SUBMISSION: "Submitting an answer",
            Intent.REQUEST_HINT: "Requesting a hint",
            Intent.NEW_TOPIC: "Starting a new topic",
            Intent.CONTINUE: "Continuing current lesson",
            Intent.REPEAT: "Requesting repetition",
            Intent.GREETING: "Greeting",
            Intent.THANKS: "Expressing gratitude",
            Intent.QUIT: "Ending conversation",
            Intent.HELP: "Requesting help",
            Intent.UNKNOWN: "Unknown intent",
        }
        return descriptions.get(intent, "Unknown")
