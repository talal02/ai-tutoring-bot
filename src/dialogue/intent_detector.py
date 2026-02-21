"""
Intent detection using zero-shot NLI classification.
Model: MoritzLaurer/deberta-v3-xsmall-zeroshot-v1.1-all-33
"""

import os
import re
import sys
from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, Optional
from pathlib import Path

import torch
from transformers import pipeline

sys.path.append(str(Path(__file__).parent.parent))
from utils.logger import get_logger

logger = get_logger(__name__)


class Intent(Enum):
    QUESTION          = "question"
    CLARIFICATION     = "clarification"
    EXAMPLE           = "example"
    EXPLANATION       = "explanation"
    ANSWER_SUBMISSION = "answer_submission"
    REQUEST_HINT      = "request_hint"
    NEW_TOPIC         = "new_topic"
    CONTINUE          = "continue"
    REPEAT            = "repeat"
    GREETING          = "greeting"
    THANKS            = "thanks"
    QUIT              = "quit"
    HELP              = "help"
    UNKNOWN           = "unknown"

INTENT_LABELS: Dict[Intent, str] = {
    Intent.GREETING:          "the user is greeting, saying hello, or starting the conversation",
    Intent.THANKS:            "the user is saying thank you, expressing gratitude, or appreciating the help received",
    Intent.QUIT:              "the user wants to quit, exit, stop, or end the conversation entirely",
    Intent.HELP:              "the user is asking how to USE this application itself, what features it has, or how to navigate the interface — NOT asking about historical events, people, dates, causes, consequences, or any subject matter content",
    Intent.REQUEST_HINT:      "the student is stuck on a question and explicitly asking for a hint, clue, or nudge toward the answer",
    Intent.ANSWER_SUBMISSION: "the student is directly answering a quiz or study question that was posed to them, stating what they believe the answer to be",
    Intent.QUESTION:          "the user is asking about historical events, people, dates, places, causes, consequences, or any factual question about history — asking what, when, where, who, why, or how something happened in the past",
    Intent.EXPLANATION:       "the user is asking the tutor to explain, describe, analyze, or elaborate on any topic, event, document, or concept in detail",
    Intent.EXAMPLE:           "the user is asking the tutor to provide a specific historical example or concrete illustration to support a concept",
    Intent.CLARIFICATION:     "the user found the tutor's previous message unclear or confusing and is asking for it to be rephrased or explained differently",
    Intent.NEW_TOPIC:         "the user wants to switch to a different subject or start studying a completely new historical topic",
    Intent.CONTINUE:          "the user wants to move forward and continue to the next part of the current lesson or explanation",
    Intent.REPEAT:            "the user missed or did not understand and wants the tutor to repeat or restate exactly what was just said",
}


@dataclass
class IntentResult:
    intent: Intent
    confidence: float
    entities: Dict[str, object] = field(default_factory=dict)


def _get_local_model_path(model_id: str) -> Optional[str]:
    hf_home = Path(os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface")))
    cache_dir = hf_home / "hub" / f"models--{model_id.replace('/', '--')}"
    snapshots_dir = cache_dir / "snapshots"
    if snapshots_dir.exists():
        snapshot_dirs = list(snapshots_dir.iterdir())
        if snapshot_dirs:
            local_path = str(snapshot_dirs[0])
            logger.info(f"Found cached classifier at: {local_path}")
            return local_path
    logger.warning(f"Classifier not in local cache, will attempt download: {model_id}")
    return None


class IntentDetector:
    MODEL_ID = "MoritzLaurer/deberta-v3-xsmall-zeroshot-v1.1-all-33"
    CONFIDENCE_THRESHOLD = 0.29
    ANSWER_SUBMISSION_EXTENDED_THRESHOLD = 0.55

    _LABELS          = list(INTENT_LABELS.values())
    _LABEL_TO_INTENT = {label: intent for intent, label in INTENT_LABELS.items()}

    def __init__(self):
        device = 0 if torch.cuda.is_available() else -1
        local_path = _get_local_model_path(self.MODEL_ID)
        load_path  = local_path if local_path else self.MODEL_ID
        logger.info(
            f"Loading zero-shot classifier from '{load_path}' "
            f"on {'cuda' if device == 0 else 'cpu'}"
        )
        self.classifier = pipeline(
            "zero-shot-classification",
            model=load_path,
            device=device,
            local_files_only=local_path is not None,
        )
        logger.info("IntentDetector ready")

    def detect(self, message: str, context: Optional[Dict] = None) -> IntentResult:
        if not message.strip():
            return IntentResult(Intent.UNKNOWN, 0.0)

        # Stage 1: Fast pre-filters (bypass NLI entirely)
        quick = self._quick_classify(message)
        if quick:
            logger.debug(
                f"Quick-classified: {quick.intent.value} ({quick.confidence:.2f})"
                f" — '{message[:60]}'"
            )
            return quick

        # Stage 2: NLI classifier
        text   = self._build_input(message, context)
        result = self.classifier(text, self._LABELS, multi_label=False)

        top_label:  str   = result["labels"][0]
        confidence: float = result["scores"][0]
        intent = self._LABEL_TO_INTENT.get(top_label, Intent.UNKNOWN)

        # Stage 3: Post-process NLI result
        if message.strip().endswith('?') and intent in {
            Intent.GREETING, Intent.THANKS, Intent.QUIT, Intent.CONTINUE
        }:
            logger.debug(
                f"NLI returned {intent.value} for '?'-ending message "
                f"({confidence:.2f}) — overriding to QUESTION"
            )
            intent     = Intent.QUESTION
            confidence = max(confidence, 0.65)

        # Stage 4: Low-confidence fallback (<0.29)
        if confidence < self.CONFIDENCE_THRESHOLD:
            if self._is_declarative_answer(message, context):
                logger.debug(
                    f"Low confidence ({confidence:.2f}) + waiting_for_answer"
                    f" → ANSWER_SUBMISSION fallback"
                )
                return IntentResult(Intent.ANSWER_SUBMISSION, 0.60)

            # WH-question ending in '?' — boost to QUESTION unless it's clearly
            msg_lower = message.lower()
            is_wh_question = (
                message.strip().endswith('?')
                and re.match(r'^(who|what|when|where|why|how|which|did|was|were|can|could)\b',
                             message.strip(), re.IGNORECASE)
            )
            is_system_meta = any(word in msg_lower for word in [
                'help me with', 'can you do', 'how do i use', 'how does this work',
                'what can you', 'what are you', 'navigate', 'interface', 'application',
                'what features', 'how to use', 'what can i', 'how do i'
            ])

            if is_wh_question and not is_system_meta:
                logger.debug(
                    f"Low confidence ({confidence:.2f}) WH-question → QUESTION override"
                )
                return IntentResult(Intent.QUESTION, 0.70)

            logger.debug(f"Low confidence ({confidence:.2f}) for '{top_label}' → UNKNOWN")
            return IntentResult(Intent.UNKNOWN, confidence)

        # Stage 4.5: UNKNOWN override for clear WH-questions
        if intent == Intent.UNKNOWN and confidence < 0.65:
            msg_lower = message.lower()
            is_wh_question = (
                message.strip().endswith('?')
                and re.match(r'^(who|what|when|where|why|how|which|did|was|were)\b',
                             message.strip(), re.IGNORECASE)
            )
            is_system_meta = any(word in msg_lower for word in [
                'help me with', 'can you do', 'how do i use', 'how does this work',
                'what can you', 'what are you', 'navigate', 'interface', 'application',
                'what features', 'how to use', 'what can i', 'how do i'
            ])

            if is_wh_question and not is_system_meta:
                logger.debug(
                    f"UNKNOWN intent ({confidence:.2f}) for WH-question → QUESTION override"
                )
                return IntentResult(Intent.QUESTION, 0.72)

        # Stage 5: Extended ANSWER_SUBMISSION fallback (0.29 ≤ conf < 0.55)
        if (
            confidence < self.ANSWER_SUBMISSION_EXTENDED_THRESHOLD
            and intent not in {Intent.GREETING, Intent.QUIT, Intent.THANKS,
                               Intent.REQUEST_HINT, Intent.REPEAT,
                               Intent.ANSWER_SUBMISSION}
            and self._is_declarative_answer(message, context)
        ):
            logger.debug(
                f"Mid-confidence ({confidence:.2f}) + waiting_for_answer"
                f" + declarative → ANSWER_SUBMISSION extended fallback"
            )
            return IntentResult(Intent.ANSWER_SUBMISSION, 0.65)

        # Stage 6: Override HELP misclassification for WH-questions
        if intent == Intent.HELP and confidence < 0.65:
            msg_lower = message.lower()
            is_wh_question = (
                message.strip().endswith('?')
                and re.match(r'^(who|what|when|where|why|how|which|did|was|were)\b',
                             message.strip(), re.IGNORECASE)
            )
            # Check if it's clearly about system usage/features
            is_system_meta = any(word in msg_lower for word in [
                'help me with', 'can you do', 'how do i use', 'how does this work',
                'what can you', 'what are you', 'navigate', 'interface', 'application',
                'what features', 'how to use', 'what can i', 'how do i', 'what topics'
            ])

            if is_wh_question and not is_system_meta:
                logger.debug(
                    f"HELP misclassification ({confidence:.2f}) → QUESTION (WH-question override)"
                )
                return IntentResult(Intent.QUESTION, 0.75)

        logger.debug(f"Intent: {intent.value} ({confidence:.2f}) — '{message[:60]}'")
        return IntentResult(intent, confidence, self._extract_entities(message, intent))

    _GREETING_RE = re.compile(
        r'^\s*(hello|hi\b|hey\b|howdy|good\s+(morning|afternoon|evening)|greetings)'
        r'|i\'m\s+ready\s+to\s+(learn|start|begin|study)'
        r'|let\'s\s+(start|begin|go\s+ahead)',
        re.IGNORECASE,
    )

    _REPEAT_RE = re.compile(
        r'\b(repeat\s+(that|what|it)|say\s+that\s+again|one\s+more\s+time|'
        r'can\s+you\s+repeat|could\s+you\s+repeat|please\s+repeat)\b'
        r'|\brepeat\b.*\bjust\s+said\b',
        re.IGNORECASE,
    )

    _QUIT_RE = re.compile(
        r'^\s*(quit|exit|bye\b|goodbye|see\s+you|end\s+session|i\'m\s+(done|finished|leaving))',
        re.IGNORECASE,
    )

    # Forward-motion continuation phrases.
    _CONTINUE_RE = re.compile(
        r'^\s*(continue|go\s+on|keep\s+going|carry\s+on|proceed)\s*[.!]?\s*$'
        r'|^\s*(yes|ok|okay|sure|alright|great|please|yep|yup)[\s,!.]+?'
        r'(continue|go\s+on|carry\s+on|keep\s+going|proceed)\s*[.!]?\s*$'
        r'|\b(tell\s+me\s+more|what\s+happened\s+next|what\s+comes\s+next|'
        r'and\s+then\s+what|next\s+please|go\s+on\s+please|please\s+continue|'
        r'continue\s+please|keep\s+going)\b',
        re.IGNORECASE,
    )

    _NEW_TOPIC_RE = re.compile(
        r'\b(new\s+topic|different\s+topic|change\s+topics?|switch\s+topics?|'
        r'something\s+(else|different)|'
        r'talk\s+about\s+something\s+(else|different)|'
        r'let\'?s\s+(switch|move\s+on)|'
        r'can\s+we\s+(switch|move\s+on\s+to|talk\s+about\s+something)|'
        r'i\s+want\s+to\s+(learn|study|talk|discuss)\s+about)\b',
        re.IGNORECASE,
    )

    _CLARIFICATION_RE = re.compile(
        r'\b(what\s+do\s+you\s+mean(\s+by)?|can\s+you\s+rephrase|'
        r'rephrase\s+that|could\s+you\s+rephrase|'
        r'didn\'?t\s+(understand|get\s+that)|'
        r'don\'?t\s+understand(\s+that)?|'
        r'i\'?m?\s+confused\s+(by|about)|confused\s+(by|about)\s+what|'
        r'say\s+that\s+(differently|in\s+simpler)|'
        r'in\s+simpler\s+(terms|words|language)|'
        r'can\s+you\s+clarify|could\s+you\s+clarify)\b',
        re.IGNORECASE,
    )

    _EXAMPLE_RE = re.compile(
        r'\b(give\s+me\s+an?\s+example|'
        r'show\s+me\s+an?\s+example|'
        r'can\s+you\s+(give|show)\s+(me\s+)?an?\s+example|'
        r'could\s+you\s+(give|show)\s+(me\s+)?an?\s+example|'
        r'provide\s+an?\s+example|'
        r'what\s+(is|are)\s+(a|an|some)\s+example)\b',
        re.IGNORECASE,
    )

    # Educational command verbs at message start → EXPLANATION.
    _COMMAND_EXPLANATION_RE = re.compile(
        r'^(explain|describe|elaborate|outline|summarize|summarise|analyse|analyze|'
        r'compare|contrast|discuss|evaluate|tell\s+me\s+about)\b',
        re.IGNORECASE,
    )

    # Command verbs at message start → QUESTION (quiz/factual requests).
    _COMMAND_QUESTION_RE = re.compile(
        r'^(list|name|define|identify|ask\s+me|quiz\s+me)\b',
        re.IGNORECASE,
    )

    # "Can you explain X?" embedded anywhere in the message → EXPLANATION.
    _INLINE_EXPLANATION_RE = re.compile(
        r'\b(can\s+you\s+(explain|describe|elaborate|tell\s+me\s+about)|'
        r'could\s+you\s+(explain|describe|elaborate)|'
        r'please\s+(explain|describe|elaborate))\b',
        re.IGNORECASE,
    )

    _ANSWER_SUBMISSION_RE = re.compile(
        r'^(i\s+think\s+(that\s+|it\s+|the\s+)?'
        r'|i\s+believe\s+(that\s+|it\s+|the\s+)?'
        r'|my\s+answer\s+is\s+(that\s+)?'
        r'|the\s+answer\s+is\s+(that\s+)?'
        r'|in\s+my\s+opinion[,\s]+'
        r'|it\s+was\s+because\s+'
        r'|the\s+main\s+(cause|reason|factor)\s+was\s+'
        r'|the\s+(primary|key|main)\s+(cause|reason)\s+was\s+'
        r'|i\s+would\s+say\s+(that\s+)?'
        r'|i\s+reckon\s+(that\s+)?)',
        re.IGNORECASE,
    )

    # Explicit hint-seeking words.
    _HINT_SEEKING_RE = re.compile(
        r'\b(hint|clue|nudge|stuck|give\s+me\s+a\s+hint)\b',
        re.IGNORECASE,
    )

    # Extended gratitude phrases that NLI scores below 0.29 (e.g. "I really appreciate…").
    _THANKS_RE = re.compile(
        r'\b(i\s+really\s+appreciate|i\s+appreciate\s+your|much\s+appreciated|'
        r'that\s+was\s+(very\s+)?helpful|truly\s+grateful|deeply\s+grateful|'
        r'so\s+grateful|many\s+thanks)\b',
        re.IGNORECASE,
    )

    # "What can you help/do…" — capability/onboarding queries that NLI returns below
    _CAPABILITY_QUERY_RE = re.compile(
        r'\bwhat\s+can\s+you\s+(help|do|teach|cover|show)\b|'
        r'\bhow\s+can\s+you\s+help\b|'
        r'\bwhat\s+will\s+you\s+(help|do|teach)\b|'
        r'\bwhat\s+are\s+you\s+(able|capable)\b',
        re.IGNORECASE,
    )

    # Clearly off-topic messages that are not about history.
    _OFF_TOPIC_RE = re.compile(
        r'\b(weather|forecast|'
        r'recommend\s+(me\s+)?(a\s+)?(?:\w+\s+)?(movie|film|tv\s+show|show|series|song)|'
        r'write\s+(me\s+)?(a\s+)?(code|program|script|function|class)|'
        r'python\b|javascript\b|css\b|html\b|programming\b|'
        r'do\s+you\s+(like|love|enjoy|hate)\b|'
        r'are\s+you\s+(happy|sad|bored|alive|a\s+(robot|human|ai))\b|'
        r'how\s+do\s+you\s+feel|'
        r'what\'?s?\s+your\s+name|'
        r'order\s+(food|pizza|delivery)|'
        r'\bjoke\b|'
        r'\d+\s*[+\-*/]\s*\d+)\b',
        re.IGNORECASE,
    )

    # Used to detect command/question starts (guards against false ANSWER_SUBMISSION).
    _QUESTION_OR_COMMAND_START_RE = re.compile(
        r'^(what|why|how|when|where|who|which|explain|describe|tell|give|'
        r'show|list|name|compare|contrast|discuss|outline|summarize|define|'
        r'can\s+you|could\s+you|would\s+you|please)\b',
        re.IGNORECASE,
    )

    def _quick_classify(self, message: str) -> Optional["IntentResult"]:
        """Return a high-confidence result for clear-cut keyword patterns."""
        strip = message.strip()
        words = strip.split()

        # Pure punctuation / empty content → UNKNOWN 
        if strip and all(c in '?!.,;:-_*#@%^&()[]{}/' for c in strip.replace(' ', '')):
            return IntentResult(Intent.UNKNOWN, 1.0)

        # Repeating-word gibberish → UNKNOWN
        words_lower = [w.lower().strip('.,!?;:-') for w in words if w.strip('.,!?;:-')]
        if words_lower and len(set(words_lower)) == 1 and len(words_lower) >= 3:
            return IntentResult(Intent.UNKNOWN, 1.0)

        # Clearly off-topic → UNKNOWN
        if self._OFF_TOPIC_RE.search(message):
            return IntentResult(Intent.UNKNOWN, 1.0)

        # GREETING — short (≤6 words), no trailing '?'
        if self._GREETING_RE.search(message):
            if len(words) <= 6 and not strip.endswith('?'):
                return IntentResult(Intent.GREETING, 0.95)

        # REPEAT
        if self._REPEAT_RE.search(message):
            return IntentResult(Intent.REPEAT, 0.95)

        # QUIT 
        if self._QUIT_RE.search(message):
            return IntentResult(Intent.QUIT, 0.95)

        # CONTINUE — before HINT so "tell me more" → CONTINUE 
        if self._CONTINUE_RE.search(message):
            return IntentResult(Intent.CONTINUE, 0.95)

        # NEW_TOPIC
        if self._NEW_TOPIC_RE.search(message):
            return IntentResult(Intent.NEW_TOPIC, 0.90)

        # CLARIFICATION 
        if self._CLARIFICATION_RE.search(message):
            return IntentResult(Intent.CLARIFICATION, 0.90)

        # EXAMPLE
        if self._EXAMPLE_RE.search(message):
            return IntentResult(Intent.EXAMPLE, 0.90)

        # EXPLANATION
        if self._COMMAND_EXPLANATION_RE.match(message):
            return IntentResult(Intent.EXPLANATION, 0.90)

        # QUESTION — list / name / define / quiz me at message start 
        if self._COMMAND_QUESTION_RE.match(message):
            return IntentResult(Intent.QUESTION, 0.85)

        # EXPLANATION — "can you explain…" embedded anywhere
        if self._INLINE_EXPLANATION_RE.search(message):
            return IntentResult(Intent.EXPLANATION, 0.88)

        #  ANSWER_SUBMISSION — unambiguous starter phrases 
        if self._ANSWER_SUBMISSION_RE.match(message):
            return IntentResult(Intent.ANSWER_SUBMISSION, 0.88)

        # REQUEST_HINT — explicit hint-seeking words
        if self._HINT_SEEKING_RE.search(message):
            return IntentResult(Intent.REQUEST_HINT, 0.90)

        # HELP — "what can you help/do…" capability queries
        if self._CAPABILITY_QUERY_RE.search(message):
            return IntentResult(Intent.HELP, 0.88)

        # THANKS — extended gratitude phrases NLI misses (<0.29)
        if self._THANKS_RE.search(message):
            return IntentResult(Intent.THANKS, 0.88)

        return None

    def _is_declarative_answer(self, message: str, context: Optional[Dict]) -> bool:
        """Return True when the message looks like a declarative student answer."""
        return bool(
            context
            and context.get("waiting_for_answer")
            and not message.strip().endswith('?')
            and not self._QUESTION_OR_COMMAND_START_RE.match(message)
            and not self._HINT_SEEKING_RE.search(message)
            and len(message.split()) >= 4
        )

    def _build_input(self, message: str, context: Optional[Dict]) -> str:
        """Prepend context cues so the NLI model resolves ambiguous inputs correctly."""
        if not context:
            return message

        cues = []
        student_wants_hint = bool(self._HINT_SEEKING_RE.search(message))

        if context.get("waiting_for_answer") and not student_wants_hint:
            cues.append("The tutor has asked a question and is waiting for the student's answer.")
        if context.get("in_explanation"):
            cues.append("An explanation is currently in progress.")
        if context.get("last_action") == "hint_given":
            cues.append("A hint was just provided to the student.")

        # Long declarative messages without a question mark are likely student answers.
        words = message.split()
        is_declarative = (
            len(words) > 15
            and not message.strip().endswith('?')
            and not self._QUESTION_OR_COMMAND_START_RE.match(message)
        )
        if is_declarative:
            cues.append(
                "The student appears to be making a declarative historical statement, "
                "possibly volunteering their knowledge or submitting an answer."
            )

        return (" ".join(cues) + " Student message: " + message) if cues else message

    def _extract_entities(self, message: str, intent: Intent) -> Dict:
        if intent != Intent.REQUEST_HINT:
            return {}
        if re.search(r'\b(big|major|more)\s+hint\b', message, re.IGNORECASE):
            return {"hint_level": "full"}
        if re.search(r'\b(small|little|subtle)\s+hint\b', message, re.IGNORECASE):
            return {"hint_level": "nudge"}
        return {"hint_level": "partial"}

    def get_intent_description(self, intent: Intent) -> str:
        return INTENT_LABELS.get(intent, "Unknown intent")
