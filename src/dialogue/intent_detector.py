import os
import re
import sys
from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, Optional, List, Tuple
from pathlib import Path

import torch
from transformers import pipeline
from sentence_transformers import SentenceTransformer, util

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


# Improved NLI labels with clearer, more distinctive descriptions
INTENT_LABELS: Dict[Intent, str] = {
    Intent.GREETING: (
        "The student is starting a new conversation by saying hello, hi, good morning, "
        "or expressing readiness to begin learning"
    ),
    Intent.THANKS: (
        "The student is expressing gratitude, saying thank you, or acknowledging that "
        "the explanation was helpful"
    ),
    Intent.QUIT: (
        "The student wants to end the tutoring session by saying goodbye, bye, exit, "
        "quit, or that they are done for today"
    ),
    Intent.HELP: (
        "The student is asking about how to use THIS TUTORING APPLICATION ITSELF - "
        "its features, navigation, commands, or what topics it can teach. "
        "NOT asking about historical content or educational material"
    ),
    Intent.REQUEST_HINT: (
        "The student is stuck on a question the tutor asked and is explicitly requesting "
        "a hint, clue, nudge, or help to find the answer themselves"
    ),
    Intent.ANSWER_SUBMISSION: (
        "The student is providing their answer to a question the tutor asked them. "
        "They are stating what they believe the answer is, using phrases like 'I think', "
        "'the answer is', 'it was because', or making a declarative factual statement"
    ),
    Intent.QUESTION: (
        "The student is asking a factual question seeking specific information about historical events, "
        "people, dates, places, causes, or consequences. NOT asking for examples, explanations, help with "
        "the app, topic changes, or clarification. Purely seeking factual historical knowledge."
    ),
    Intent.EXPLANATION: (
        "The student is requesting a detailed explanation, description, or analysis. "
        "Using words like explain, describe, elaborate, tell me about, or requesting "
        "in-depth information beyond a simple factual answer"
    ),
    Intent.EXAMPLE: (
        "The student is explicitly requesting the tutor to provide specific historical examples, "
        "concrete illustrations, or instances. Uses phrases like 'give me an example', "
        "'what are some examples', 'can you show me an example', 'for instance'"
    ),
    Intent.CLARIFICATION: (
        "The student did not understand the tutor's previous message and is asking "
        "for it to be rephrased, simplified, or explained differently"
    ),
    Intent.NEW_TOPIC: (
        "The student wants to stop studying the current topic and switch to a completely different "
        "historical subject or period. Uses phrases like 'let's switch topics', 'can we study X instead', "
        "'I want to learn about something else', 'change topic', 'new topic'"
    ),
    Intent.CONTINUE: (
        "The student wants the tutor to continue with the current explanation or lesson, "
        "keep going, tell more, or move to the next part"
    ),
    Intent.REPEAT: (
        "The student missed what the tutor said and wants them to repeat or restate "
        "exactly what was just said, without changes"
    ),
}

# Prototype examples for sentence similarity fallback
INTENT_PROTOTYPES: Dict[Intent, List[str]] = {
    Intent.GREETING: [
        "Hello", "Hi there", "Good morning", "Hey", "I'm ready to start"
    ],
    Intent.THANKS: [
        "Thank you", "Thanks for the help", "That was helpful", "I appreciate it"
    ],
    Intent.QUIT: [
        "Goodbye", "Bye", "I'm done for today", "Exit", "See you later"
    ],
    Intent.HELP: [
        "How do I use this app?", "What can you teach me?", "What features are available?",
        "How does this work?", "What topics can we study?"
    ],
    Intent.REQUEST_HINT: [
        "I need a hint", "Can you give me a clue?", "I'm stuck", "Give me a nudge"
    ],
    Intent.ANSWER_SUBMISSION: [
        "I think the answer is X", "The answer is Y", "It was because of Z",
        "The main cause was A", "My answer is B"
    ],
    Intent.QUESTION: [
        "What caused the Berlin Wall to fall?", "When was the Treaty signed?",
        "Who was the first chancellor?", "Why did the war start?", "How did it happen?"
    ],
    Intent.EXPLANATION: [
        "Explain the Treaty of Versailles", "Can you elaborate on that?",
        "Tell me about the economic miracle", "Describe the political system"
    ],
    Intent.EXAMPLE: [
        "Give me an example", "Can you show me an example?",
        "What are some examples?", "Provide an illustration"
    ],
    Intent.CLARIFICATION: [
        "I don't understand", "Can you rephrase that?", "What do you mean?",
        "I'm confused", "Say that differently"
    ],
    Intent.NEW_TOPIC: [
        "Let's switch topics", "I want to study something else",
        "Can we learn about a different period?", "New topic please"
    ],
    Intent.CONTINUE: [
        "Continue", "Tell me more", "Keep going", "What happened next?", "Go on"
    ],
    Intent.REPEAT: [
        "Repeat that", "Say that again", "Can you repeat?", "One more time"
    ],
}


@dataclass
class IntentResult:
    intent: Intent
    confidence: float
    method: str  # "regex", "nli", "similarity", "fallback"
    entities: Dict[str, object] = field(default_factory=dict)
    debug_info: Dict[str, object] = field(default_factory=dict)


def _get_local_model_path(model_id: str) -> Optional[str]:
    """Check if model is cached locally"""
    hf_home = Path(os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface")))
    cache_dir = hf_home / "hub" / f"models--{model_id.replace('/', '--')}"
    snapshots_dir = cache_dir / "snapshots"
    if snapshots_dir.exists():
        snapshot_dirs = list(snapshots_dir.iterdir())
        if snapshot_dirs:
            local_path = str(snapshot_dirs[0])
            logger.info(f"Found cached model at: {local_path}")
            return local_path
    logger.warning(f"Model not in cache, will download: {model_id}")
    return None


class IntentDetector:
    NLI_MODEL_ID = "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli"
    SIMILARITY_MODEL_ID = "sentence-transformers/all-mpnet-base-v2"

    NLI_HIGH_CONFIDENCE = 0.70      # Accept immediately
    NLI_MEDIUM_CONFIDENCE = 0.45    # Apply heuristics
    NLI_LOW_CONFIDENCE = 0.30       # Use similarity fallback
    SIMILARITY_THRESHOLD = 0.65     # Minimum similarity score

    _LABELS = list(INTENT_LABELS.values())
    _LABEL_TO_INTENT = {label: intent for intent, label in INTENT_LABELS.items()}

    def __init__(self):
        device = 0 if torch.cuda.is_available() else -1
        local_path = _get_local_model_path(self.NLI_MODEL_ID)
        load_path = local_path if local_path else self.NLI_MODEL_ID
        logger.info(f"Loading NLI classifier from '{load_path}'")

        self.classifier = pipeline(
            "zero-shot-classification",
            model=load_path,
            device=device,
            local_files_only=local_path is not None,
        )

        # Load sentence similarity model
        logger.info(f"Loading similarity model: {self.SIMILARITY_MODEL_ID}")
        try:
            load_path = _get_local_model_path(self.SIMILARITY_MODEL_ID)
            self.similarity_model = SentenceTransformer(load_path)
            logger.info(f"Similarity model loaded!!!")
        except Exception as e:
            logger.error(f"Failed to load Similarity model: {e}", exc_info=True)
            raise

        # Pre-compute prototype embeddings
        logger.info("Computing prototype embeddings...")
        self.prototype_embeddings = {}
        for intent, examples in INTENT_PROTOTYPES.items():
            embeddings = self.similarity_model.encode(examples, convert_to_tensor=True)
            # Average embeddings for each intent
            self.prototype_embeddings[intent] = torch.mean(embeddings, dim=0)

        logger.info("ImprovedIntentDetector ready")

    def detect(self, message: str, context: Optional[Dict] = None) -> IntentResult:
        """Detect intent using multi-stage pipeline"""
        if not message.strip():
            return IntentResult(Intent.UNKNOWN, 0.0, "empty")

        context = context or {}

        # Stage 1: Fast regex pre-filters
        quick = self._quick_classify(message, context)
        if quick:
            logger.debug(f"Regex: {quick.intent.value} ({quick.confidence:.2f})")
            return quick

        # Stage 2: NLI classification with context
        augmented_text = self._build_input(message, context)
        nli_result = self.classifier(augmented_text, self._LABELS, multi_label=False)

        top_label = nli_result["labels"][0]
        nli_confidence = nli_result["scores"][0]
        nli_intent = self._LABEL_TO_INTENT.get(top_label, Intent.UNKNOWN)

        # Stage 3: Confidence-based decision
        if nli_confidence >= self.NLI_HIGH_CONFIDENCE:
            result = IntentResult(
                nli_intent, nli_confidence, "nli_high",
                entities=self._extract_entities(message, nli_intent),
                debug_info={"nli_top_3": list(zip(nli_result["labels"][:3], nli_result["scores"][:3]))}
            )
            logger.debug(f"NLI high conf: {result.intent.value} ({result.confidence:.2f})")
            return result

        elif nli_confidence >= self.NLI_MEDIUM_CONFIDENCE:
            # Medium confidence - apply heuristics
            refined = self._apply_heuristics(message, nli_intent, nli_confidence, context)
            logger.debug(f"NLI medium + heuristics: {refined.intent.value} ({refined.confidence:.2f})")
            return refined

        else:
            # Low confidence - use similarity fallback
            sim_intent, sim_score = self._similarity_fallback(message)

            if sim_score >= self.SIMILARITY_THRESHOLD:
                result = IntentResult(
                    sim_intent, sim_score, "similarity",
                    debug_info={
                        "nli_suggestion": (nli_intent.value, nli_confidence),
                        "similarity_score": sim_score
                    }
                )
                logger.debug(f"Similarity: {result.intent.value} ({result.confidence:.2f})")
                return result
            else:
                fallback = self._context_fallback(message, context, nli_intent, nli_confidence)
                logger.debug(f"Fallback: {fallback.intent.value} ({fallback.confidence:.2f})")
                return fallback

    def _similarity_fallback(self, message: str) -> Tuple[Intent, float]:
        """Use sentence similarity to find closest intent prototype"""
        msg_embedding = self.similarity_model.encode(message, convert_to_tensor=True)

        best_intent = Intent.UNKNOWN
        best_score = 0.0

        for intent, prototype_emb in self.prototype_embeddings.items():
            score = util.cos_sim(msg_embedding, prototype_emb).item()
            if score > best_score:
                best_score = score
                best_intent = intent

        return best_intent, best_score

    def _build_input(self, message: str, context: Dict) -> str:
        """Add context cues for better NLI classification"""
        cues = []

        if context.get("waiting_for_answer") and not self._is_hint_seeking(message):
            cues.append("The tutor asked the student a question and is waiting for their answer.")

        if context.get("in_explanation"):
            cues.append("The tutor is currently providing a detailed explanation.")

        if context.get("last_action") == "hint_given":
            cues.append("The tutor just gave the student a hint to help them answer.")

        # For long declarative messages without '?'
        words = message.split()
        if (len(words) > 12 and
            not message.strip().endswith('?') and
            not self._starts_with_command(message)):
            cues.append(
                "The student is making a long declarative statement, "
                "possibly providing their answer or sharing their understanding."
            )

        if cues:
            return " ".join(cues) + " Student's message: " + message
        return message

    def _apply_heuristics(self, message: str, nli_intent: Intent,
                         nli_conf: float, context: Dict) -> IntentResult:
        """Apply heuristics to refine medium-confidence NLI predictions"""

        # Override: Question mark usually means QUESTION (unless it's HELP/CLARIFICATION)
        if message.strip().endswith('?'):
            if nli_intent not in {Intent.QUESTION, Intent.HELP, Intent.CLARIFICATION, Intent.EXAMPLE}:
                if self._is_wh_question(message) and not self._is_system_meta(message):
                    return IntentResult(Intent.QUESTION, 0.75, "heuristic_wh_question")

        # Override: Waiting for answer + declarative = likely ANSWER_SUBMISSION
        if (context.get("waiting_for_answer") and
            not message.strip().endswith('?') and
            len(message.split()) >= 4 and
            not self._is_hint_seeking(message)):
            return IntentResult(Intent.ANSWER_SUBMISSION, 0.70, "heuristic_declarative_answer")

        # Trust NLI for these specific intents even at medium confidence
        if nli_intent in {Intent.GREETING, Intent.THANKS, Intent.QUIT, Intent.REPEAT}:
            return IntentResult(nli_intent, nli_conf, "nli_medium_trusted")

        # Default: accept NLI with adjusted confidence
        return IntentResult(nli_intent, nli_conf * 0.9, "nli_medium")

    def _context_fallback(self, message: str, context: Dict,
                         nli_intent: Intent, nli_conf: float) -> IntentResult:
        """Last resort fallback based on context"""

        # If waiting for answer and message is declarative, assume answer
        if (context.get("waiting_for_answer") and
            not message.strip().endswith('?') and
            len(message.split()) >= 3):
            return IntentResult(Intent.ANSWER_SUBMISSION, 0.55, "context_fallback_answer")

        # If message is a WH-question, default to QUESTION
        if self._is_wh_question(message) and not self._is_system_meta(message):
            return IntentResult(Intent.QUESTION, 0.60, "context_fallback_question")

        # Otherwise, mark as UNKNOWN
        return IntentResult(Intent.UNKNOWN, nli_conf, "context_fallback_unknown")

    # === Regex patterns (same as v1, proven to work) ===

    _GREETING_RE = re.compile(
        r'^\s*(hello|hi\b|hey\b|howdy|good\s+(morning|afternoon|evening)|greetings|'
        r'i\'m\s+ready\s+to\s+(learn|start|begin|study)|'
        r'let\'s\s+(start|begin|go\s+ahead))',
        re.IGNORECASE,
    )

    _REPEAT_RE = re.compile(
        r'\b(repeat\s+(that|what|it)|say\s+that\s+again|one\s+more\s+time|'
        r'can\s+you\s+repeat|could\s+you\s+repeat|please\s+repeat)\b',
        re.IGNORECASE,
    )

    _QUIT_RE = re.compile(
        r'^\s*(quit|exit|bye\b|goodbye|see\s+you|end\s+session|'
        r'i\'m\s+(done|finished|leaving))',
        re.IGNORECASE,
    )

    _CONTINUE_RE = re.compile(
        r'^\s*(continue|go\s+on|keep\s+going|carry\s+on|proceed)\s*[.!]?\s*$|'
        r'^\s*(yes|ok|okay|sure|alright|great|please|yep|yup)[\s,!.]+?'
        r'(continue|go\s+on|carry\s+on|keep\s+going|proceed)\s*[.!]?\s*$|'
        r'\b(tell\s+me\s+more|what\s+happened\s+next|what\s+comes\s+next|'
        r'and\s+then\s+what|next\s+please|go\s+on\s+please|please\s+continue|'
        r'continue\s+please|keep\s+going)\b',
        re.IGNORECASE,
    )

    _NEW_TOPIC_RE = re.compile(
        r'\b(new\s+topic|different\s+topic|change\s+topics?|switch\s+topics?|'
        r'something\s+(else|different)|'
        r'talk\s+about\s+something\s+(else|different)|'
        r'let\'?s\s+(switch|move\s+on)|'
        r'can\s+we\s+(switch|move\s+on\s+to|talk\s+about\s+something|study\s+\w+\s+instead|discuss\s+\w+\s+instead|learn\s+about)|'
        r'i\s+want\s+to\s+(learn|study|talk|discuss)\s+about|'
        r'\binstead\b)\b',
        re.IGNORECASE,
    )

    _CLARIFICATION_RE = re.compile(
        r'\b(what\s+do\s+you\s+mean(\s+by)?|can\s+you\s+rephrase|'
        r'rephrase\s+that|could\s+you\s+rephrase|'
        r'didn\'?t\s+(understand|get\s+that)|'
        r'don\'?t\s+understand(\s+that)?|'
        r'i\'?m?\s+(confused|lost)(\s+(by|about))?|confused\s+(by|about)\s+what|'
        r'say\s+that\s+(differently|in\s+simpler)|'
        r'in\s+simpler\s+(terms|words|language)|'
        r'(too\s+)?(complicated|complex)|'
        r'(can|could)\s+you\s+(clarify|simplify)|'
        r'explain\s+that\s+(differently|again)|'
        r'\bhuh\??\b|'
        r'broken\s+down\s+more|'
        r'not\s+(quite\s+)?following)\b',
        re.IGNORECASE,
    )

    _EXAMPLE_RE = re.compile(
        r'\b(give\s+me\s+an?\s+examples?|'
        r'show\s+me\s+an?\s+examples?|'
        r'can\s+you\s+(give|show)\s+(me\s+)?an?\s+examples?|'
        r'could\s+you\s+(give|show)\s+(me\s+)?an?\s+examples?|'
        r'provide\s+an?\s+examples?|'
        r'what\s+(is|are)\s+(a|an|some)\s+examples?)\b',
        re.IGNORECASE,
    )

    _COMMAND_EXPLANATION_RE = re.compile(
        r'^(explain|describe|elaborate|outline|summarize|summarise|'
        r'analyse|analyze|compare|contrast|discuss|evaluate|tell\s+me\s+about)\b',
        re.IGNORECASE,
    )

    _COMMAND_QUESTION_RE = re.compile(
        r'^(list|name|define|identify|ask\s+me|quiz\s+me)\b',
        re.IGNORECASE,
    )

    _INLINE_EXPLANATION_RE = re.compile(
        r'\b((can|could)\s+you\s+(explain|describe|elaborate)|'
        r'(can|could|please)\s+you\s+tell\s+me\s+(more\s+)?about|'
        r'please\s+(explain|describe|elaborate)|'
        r'tell\s+me\s+(more\s+)?about|'
        r'i?\s*want\s+to\s+know\s+more\s+about)\b',
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

    _HINT_SEEKING_RE = re.compile(
        r'\b(hint|clue|nudge|stuck|give\s+me\s+a\s+hint)\b',
        re.IGNORECASE,
    )

    _THANKS_RE = re.compile(
        r'\b(thank|thanks|thx|appreciate|grateful|helpful|'
        r'that\s+helped|that\s+makes\s+sense|got\s+it)\b',
        re.IGNORECASE,
    )

    _CAPABILITY_QUERY_RE = re.compile(
        r'\bwhat\s+can\s+you\s+(help|do|teach|cover|show)\b|'
        r'\bhow\s+can\s+you\s+help\b|'
        r'\bhow\s+does\s+(this|that|it)\s+work\b|'
        r'\bwhat\s+will\s+you\s+(help|do|teach)\b|'
        r'\bwhat\s+(are\s+you\s+|can\s+this\s+)(able|capable)\b|'
        r'\bwhat\s+features\b|'
        r'\bhow\s+do\s+i\s+use\b|'
        r'\bwhat\s+topics\b',
        re.IGNORECASE,
    )

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
        r'homework\b|'
        r'math\s+(homework|problem|question)|'
        r'\d+\s*[+\-*/]\s*\d+)\b',
        re.IGNORECASE,
    )

    def _quick_classify(self, message: str, context: Dict) -> Optional[IntentResult]:
        """Fast regex-based classification for clear-cut cases"""
        strip = message.strip()
        words = strip.split()

        # Pure gibberish
        if strip and all(c in '?!.,;:-_*#@%^&()[]{}/' for c in strip.replace(' ', '')):
            return IntentResult(Intent.UNKNOWN, 1.0, "regex_gibberish")

        # Repeating word gibberish
        words_lower = [w.lower().strip('.,!?;:-') for w in words if w.strip('.,!?;:-')]
        if words_lower and len(set(words_lower)) == 1 and len(words_lower) >= 3:
            return IntentResult(Intent.UNKNOWN, 1.0, "regex_repetition")

        # Off-topic
        if self._OFF_TOPIC_RE.search(message):
            return IntentResult(Intent.UNKNOWN, 1.0, "regex_off_topic")

        # GREETING (short, no '?')
        if self._GREETING_RE.search(message):
            if len(words) <= 6 and not strip.endswith('?'):
                return IntentResult(Intent.GREETING, 0.95, "regex_greeting")

        # REPEAT
        if self._REPEAT_RE.search(message):
            return IntentResult(Intent.REPEAT, 0.95, "regex_repeat")

        # QUIT
        if self._QUIT_RE.search(message):
            return IntentResult(Intent.QUIT, 0.95, "regex_quit")

        # CONTINUE (before HINT so "tell me more" doesn't match hint)
        if self._CONTINUE_RE.search(message):
            return IntentResult(Intent.CONTINUE, 0.95, "regex_continue")

        # NEW_TOPIC
        if self._NEW_TOPIC_RE.search(message):
            return IntentResult(Intent.NEW_TOPIC, 0.90, "regex_new_topic")

        # CLARIFICATION
        if self._CLARIFICATION_RE.search(message):
            return IntentResult(Intent.CLARIFICATION, 0.90, "regex_clarification")

        # EXAMPLE (but not if it's embedded in a multi-part question)
        if self._EXAMPLE_RE.search(message):
            # Check if this is a multi-intent message where EXAMPLE is secondary
            # Pattern: "Question... and/then can you give me an example"
            if re.search(r'^(what|why|how|when|where|who|which).{20,}\s+(and|then|also)\s+.*example', message, re.IGNORECASE):
                # This is likely a QUESTION with an example request tacked on - don't override
                pass
            else:
                return IntentResult(Intent.EXAMPLE, 0.90, "regex_example")

        # EXPLANATION (command form)
        if self._COMMAND_EXPLANATION_RE.match(message):
            return IntentResult(Intent.EXPLANATION, 0.90, "regex_explanation_cmd")

        # QUESTION (command form)
        if self._COMMAND_QUESTION_RE.match(message):
            return IntentResult(Intent.QUESTION, 0.85, "regex_question_cmd")

        # EXPLANATION (inline)
        if self._INLINE_EXPLANATION_RE.search(message):
            return IntentResult(Intent.EXPLANATION, 0.88, "regex_explanation_inline")

        # ANSWER_SUBMISSION (clear phrases)
        if self._ANSWER_SUBMISSION_RE.match(message):
            return IntentResult(Intent.ANSWER_SUBMISSION, 0.88, "regex_answer")

        # REQUEST_HINT (explicit - but check for negation)
        if self._is_hint_seeking(message):
            return IntentResult(Intent.REQUEST_HINT, 0.90, "regex_hint")

        # HELP (capability queries)
        if self._CAPABILITY_QUERY_RE.search(message):
            return IntentResult(Intent.HELP, 0.88, "regex_help")

        # THANKS
        if self._THANKS_RE.search(message):
            return IntentResult(Intent.THANKS, 0.85, "regex_thanks")

        return None

    def _is_wh_question(self, message: str) -> bool:
        """Check if message is a WH-question"""
        return bool(re.match(
            r'^(who|what|when|where|why|how|which|did|was|were|can|could|would|should)\b',
            message.strip(),
            re.IGNORECASE
        ))

    def _is_system_meta(self, message: str) -> bool:
        """Check if question is about the system itself"""
        msg_lower = message.lower()
        return any(word in msg_lower for word in [
            'help me with', 'can you do', 'how do i use', 'how does this work',
            'what can you', 'what are you', 'navigate', 'interface', 'application',
            'what features', 'how to use', 'what can i', 'how do i', 'what topics'
        ])

    def _is_hint_seeking(self, message: str) -> bool:
        """Check if message contains hint-seeking words"""
        if not self._HINT_SEEKING_RE.search(message):
            return False
        # Check for negation
        return not self._has_negation(message, ['hint', 'clue', 'help'])

    def _has_negation(self, message: str, keywords: list) -> bool:
        """Check if keywords are negated in the message"""
        msg_lower = message.lower()
        for keyword in keywords:
            if keyword in msg_lower:
                # Find position of keyword
                pos = msg_lower.find(keyword)
                # Check 20 chars before for negation words
                before = msg_lower[max(0, pos-20):pos]
                if re.search(r'\b(no|not|don\'?t|doesn\'?t|won\'?t|can\'?t|never)\b', before):
                    return True
        return False

    def _starts_with_command(self, message: str) -> bool:
        """Check if message starts with command words"""
        return bool(re.match(
            r'^(what|why|how|when|where|who|which|explain|describe|tell|give|'
            r'show|list|name|compare|contrast|discuss|outline|summarize|define|'
            r'can\s+you|could\s+you|would\s+you|please)\b',
            message.strip(),
            re.IGNORECASE
        ))

    def _extract_entities(self, message: str, intent: Intent) -> Dict:
        """Extract entities based on intent"""
        if intent != Intent.REQUEST_HINT:
            return {}

        if re.search(r'\b(big|major|more|full)\s+hint\b', message, re.IGNORECASE):
            return {"hint_level": "full"}
        if re.search(r'\b(small|little|subtle|tiny)\s+hint\b', message, re.IGNORECASE):
            return {"hint_level": "nudge"}
        return {"hint_level": "partial"}

    def get_intent_description(self, intent: Intent) -> str:
        """Get human-readable description of intent"""
        return INTENT_LABELS.get(intent, "Unknown intent")
