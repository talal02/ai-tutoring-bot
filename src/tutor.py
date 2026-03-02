from pathlib import Path
from typing import Optional, Dict, Any
import sys
import re

sys.path.append(str(Path(__file__).parent))

from utils.config import get_config
from utils.logger import setup_logger, get_logger

from llm.model_loader import ModelLoader
from llm.generator import LLMGenerator
from rag.document_processor import DocumentProcessor
from rag.embedder import Embedder
from rag.retriever import FAISSRetriever
from dialogue.dialogue_manager import DialogueManager
from dialogue.intent_detector import Intent
from assessment.assessment_engine import AssessmentEngine

logger = get_logger(__name__)


class HistoryTutor:
    _QUIZ_REQUEST_RE = re.compile(
        r"\b(quiz\s+me|ask\s+me\s+(a\s+)?question|test\s+me|give\s+me\s+(a\s+)?quiz)\b",
        re.IGNORECASE,
    )

    def __init__(self, config_path: Optional[str] = None):
        self.config = get_config(config_path)

        self.logger = setup_logger(
            __name__,
            level=self.config.logging.level,
            log_file=self.config.logging.log_file,
            console_output=self.config.logging.console_output,
        )

        self.logger.info("="*70)
        self.logger.info("History Tutoring System")
        self.logger.info("="*70)

        self.model_loader = None
        self.generator = None
        self.doc_processor = None
        self.embedder = None
        self.retriever = None
        self.dialogue_manager = DialogueManager()
        self.assessment_engine = AssessmentEngine()

        # Last detected intent/confidence/sources (for API to read after chat())
        self.last_intent = None
        self.last_confidence = 0.0
        self.last_sources = []

        self.logger.info("System Initialized")

    def setup_llm(self, adapter_path: Optional[str] = None) -> None:
        self.logger.info("Setting up LLM layer...")

        self.model_loader = ModelLoader(self.config.llm)
        model, tokenizer = self.model_loader.load_model(adapter_path=adapter_path)

        self.generator = LLMGenerator(
            model=model,
            tokenizer=tokenizer,
            llm_config=self.config.llm,
            prompts_config=self.config.prompts,
        )

        self.assessment_engine.set_llm_generator(self.generator)

        memory_stats = self.model_loader.get_memory_usage()
        self.logger.info(f"LLM setup complete. Memory: {memory_stats}")

    def setup_rag(
        self,
        dataset_path: Optional[str] = None,
        pdf_directory: Optional[str] = None,
        index_name: str = "history_index",
        rebuild_index: bool = False,
    ) -> None:
        self.doc_processor = DocumentProcessor(self.config.rag)
        self.embedder = Embedder(self.config.rag)
        self.embedder.load_model()
        self.retriever = FAISSRetriever(self.config.rag, self.embedder)

        if not rebuild_index:
            try:
                self.retriever.load_index(index_name)
                self.logger.info("Loaded existing RAG index")
                return
            except FileNotFoundError:
                self.logger.info("No existing index found, building new one")

        all_documents = []

        if dataset_path:
            documents = self.doc_processor.load_json_dataset(dataset_path)
            all_documents.extend(documents)
            self.logger.info(f"Loaded {len(documents)} documents from JSON dataset")

        if pdf_directory:
            pdf_docs = self.doc_processor.load_directory(
                pdf_directory,
                file_pattern="*.pdf",
                recursive=False
            )
            all_documents.extend(pdf_docs)
            self.logger.info(f"Loaded {len(pdf_docs)} PDF documents")

        if all_documents:
            processed_docs = self.doc_processor.process_documents(all_documents, chunk=True)
            self.retriever.build_index(processed_docs, show_progress=True)
            self.retriever.save_index(index_name)
            stats = self.doc_processor.get_statistics(processed_docs)
            self.logger.info(f"RAG setup complete. Stats: {stats}")
        else:
            self.logger.warning("No documents loaded for RAG system")

    def chat(self, message: str, use_rag: bool = True) -> str:
        self.logger.info(f"\n{'='*70}\nUser: {message}\n{'='*70}")

        processing_result = self.dialogue_manager.process_message(message)
        intent = processing_result['intent']
        confidence = processing_result['confidence']
        strategy = processing_result['strategy']

        self.last_intent = intent
        self.last_confidence = confidence

        self.logger.info(f"Intent detected: {intent.value} (confidence: {confidence:.2f})")

        if intent == Intent.GREETING:
            response = self._handle_greeting()
        elif intent == Intent.QUIT:
            response = self._handle_farewell()
        elif intent == Intent.HELP:
            response = self._handle_help()
        elif intent == Intent.THANKS:
            response = "You're welcome! What else would you like to learn about?"
        elif intent == Intent.REPEAT:
            # Return the last assistant response instead of generating a new one
            response = self._handle_repeat()
        elif intent == Intent.UNKNOWN:
            response = self._handle_unknown()
        else:
            is_quiz_turn = intent == Intent.QUESTION and self._is_quiz_request(message)
            context = None
            self.last_sources = []
            if use_rag and strategy.get('use_rag') and self.retriever:
                retrieval_query = (
                    strategy.get('question')
                    or strategy.get('student_answer')
                    or strategy.get('student_response')
                    or message
                )
                results = self.retriever.retrieve(retrieval_query, top_k=5)
                context = self.retriever.format_retrieved_context(results)
                self.last_sources = results  # cache so API avoids a second retrieval
                self.logger.info(f"Retrieved {len(results)} context documents")
            elif use_rag and not self.retriever:
                self.logger.warning("RAG requested but retriever not initialized")

            if is_quiz_turn:
                response = self._generate_quiz_question(message, context)
                # Track the exact quiz question asked by the tutor.
                self.dialogue_manager.set_current_question(response)

            elif strategy.get('assessment_needed') and self.assessment_engine:
                action = strategy.get('action')

                if action == 'provide_hint':
                    result = self.assessment_engine.provide_hint(
                        question=strategy.get('question', message),
                        student_response=strategy.get('student_response', ''),
                        hint_level=strategy.get('hint_level', 'partial'),
                        context=context,
                        previous_hints=self.dialogue_manager.lesson.hints_given,
                    )
                    response = result.response
                    self.dialogue_manager.add_hint_given(response, strategy.get('hint_level', 'partial'))

                elif action == 'assess_answer':
                    result = self.assessment_engine.assess_answer(
                        question=strategy.get('question', ''),
                        student_answer=strategy.get('student_answer', message),
                        context=context,
                        provide_feedback=True,
                    )
                    response = result.response
                    if result.is_correct is not None:
                        self.dialogue_manager.update_lesson_result(result.is_correct)
                else:
                    response = self._generate_llm_response(message, context)
            else:
                action = strategy.get('action')
                if action == 'hint_limit_reached':
                    response = strategy.get('message', "You've used all hints for this question. Try answering now.")
                elif action == 'fallback_question' and strategy.get('fallback_message'):
                    response = strategy['fallback_message']
                else:
                    response = self._generate_llm_response(message, context)

        self.dialogue_manager.record_response(
            user_message=message,
            intent=intent,
            system_response=response,
            metadata={'action': strategy['action']},
        )

        self.logger.info(f"Response: {response[:100]}...")
        return response

    def _generate_llm_response(self, message: str, context: Optional[str]) -> str:
        if not self.generator:
            return "I'm ready to help! Please set up the LLM first."

        if context:
            return self.generator.generate_with_rag(
                question=message,
                context=context,
                temperature=0.7,
            )
        else:
            history = self.generator.get_history()
            prompt = self.generator.format_chat_prompt(
                user_message=message,
                history=history[-6:] if history else None,
            )
            response = self.generator.generate(prompt)
            # Update history manually since generate_with_rag does it internally
            self.generator.add_to_history(message, response)
            return response

    def _handle_repeat(self) -> str:
        history = self.generator.get_history() if self.generator else []
        # History is stored as [user, assistant, user, assistant, ...]
        for entry in reversed(history):
            if entry.get("role") == "assistant":
                return entry["content"]
        return "I don't have anything to repeat yet. What would you like to learn about?"

    def _handle_greeting(self) -> str:
        return (
            "Hello! I'm your history tutor. I can help you learn about historical sources, "
            "chronology, causation, and more. What would you like to explore today?"
        )

    def _handle_farewell(self) -> str:
        stats = self.dialogue_manager.get_statistics()
        return (
            f"Great work today! We covered {stats['conversation']['total_turns']} exchanges. "
            "Keep learning! Goodbye!"
        )

    def _handle_help(self) -> str:
        return (
            "I can help you with:\n"
            "• Answering history questions\n"
            "• Explaining historical concepts\n"
            "• Providing examples from historical sources\n\n"
            "Just ask me anything about history!"
        )

    def _handle_unknown(self) -> str:
        return (
            "I'm a history tutor and can only help with history-related topics. "
            "Try asking a history question, requesting an explanation, or saying "
            "'Ask me a question about [topic]' to get quizzed."
        )

    def _is_quiz_request(self, message: str) -> bool:
        return bool(self._QUIZ_REQUEST_RE.search(message))

    def _generate_quiz_question(self, user_request: str, context: Optional[str]) -> str:
        if not self.generator:
            return "What year did the Berlin Wall fall?"

        prompt_parts = [
            "You are a history tutor creating a quiz item.",
            "Output EXACTLY ONE short question.",
            "Do NOT include an answer, explanation, options, or any extra text.",
            "Question must end with '?'.",
            f"Student request: {user_request}",
        ]
        if context:
            prompt_parts.append(f"Source context: {context[:350]}")
        prompt_parts.append("Quiz question:")

        prompt = self.generator.format_chat_prompt(user_message="\n".join(prompt_parts))
        raw = self.generator.generate(prompt=prompt, temperature=0.4, max_new_tokens=48)
        question = self._extract_first_question(raw)
        return question or "What key event marked German reunification?"

    def _extract_first_question(self, text: str) -> str:
        if not text:
            return ""
        cleaned = text.replace("\n", " ").strip().strip('"\'')
        q_idx = cleaned.find('?')
        if q_idx == -1:
            return ""
        candidate = cleaned[:q_idx + 1].strip()
        # Keep only the final sentence fragment if the model adds preamble.
        for sep in [". ", ": ", " - "]:
            if sep in candidate:
                candidate = candidate.split(sep)[-1].strip()
        return candidate if candidate.endswith('?') else ""

    def reset(self) -> None:
        self.dialogue_manager.reset_conversation()
        if self.generator:
            self.generator.clear_history()
        self.logger.info("System reset")

    def get_statistics(self) -> Dict[str, Any]:
        stats = {'dialogue': self.dialogue_manager.get_statistics()}
        if self.model_loader:
            stats['llm_memory'] = self.model_loader.get_memory_usage()
        if self.retriever:
            stats['rag'] = self.retriever.get_statistics()
        if self.assessment_engine:
            stats['assessment'] = self.assessment_engine.get_statistics()
        return stats

    def cleanup(self) -> None:
        self.logger.info("Cleaning up resources...")
        if self.model_loader:
            self.model_loader.unload_model()
        if self.embedder:
            self.embedder.unload_model()
        self.logger.info("Cleanup complete")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="History Tutoring System")
    parser.add_argument("--config", type=str, default="../configs/config.yaml")
    parser.add_argument("--dataset", type=str, default="../questions_dataset.json")
    parser.add_argument("--use-rag", action="store_true", default=True)
    parser.add_argument("--no-rag", action="store_true")
    parser.add_argument("--rebuild-index", action="store_true")
    args = parser.parse_args()

    use_rag_mode = args.use_rag and not args.no_rag

    print("\nInitializing...\n")

    tutor = HistoryTutor(config_path=args.config)

    print("Loading language model...")
    tutor.setup_llm()

    if use_rag_mode:
        print("\nSetting up RAG system...")
        print(f"  - Loading JSON dataset: {args.dataset}")
        tutor.setup_rag(
            dataset_path=args.dataset,
            rebuild_index=args.rebuild_index,
            pdf_directory="./data"
        )
    else:
        print("\nRAG disabled - using LLM only mode")

    print("\n" + "="*70)
    print("SYSTEM READY")
    print("="*70)

    stats = tutor.get_statistics()
    print("\nSystem Information:")
    for key, value in stats.items():
        print(f"  {key}: {value}")

    print("\n" + "="*70)
    print("MODE: " + ("RAG + LLM" if use_rag_mode else "LLM Only"))
    print("="*70)
    print("COMMANDS:")
    print("  - Ask any history question")
    print("  - Type 'reset' to start fresh")
    print("  - Type 'stats' to see statistics")
    print("  - Type 'quit' to exit")
    print("="*70 + "\n")

    while True:
        try:
            message = input("\nYou: ").strip()

            if not message:
                continue

            if message.lower() == 'quit':
                response = tutor.chat("goodbye", use_rag=use_rag_mode)
                print(f"\nTutor: {response}")
                break

            if message.lower() == 'reset':
                tutor.reset()
                print("✓ Conversation reset")
                continue

            if message.lower() == 'stats':
                stats = tutor.get_statistics()
                print("\nStatistics:")
                for key, value in stats.items():
                    print(f"  {key}: {value}")
                continue

            response = tutor.chat(message, use_rag=use_rag_mode)
            print(f"\nTutor: {response}")

        except KeyboardInterrupt:
            print("\n\nExiting...")
            break
        except Exception as e:
            print(f"\nError: {e}")
            import traceback
            traceback.print_exc()

    tutor.cleanup()
    print("\nGoodbye!\n")


if __name__ == "__main__":
    main()
