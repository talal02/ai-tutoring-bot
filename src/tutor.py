"""
Demo: History Tutoring System
Core integration of LLM, RAG, and Dialogue layers.
"""

from pathlib import Path
from typing import Optional, Dict, Any
import sys

sys.path.append(str(Path(__file__).parent))

from utils.config import get_config
from utils.logger import setup_logger, get_logger

# LLM and RAG
from llm.model_loader import ModelLoader
from llm.generator import LLMGenerator
from rag.document_processor import DocumentProcessor
from rag.embedder import Embedder
from rag.retriever import FAISSRetriever

# Dialogue Management
from dialogue.dialogue_manager import DialogueManager
from dialogue.intent_detector import Intent

logger = get_logger(__name__)


class HistoryTutor:
    """
    Demo: Integrated History Tutoring System

    Demonstrates:
    - LLM Layer: Multi-model support with quantization
    - RAG System: FAISS-based retrieval with embeddings
    - Dialogue Management: Intent detection and conversation flow
    """

    def __init__(self, config_path: Optional[str] = None):
        """Initialize the demo tutoring system."""
        self.config = get_config(config_path)

        self.logger = setup_logger(
            __name__,
            level=self.config.logging.level,
            log_file=self.config.logging.log_file,
            console_output=self.config.logging.console_output,
        )

        self.logger.info("="*70)
        self.logger.info("Demo: History Tutoring System")
        self.logger.info("="*70)

        # Core components
        self.model_loader = None
        self.generator = None
        self.doc_processor = None
        self.embedder = None
        self.retriever = None
        self.dialogue_manager = DialogueManager()

        self.logger.info("system initialized")

    def setup_llm(self, adapter_path: Optional[str] = None) -> None:
        """Setup LLM layer with optional LoRA adapter."""
        self.logger.info("Setting up LLM layer...")

        self.model_loader = ModelLoader(self.config.llm)
        model, tokenizer = self.model_loader.load_model(adapter_path=adapter_path)

        self.generator = LLMGenerator(
            model=model,
            tokenizer=tokenizer,
            llm_config=self.config.llm,
            prompts_config=self.config.prompts,
        )

        memory_stats = self.model_loader.get_memory_usage()
        self.logger.info(f"LLM setup complete. Memory: {memory_stats}")

    def setup_rag(
        self,
        dataset_path: Optional[str] = None,
        pdf_directory: Optional[str] = None,
        index_name: str = "history_index",
        rebuild_index: bool = False,
    ) -> None:
        """Setup RAG system with FAISS retrieval."""
        self.doc_processor = DocumentProcessor(self.config.rag)
        self.embedder = Embedder(self.config.rag)
        self.embedder.load_model()
        self.retriever = FAISSRetriever(self.config.rag, self.embedder)

        # Try to load existing index
        if not rebuild_index:
            try:
                self.retriever.load_index(index_name)
                self.logger.info("Loaded existing RAG index")
                return
            except FileNotFoundError:
                self.logger.info("No existing index found, building new one")

        # Build new index from multiple sources
        all_documents = []
        
        # Load JSON dataset if provided
        if dataset_path:
            documents = self.doc_processor.load_json_dataset(dataset_path)
            all_documents.extend(documents)
            self.logger.info(f"Loaded {len(documents)} documents from JSON dataset")
        
        # Load PDFs from directory if provided
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

    def chat(self, message: str) -> str:
        """
        Main chat interface demonstrating all three layers.

        Flow:
        1. Dialogue layer detects intent
        2. RAG layer retrieves relevant context
        3. LLM layer generates response
        """
        self.logger.info(f"\n{'='*70}\nUser: {message}\n{'='*70}")

        # Dialogue layer: Process message and detect intent
        processing_result = self.dialogue_manager.process_message(message)
        intent = processing_result['intent']
        strategy = processing_result['strategy']

        self.logger.info(f"Intent detected: {intent.value}")

        # Handle simple intents without LLM
        if intent == Intent.GREETING:
            response = self._handle_greeting()
        elif intent == Intent.QUIT:
            response = self._handle_farewell()
        elif intent == Intent.HELP:
            response = self._handle_help()
        elif intent == Intent.THANKS:
            response = "You're welcome! What else would you like to learn about?"
        else:
            # RAG layer: Retrieve relevant context
            context = None
            if strategy.get('use_rag') and self.retriever:
                results = self.retriever.retrieve(message, top_k=5)
                context = self.retriever.format_retrieved_context(results)
                self.logger.info(f"Retrieved {len(results)} context documents")

            # LLM layer: Generate response
            if self.generator:
                if context:
                    response = self.generator.generate_with_rag(
                        question=message,
                        context=context,
                        temperature=0.7,
                    )
                else:
                    prompt = self.generator.format_chat_prompt(user_message=message)
                    response = self.generator.generate(prompt)
            else:
                response = "I'm ready to help! Please set up the LLM first."

        # Record interaction
        self.dialogue_manager.record_response(
            user_message=message,
            intent=intent,
            system_response=response,
            metadata={'action': strategy['action']},
        )

        self.logger.info(f"Response: {response[:100]}...")
        return response

    def _handle_greeting(self) -> str:
        """Handle greeting intent."""
        return (
            "Hello! I'm your history tutor. I can help you learn about historical sources, "
            "chronology, causation, and more. What would you like to explore today?"
        )

    def _handle_farewell(self) -> str:
        """Handle farewell intent."""
        stats = self.dialogue_manager.get_statistics()
        return (
            f"Great work today! We covered {stats['conversation']['total_turns']} exchanges. "
            "Keep learning! Goodbye!"
        )

    def _handle_help(self) -> str:
        """Handle help request."""
        return (
            "I can help you with:\n"
            "• Answering history questions\n"
            "• Explaining historical concepts\n"
            "• Providing examples from historical sources\n\n"
            "Just ask me anything about history!"
        )

    def reset(self) -> None:
        """Reset conversation state."""
        self.dialogue_manager.reset_conversation()
        if self.generator:
            self.generator.clear_history()
        self.logger.info("System reset")

    def get_statistics(self) -> Dict[str, Any]:
        """Get system statistics."""
        stats = {'dialogue': self.dialogue_manager.get_statistics()}

        if self.model_loader:
            stats['llm_memory'] = self.model_loader.get_memory_usage()

        if self.retriever:
            stats['rag'] = self.retriever.get_statistics()

        return stats

    def cleanup(self) -> None:
        """Cleanup resources."""
        self.logger.info("Cleaning up resources...")

        if self.model_loader:
            self.model_loader.unload_model()

        if self.embedder:
            self.embedder.unload_model()

        self.logger.info("Cleanup complete")


def main():
    """Interactive CLI for demo."""
    import argparse

    parser = argparse.ArgumentParser(description="Demo: History Tutoring System")
    parser.add_argument(
        "--config",
        type=str,
        default="../configs/config.yaml",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="../questions_dataset.json",
        help="Path to dataset file",
    )
    parser.add_argument(
        "--rebuild-index",
        action="store_true",
        help="Rebuild RAG index",
    )
    args = parser.parse_args()

    print("\nInitializing...\n")

    tutor = HistoryTutor(config_path=args.config)

    print("Loading language model...")
    tutor.setup_llm()

    print("\nSetting up RAG system...")
    tutor.setup_rag(dataset_path=args.dataset, rebuild_index=args.rebuild_index, pdf_directory="./data")

    print("\n" + "="*70)
    print("SYSTEM READY")
    print("="*70)

    stats = tutor.get_statistics()
    print("\nSystem Information:")
    for key, value in stats.items():
        print(f"  {key}: {value}")

    print("\n" + "="*70)
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
                response = tutor.chat("goodbye")
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

            response = tutor.chat(message)
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
