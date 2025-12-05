import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from llm.model_loader import ModelLoader
from llm.generator import LLMGenerator
from dialogue.dialogue_manager import DialogueManager
from dialogue.intent_detector import Intent
from utils.config import Config
from utils.logger import get_logger

logger = get_logger(__name__)


class TutorDemo:
    def __init__(self, config_path: str = "./configs/config.yaml"):
        self.config = Config(config_path)
        self.config.ensure_directories()
        print("Configuration loaded")

        self.model_loader = None
        self.generator = None
        self.dialogue_manager = DialogueManager()
        print("Dialogue manager initialized")

        self.llm_ready = False

    def setup_llm(self):
        print(f"  Model: {self.config.llm.model_name}")
        print(f"  Device: {self.config.llm.device}")

        try:
            self.model_loader = ModelLoader(self.config.llm)
            model, tokenizer = self.model_loader.load_model()

            self.generator = LLMGenerator(
                model=model,
                tokenizer=tokenizer,
                llm_config=self.config.llm,
                prompts_config=self.config.prompts,
            )

            self.llm_ready = True

            model_size = self.model_loader.get_model_size()
            memory_info = self.model_loader.get_memory_usage()

            print(f"Model loaded successfully")
            print(f"  Parameters: {model_size:.2f}B")
            print(f"  Device: {memory_info['device']}")

            if memory_info['device'] == 'cuda':
                print(f"  GPU Memory: {memory_info.get('allocated_gb', 0):.2f} GB")

        except Exception as e:
            print(f"✗ Error loading model: {e}")
            raise

    def generate_response(self, user_message: str) -> str:
        if not self.llm_ready:
            return "Error: LLM not initialized"

        dialogue_result = self.dialogue_manager.process_message(user_message)
        intent = dialogue_result['intent']
        strategy = dialogue_result['strategy']
        if intent == Intent.GREETING:
            response = "Hello! I'm your history tutor. How can I help you today?"

        elif intent == Intent.QUIT:
            response = "Goodbye! Keep up the great work with your history studies!"

        elif intent == Intent.HELP:
            response = (
                "I'm here to help you learn history! You can:\n"
                "- Ask me questions about historical topics\n"
                "- Request explanations or examples\n"
                "- Ask for hints if you're stuck\n"
                "What would you like to know?"
            )

        elif intent == Intent.THANKS:
            response = "You're welcome! Feel free to ask if you have more questions."

        elif intent == Intent.REQUEST_HINT:
            question = strategy.get('question', user_message)
            student_response = strategy.get('student_response', '')
            hint_level = strategy.get('hint_level', 'partial')
            response = self.generator.generate_hint(
                question=question,
                student_response=student_response,
                hint_level=hint_level,
            )

        elif intent == Intent.QUESTION or intent == Intent.EXPLANATION:
            history = self.dialogue_manager.get_conversation_history(n=3)
            prompt = self.generator.format_chat_prompt(
                user_message=user_message,
                history=history,
            )
            response = self.generator.generate(prompt)

        elif intent == Intent.EXAMPLE:
            prompt = self.generator.format_chat_prompt(
                user_message=f"Please provide a clear example of: {user_message}",
            )
            response = self.generator.generate(prompt)

        elif intent == Intent.CLARIFICATION:
            history = self.dialogue_manager.get_conversation_history(n=2)
            prompt = self.generator.format_chat_prompt(
                user_message=f"Please clarify: {user_message}",
                history=history,
            )
            response = self.generator.generate(prompt)

        else:
            prompt = self.generator.format_chat_prompt(user_message=user_message)
            response = self.generator.generate(prompt)
        self.dialogue_manager.record_response(
            user_message=user_message,
            intent=intent,
            system_response=response,
        )

        return response

    def run_interactive_mode(self):
        print("\nYou can now chat with the tutor!")
        print("Type 'quit' or 'exit' to end the conversation.")
        print("Type 'stats' to see conversation statistics.\n")

        while True:
            try:
                user_input = input("\n You: ").strip()
                if not user_input:
                    continue

                if user_input.lower() in ['quit', 'exit', 'bye']:
                    print("\n🤖 Tutor: Goodbye! Keep learning!")
                    break

                if user_input.lower() == 'stats':
                    stats = self.dialogue_manager.get_statistics()
                    print("Conversation Statistics")
                    print(f"Total turns: {stats['conversation']['total_turns']}")
                    print(f"Duration: {stats['conversation']['session_duration_minutes']:.2f} min")
                    print("\nIntent distribution:")
                    for intent, count in stats['conversation']['intent_distribution'].items():
                        print(f"  {intent}: {count}")
                    continue

                response = self.generate_response(user_input)
                print(f"\n🤖 Tutor: {response}")

            except KeyboardInterrupt:
                print("\n\nInterrupted by user.")
                break
            except Exception as e:
                print(f"\nError: {e}")
                logger.error(f"Error in interactive mode: {e}", exc_info=True)

    def cleanup(self):
        if self.model_loader:
            self.model_loader.unload_model()
        print("\nResources cleaned up")


def main():
    demo = TutorDemo(config_path="./configs/config.yaml")

    try:
        demo.setup_llm()
        demo.run_interactive_mode()

        demo.cleanup()
    except KeyboardInterrupt:
        print("\n\nDemo interrupted by user")
        demo.cleanup()
    except Exception as e:
        print(f"\n✗ Demo failed: {e}")
        logger.error(f"Demo failed: {e}", exc_info=True)
        demo.cleanup()
        sys.exit(1)


if __name__ == "__main__":
    main()
