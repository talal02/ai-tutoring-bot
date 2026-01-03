import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import json


def load_base_model(model_name: str = "meta-llama/Llama-3.1-8B-Instruct"):
    print("Loading base model...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    print("Base model loaded")
    return model, tokenizer


def load_finetuned_model(
    base_model_name: str = "meta-llama/Llama-3.1-8B-Instruct",
    adapter_path: str = "models/finetuned/final"
):
    print("Loading fine-tuned model...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)

    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )

    model = PeftModel.from_pretrained(base_model, adapter_path)
    print("Fine-tuned model loaded")
    return model, tokenizer


def generate_answer(model, tokenizer, question: str, max_tokens: int = 256):
    prompt = f"""<|system|>
You are a helpful and patient history tutor for high school students. Provide clear, accurate, and pedagogically sound explanations.<|end|>
<|user|>
{question}<|end|>
<|assistant|>
"""

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    if "<|assistant|>" in response:
        answer = response.split("<|assistant|>")[-1].strip()
    else:
        answer = response.strip()

    return answer


def compare_models():
    print("Comparing base vs fine-tuned model")
    print("-" * 80)

    test_questions = [
        "What is the difference between primary and secondary sources?",
        "Explain the concept of cause and consequence in history.",
        "How do historians interpret historical events?",
        "What is chronology and why is it important in history?",
        "Describe the process of analyzing historical sources."
    ]

    base_model, base_tokenizer = load_base_model()
    finetuned_model, finetuned_tokenizer = load_finetuned_model()

    for i, question in enumerate(test_questions, 1):
        print(f"\nTest {i}/{len(test_questions)}")
        print(f"Q: {question}\n")

        print("Base model:")
        base_answer = generate_answer(base_model, base_tokenizer, question)
        print(f"{base_answer}\n")

        print("Fine-tuned model:")
        finetuned_answer = generate_answer(finetuned_model, finetuned_tokenizer, question)
        print(f"{finetuned_answer}\n")
        print("-" * 80)

    print("\nComparison complete")


def save_comparison_results():
    print("Saving comparison results...")

    test_questions = [
        "What is the difference between primary and secondary sources?",
        "Explain the concept of cause and consequence in history."
    ]

    base_model, base_tokenizer = load_base_model()
    finetuned_model, finetuned_tokenizer = load_finetuned_model()

    results = []
    for question in test_questions:
        base_answer = generate_answer(base_model, base_tokenizer, question)
        finetuned_answer = generate_answer(finetuned_model, finetuned_tokenizer, question)

        results.append({
            "question": question,
            "base_model_answer": base_answer,
            "finetuned_model_answer": finetuned_answer
        })

    output_file = "data/finetuning/comparison_results.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"Results saved to: {output_file}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--save":
        save_comparison_results()
    else:
        compare_models()
