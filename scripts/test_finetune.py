import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import json


def load_base_model(model_name: str = "meta-llama/Llama-3.2-3B-Instruct"):
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
    base_model_name: str = "meta-llama/Llama-3.2-3B-Instruct",
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


def generate_answer(model, tokenizer, question: str, max_tokens: int = 1024):
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
    
    if "<|end|>" in answer:
        answer = answer.split("<|end|>")[0].strip()

    return answer


def compare_models():
    print("Comparing base vs fine-tuned model")
    print("-" * 80)

    test_questions = [
        "What does the architectural evidence of the Berlin Wall reveal about the nature of division between East and West?",
        "What does the evidence of hyperinflation in 1923 reveal about the Weimar Republic's economic problems?",
        "What do the terms of the 1970 Moscow Treaty and 1970 Warsaw Treaty reveal about Willy Brandt's approach to improving East-West relations?",
        "What do images and descriptions of East German housing reveal about life under communist rule?",
        "DWhat was Helmut Kohl's 10 Point Plan proposed on 28th November 1989, and what did it aim to achieve?"
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



if __name__ == "__main__":
    compare_models()
