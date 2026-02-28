import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Any
from dataclasses import dataclass, field

import torch
from tqdm import tqdm

sys.path.append(str(Path(__file__).parent.parent / "src"))

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


@dataclass
class BenchmarkResult:
    model_name: str
    question_id: int
    question: str
    reference_answer: str
    generated_answer: str
    generation_time: float
    metrics: Dict[str, float] = field(default_factory=dict)


class ModelBenchmark:
    def __init__(self, base_model_name: str = "meta-llama/Meta-Llama-3.1-8B-Instruct",
                 adapter_path: str = "models/finetuned_8b/final",
                 config_path: str = None):
        self.base_model_name = base_model_name
        self.adapter_path = adapter_path
        if config_path is None:
            self.config_path = str(Path(__file__).parent.parent / "configs" / "config.yaml")
        else:
            self.config_path = config_path
        self.models = {}
        self.tokenizers = {}
        self.rag_components = None
        self.system_prompt = "You are a helpful and patient history tutor for high school students. Provide clear, accurate, and pedagogically sound explanations."

    def load_base_model(self):
        print(f"\nLoading base model: {self.base_model_name}")
        tokenizer = AutoTokenizer.from_pretrained(self.base_model_name, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            self.base_model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        self.models['base'] = model
        self.tokenizers['base'] = tokenizer
        print("Base model loaded")

    def load_finetuned_model(self):
        print(f"\nLoading fine-tuned model from: {self.adapter_path}")
        adapter_full_path = Path(__file__).parent.parent / self.adapter_path

        if not adapter_full_path.exists():
            print(f"Adapter not found at {adapter_full_path}")
            return

        tokenizer = AutoTokenizer.from_pretrained(self.base_model_name, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        base_model = AutoModelForCausalLM.from_pretrained(
            self.base_model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        model = PeftModel.from_pretrained(base_model, str(adapter_full_path))
        self.models['finetuned'] = model
        self.tokenizers['finetuned'] = tokenizer
        print("Fine-tuned model loaded")

    def setup_rag(self, use_finetuned=False):
        print(f"\nSetting up RAG system with {'finetuned' if use_finetuned else 'base'} model...")
        try:
            from utils.config import get_config
            from rag.embedder import Embedder
            from rag.retriever import FAISSRetriever
            from rag.document_processor import DocumentProcessor

            config = get_config(self.config_path)
            doc_processor = DocumentProcessor(config.rag)
            embedder = Embedder(config.rag)
            embedder.load_model()
            retriever = FAISSRetriever(config.rag, embedder)

            try:
                retriever.load_index("history_index")
                print("Loaded existing RAG index")
            except FileNotFoundError:
                print("Building RAG index...")
                dataset_path = Path(__file__).parent.parent / "questions_dataset.json"
                if dataset_path.exists():
                    documents = doc_processor.load_json_dataset(str(dataset_path))
                    processed = doc_processor.process_documents(documents, chunk=True)
                    retriever.build_index(processed, show_progress=True)
                    retriever.save_index("history_index")

            self.rag_components = {'retriever': retriever, 'embedder': embedder}

            if use_finetuned:
                if 'finetuned' not in self.models:
                    self.load_finetuned_model()
                self.models['finetuned_rag'] = self.models['finetuned']
                self.tokenizers['finetuned_rag'] = self.tokenizers['finetuned']
                print("RAG setup complete with finetuned model")
            else:
                if 'base' not in self.models:
                    self.load_base_model()
                self.models['rag'] = self.models['base']
                self.tokenizers['rag'] = self.tokenizers['base']
                print("RAG setup complete with base model")

        except Exception as e:
            print(f"Could not setup RAG: {e}")

    def format_prompt(self, question: str, context: str = None) -> str:
        if context:
            user_content = f"Based on the following context, answer the question.\n\nContext:\n{context}\n\nQuestion: {question}"
        else:
            user_content = question

        return f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

{self.system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>

{user_content}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""

    def generate(self, model_key: str, question: str, max_new_tokens: int = 512) -> tuple:
        if model_key not in self.models:
            return f"Model '{model_key}' not loaded", 0.0

        model = self.models[model_key]
        tokenizer = self.tokenizers[model_key]

        context = None
        if model_key in ['rag', 'finetuned_rag'] and self.rag_components:
            retriever = self.rag_components['retriever']
            results = retriever.retrieve(question, top_k=5)
            context = retriever.format_retrieved_context(results)

        prompt = self.format_prompt(question, context)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        start = time.time()
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        gen_time = time.time() - start

        full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)

        if "assistant" in full_response.lower():
            answer = full_response.split("assistant")[-1].strip()
        else:
            answer = full_response[len(prompt):].strip()

        for token in ['<|eot_id|>', '<|end_of_text|>', '</s>']:
            answer = answer.replace(token, '')

        return answer.strip(), gen_time

    def run_benchmark(self, eval_dataset: List[Dict], model_keys: List[str] = None,
                      max_samples: int = None) -> Dict[str, List[BenchmarkResult]]:
        if model_keys is None:
            model_keys = list(self.models.keys())
        if max_samples:
            eval_dataset = eval_dataset[:max_samples]

        results = {key: [] for key in model_keys}

        print(f"\nRunning benchmark on {len(eval_dataset)} samples")
        print(f"Models: {', '.join(model_keys)}\n")

        for sample in tqdm(eval_dataset, desc="Evaluating"):
            question = sample['question']
            reference = sample['reference_answer']
            qid = sample.get('id', 0)

            for model_key in model_keys:
                answer, gen_time = self.generate(model_key, question)

                result = BenchmarkResult(
                    model_name=model_key,
                    question_id=qid,
                    question=question,
                    reference_answer=reference,
                    generated_answer=answer,
                    generation_time=gen_time
                )
                results[model_key].append(result)

        return results

    def cleanup(self):
        print("\nCleaning up...")
        for key in list(self.models.keys()):
            del self.models[key]
            del self.tokenizers[key]

        if self.rag_components and 'embedder' in self.rag_components:
            self.rag_components['embedder'].unload_model()
            self.rag_components = None

        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def load_eval_dataset(path: str) -> List[Dict]:
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_benchmark_results(results: Dict[str, List[BenchmarkResult]], output_path: str):
    from datetime import datetime
    output = {'timestamp': datetime.now().isoformat(), 'results': {}}

    for model_key, model_results in results.items():
        output['results'][model_key] = [
            {
                'question_id': r.question_id,
                'question': r.question,
                'reference_answer': r.reference_answer,
                'generated_answer': r.generated_answer,
                'generation_time': r.generation_time
            }
            for r in model_results
        ]

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"Results saved to: {output_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='evaluation/eval_dataset.json')
    parser.add_argument('--output', default='evaluation/results/benchmark_results.json')
    parser.add_argument('--models', nargs='+', choices=['base', 'finetuned', 'rag', 'finetuned_rag'],
                        default=['base', 'finetuned', 'rag', 'finetuned_rag'])
    parser.add_argument('--max-samples', type=int, default=None)
    args = parser.parse_args()

    benchmark = ModelBenchmark()

    if 'base' in args.models or 'rag' in args.models:
        benchmark.load_base_model()
    if 'finetuned' in args.models or 'finetuned_rag' in args.models:
        benchmark.load_finetuned_model()
    if 'rag' in args.models:
        benchmark.setup_rag(use_finetuned=False)
    if 'finetuned_rag' in args.models:
        benchmark.setup_rag(use_finetuned=True)

    dataset_path = Path(__file__).parent.parent / args.dataset
    eval_dataset = load_eval_dataset(str(dataset_path))

    available = [m for m in args.models if m in benchmark.models]
    results = benchmark.run_benchmark(eval_dataset, available, args.max_samples)

    output_path = Path(__file__).parent.parent / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_benchmark_results(results, str(output_path))

    benchmark.cleanup()
