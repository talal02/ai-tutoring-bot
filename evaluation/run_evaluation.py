import json
import sys
import argparse
from pathlib import Path
from datetime import datetime

sys.path.append(str(Path(__file__).parent.parent / "src"))
sys.path.append(str(Path(__file__).parent))

from metrics import EvaluationMetrics
from benchmark import ModelBenchmark, load_eval_dataset, save_benchmark_results
from visualize import EvaluationVisualizer


def run_evaluation(models=['base', 'finetuned', 'rag'], max_samples=None, output_dir='evaluation/results'):
    output_path = Path(__file__).parent.parent / output_dir
    output_path.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("AI TUTORING BOT - MODEL EVALUATION")
    print("=" * 60)
    print(f"Models: {', '.join(models)}")

    print("\n[1/4] Loading dataset...")
    dataset_path = Path(__file__).parent / "eval_dataset.json"
    eval_dataset = load_eval_dataset(str(dataset_path))
    if max_samples:
        eval_dataset = eval_dataset[:max_samples]
    print(f"  Loaded {len(eval_dataset)} samples")

    print("\n[2/4] Running benchmarks...")
    benchmark = ModelBenchmark()

    if 'base' in models or 'rag' in models:
        benchmark.load_base_model()
    if 'finetuned' in models:
        benchmark.load_finetuned_model()
    if 'rag' in models:
        benchmark.setup_rag()

    available = [m for m in models if m in benchmark.models]
    if not available:
        print("No models available!")
        return {}

    results = benchmark.run_benchmark(eval_dataset, available, max_samples)
    save_benchmark_results(results, str(output_path / "benchmark_results.json"))

    print("\n[3/4] Calculating metrics...")
    embedder = None
    if benchmark.rag_components and 'embedder' in benchmark.rag_components:
        embedder = benchmark.rag_components['embedder']

    calculator = EvaluationMetrics(embedder=embedder)
    metrics = {}

    for model_name in available:
        questions = [r.question for r in results[model_name]]
        references = [r.reference_answer for r in results[model_name]]
        candidates = [r.generated_answer for r in results[model_name]]
        metrics[model_name] = calculator.evaluate_batch(questions, references, candidates)

        agg = metrics[model_name]['aggregate_statistics']
        print(f"\n  {model_name}:")
        print(f"    BLEU: {agg.get('bleu_combined_mean', 0):.4f}")
        print(f"    ROUGE-L: {agg.get('rouge_l_mean', 0):.4f}")
        if 'semantic_similarity_mean' in agg:
            print(f"    Semantic: {agg.get('semantic_similarity_mean', 0):.4f}")

    with open(output_path / "evaluation_metrics.json", 'w') as f:
        json.dump(metrics, f, indent=2)

    print("\n[4/4] Generating visualizations...")
    visualizer = EvaluationVisualizer(str(output_path))
    results_dict = {
        m: [{'question_id': r.question_id, 'question': r.question,
             'reference_answer': r.reference_answer, 'generated_answer': r.generated_answer,
             'generation_time': r.generation_time} for r in res]
        for m, res in results.items()
    }
    visualizer.generate_all(metrics, results_dict)

    print_summary(metrics, available)
    benchmark.cleanup()

    print("\n" + "=" * 60)
    print("EVALUATION COMPLETE")
    print(f"Results saved to: {output_path}")
    print("=" * 60)


def print_summary(metrics, models):
    labels = {'base': 'Base Model', 'finetuned': 'Fine-tuned', 'rag': 'Base + RAG'}

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    weights = {'bleu_combined': 0.2, 'rouge_l': 0.2, 'semantic_similarity': 0.35, 'reference_similarity': 0.25}
    scores = {}

    for m in models:
        agg = metrics[m].get('aggregate_statistics', {})
        score = sum(agg.get(f'{k}_mean', 0) * w for k, w in weights.items())
        scores[m] = score

    print("\nOverall Ranking:")
    for rank, (m, score) in enumerate(sorted(scores.items(), key=lambda x: x[1], reverse=True), 1):
        marker = " <-- BEST" if rank == 1 else ""
        print(f"  {rank}. {labels.get(m, m)}: {score:.4f}{marker}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--models', nargs='+', choices=['base', 'finetuned', 'rag'],
                        default=['base', 'finetuned', 'rag'])
    parser.add_argument('--max-samples', type=int, default=None)
    parser.add_argument('--quick', action='store_true', help='Quick test with 5 samples')
    parser.add_argument('--output-dir', default='evaluation/results')
    args = parser.parse_args()

    max_samples = 5 if args.quick else args.max_samples
    run_evaluation(args.models, max_samples, args.output_dir)
