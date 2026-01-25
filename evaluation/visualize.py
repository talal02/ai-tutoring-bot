import json
from pathlib import Path
from typing import Dict, List, Any
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')


class EvaluationVisualizer:
    def __init__(self, output_dir: str = "evaluation/results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.colors = {'base': '#3498db', 'finetuned': '#2ecc71', 'rag': '#e74c3c'}
        self.labels = {'base': 'Base Model', 'finetuned': 'Fine-tuned (LoRA)', 'rag': 'Base + RAG'}

    def load_results(self, path: str) -> Dict:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def create_comparison_chart(self, metrics: Dict, metric_keys: List[str],
                                 metric_labels: List[str], title: str, filename: str) -> str:
        fig, ax = plt.subplots(figsize=(12, 6))
        models = list(metrics.keys())
        x = np.arange(len(metric_keys))
        width = 0.25

        for i, model in enumerate(models):
            values = []
            for key in metric_keys:
                val = metrics[model].get('aggregate_statistics', {}).get(f'{key}_mean', 0)
                values.append(val)
            bars = ax.bar(x + width * i, values, width,
                         label=self.labels.get(model, model),
                         color=self.colors.get(model, '#95a5a6'))
            ax.bar_label(bars, fmt='%.3f', padding=3, fontsize=8)

        ax.set_ylabel('Score')
        ax.set_title(title)
        ax.set_xticks(x + width)
        ax.set_xticklabels(metric_labels, rotation=45, ha='right')
        ax.legend(loc='upper right')
        ax.set_ylim(0, 1.1)
        ax.grid(axis='y', alpha=0.3)
        plt.tight_layout()

        path = self.output_dir / filename
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        return str(path)

    def create_bleu_rouge_chart(self, metrics: Dict) -> str:
        keys = ['bleu_1', 'bleu_2', 'bleu_combined', 'rouge_1', 'rouge_2', 'rouge_l']
        labels = ['BLEU-1', 'BLEU-2', 'BLEU', 'ROUGE-1', 'ROUGE-2', 'ROUGE-L']
        return self.create_comparison_chart(metrics, keys, labels,
                                            'BLEU & ROUGE Score Comparison', 'bleu_rouge_comparison.png')

    def create_semantic_chart(self, metrics: Dict) -> str:
        keys = ['semantic_similarity', 'reference_similarity', 'question_relevance', 'keyword_coverage']
        labels = ['Semantic\nSimilarity', 'Reference\nSimilarity', 'Question\nRelevance', 'Keyword\nCoverage']
        return self.create_comparison_chart(metrics, keys, labels,
                                            'Semantic & Relevance Metrics', 'semantic_comparison.png')

    def create_radar_chart(self, metrics: Dict) -> str:
        categories = ['BLEU', 'ROUGE-1', 'ROUGE-L', 'Semantic\nSimilarity', 'Question\nRelevance']
        keys = ['bleu_combined', 'rouge_1', 'rouge_l', 'semantic_similarity', 'question_relevance']

        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))
        angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
        angles += angles[:1]

        for model, data in metrics.items():
            values = [data.get('aggregate_statistics', {}).get(f'{k}_mean', 0) for k in keys]
            values += values[:1]
            ax.plot(angles, values, 'o-', linewidth=2,
                   label=self.labels.get(model, model), color=self.colors.get(model, '#95a5a6'))
            ax.fill(angles, values, alpha=0.15, color=self.colors.get(model, '#95a5a6'))

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories)
        ax.set_ylim(0, 1)
        ax.set_title('Multi-Metric Model Comparison', size=14, y=1.08)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        plt.tight_layout()

        path = self.output_dir / 'radar_comparison.png'
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        return str(path)

    def create_time_chart(self, results: Dict) -> str:
        fig, ax = plt.subplots(figsize=(10, 6))
        models = list(results.keys())
        times = [np.mean([r['generation_time'] for r in results[m]]) for m in models]
        stds = [np.std([r['generation_time'] for r in results[m]]) for m in models]

        colors = [self.colors.get(m, '#95a5a6') for m in models]
        labels = [self.labels.get(m, m) for m in models]

        bars = ax.bar(range(len(models)), times, yerr=stds, color=colors, capsize=5)
        ax.set_ylabel('Generation Time (seconds)')
        ax.set_title('Average Generation Time per Model')
        ax.set_xticks(range(len(models)))
        ax.set_xticklabels(labels)
        ax.grid(axis='y', alpha=0.3)

        for bar, t in zip(bars, times):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                   f'{t:.2f}s', ha='center', va='bottom')

        plt.tight_layout()
        path = self.output_dir / 'generation_time.png'
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        return str(path)

    def generate_report(self, metrics: Dict, results: Dict = None) -> str:
        lines = []
        lines.append("=" * 70)
        lines.append("MODEL EVALUATION REPORT")
        lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("=" * 70)

        key_metrics = [
            ('bleu_combined', 'BLEU Combined'),
            ('rouge_1', 'ROUGE-1'),
            ('rouge_l', 'ROUGE-L'),
            ('semantic_similarity', 'Semantic Similarity'),
            ('reference_similarity', 'Reference Similarity'),
        ]

        models = list(metrics.keys())
        header = f"{'Metric':<25}" + "".join(f"{self.labels.get(m, m):<20}" for m in models)
        lines.append(header)
        lines.append("-" * 70)

        for key, name in key_metrics:
            row = f"{name:<25}"
            for m in models:
                val = metrics[m].get('aggregate_statistics', {}).get(f'{key}_mean', 0)
                row += f"{val:<20.4f}"
            lines.append(row)

        lines.append("\nBEST MODEL PER METRIC:")
        for key, name in key_metrics:
            best = max(models, key=lambda m: metrics[m].get('aggregate_statistics', {}).get(f'{key}_mean', 0))
            val = metrics[best].get('aggregate_statistics', {}).get(f'{key}_mean', 0)
            lines.append(f"  {name}: {self.labels.get(best, best)} ({val:.4f})")

        weights = {'bleu_combined': 0.2, 'rouge_l': 0.2, 'semantic_similarity': 0.35, 'reference_similarity': 0.25}
        scores = {}
        for m in models:
            score = sum(metrics[m].get('aggregate_statistics', {}).get(f'{k}_mean', 0) * w
                       for k, w in weights.items())
            scores[m] = score

        lines.append("\nOVERALL RANKING:")
        for rank, (m, score) in enumerate(sorted(scores.items(), key=lambda x: x[1], reverse=True), 1):
            marker = " <-- BEST" if rank == 1 else ""
            lines.append(f"  {rank}. {self.labels.get(m, m)}: {score:.4f}{marker}")

        lines.append("=" * 70)
        return "\n".join(lines)

    def generate_all(self, metrics: Dict, results: Dict = None) -> Dict[str, str]:
        outputs = {}
        print("\nGenerating visualizations...")

        print("  - BLEU & ROUGE chart")
        outputs['bleu_rouge'] = self.create_bleu_rouge_chart(metrics)

        print("  - Semantic metrics chart")
        outputs['semantic'] = self.create_semantic_chart(metrics)

        print("  - Radar chart")
        outputs['radar'] = self.create_radar_chart(metrics)

        if results:
            print("  - Generation time chart")
            outputs['time'] = self.create_time_chart(results)

        print("  - Text report")
        report = self.generate_report(metrics, results)
        report_path = self.output_dir / 'evaluation_report.txt'
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)
        outputs['report'] = str(report_path)

        print(f"\nSaved to: {self.output_dir}")
        return outputs


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--metrics', default='evaluation/results/evaluation_metrics.json')
    parser.add_argument('--results', default='evaluation/results/benchmark_results.json')
    parser.add_argument('--output-dir', default='evaluation/results')
    args = parser.parse_args()

    viz = EvaluationVisualizer(args.output_dir)

    metrics_path = Path(__file__).parent.parent / args.metrics
    results_path = Path(__file__).parent.parent / args.results

    metrics = viz.load_results(str(metrics_path)) if metrics_path.exists() else None
    results = viz.load_results(str(results_path)).get('results') if results_path.exists() else None

    if metrics:
        viz.generate_all(metrics, results)
