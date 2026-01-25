import re
from typing import List, Dict, Any, Tuple
from collections import Counter
import numpy as np


class EvaluationMetrics:
    def __init__(self, embedder=None):
        self.embedder = embedder

    def tokenize(self, text: str) -> List[str]:
        text = text.lower()
        text = re.sub(r'[^\w\s]', ' ', text)
        return [t for t in text.split() if t]

    def get_ngrams(self, tokens: List[str], n: int) -> List[Tuple[str, ...]]:
        if len(tokens) < n:
            return []
        return [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]

    def calculate_bleu(self, reference: str, candidate: str, max_n: int = 4) -> Dict[str, float]:
        ref_tokens = self.tokenize(reference)
        cand_tokens = self.tokenize(candidate)

        if len(cand_tokens) == 0:
            return {f'bleu_{i}': 0.0 for i in range(1, max_n + 1)} | {'bleu_combined': 0.0}

        bp = 1.0
        if len(cand_tokens) < len(ref_tokens):
            bp = np.exp(1 - len(ref_tokens) / len(cand_tokens))

        precisions = []
        results = {}

        for n in range(1, max_n + 1):
            ref_ngrams = Counter(self.get_ngrams(ref_tokens, n))
            cand_ngrams = Counter(self.get_ngrams(cand_tokens, n))

            if len(cand_ngrams) == 0:
                precision = 0.0
            else:
                clipped = sum(min(count, ref_ngrams.get(ng, 0)) for ng, count in cand_ngrams.items())
                precision = clipped / sum(cand_ngrams.values())

            precisions.append(precision)
            results[f'bleu_{n}'] = precision

        weights = [1.0 / max_n] * max_n
        if all(p > 0 for p in precisions):
            log_prec = [w * np.log(p) for w, p in zip(weights, precisions)]
            combined = bp * np.exp(sum(log_prec))
        else:
            combined = 0.0

        results['bleu_combined'] = combined
        return results

    def calculate_rouge(self, reference: str, candidate: str) -> Dict[str, float]:
        ref_tokens = self.tokenize(reference)
        cand_tokens = self.tokenize(candidate)

        return {
            'rouge_1': self._rouge_n(ref_tokens, cand_tokens, 1),
            'rouge_2': self._rouge_n(ref_tokens, cand_tokens, 2),
            'rouge_l': self._rouge_l(ref_tokens, cand_tokens)
        }

    def _rouge_n(self, ref_tokens: List[str], cand_tokens: List[str], n: int) -> float:
        ref_ngrams = Counter(self.get_ngrams(ref_tokens, n))
        cand_ngrams = Counter(self.get_ngrams(cand_tokens, n))

        if not ref_ngrams or not cand_ngrams:
            return 0.0

        overlap = sum(min(ref_ngrams[ng], cand_ngrams[ng]) for ng in ref_ngrams if ng in cand_ngrams)
        precision = overlap / sum(cand_ngrams.values()) if cand_ngrams else 0
        recall = overlap / sum(ref_ngrams.values()) if ref_ngrams else 0

        if precision + recall == 0:
            return 0.0
        return 2 * precision * recall / (precision + recall)

    def _rouge_l(self, ref_tokens: List[str], cand_tokens: List[str]) -> float:
        if not ref_tokens or not cand_tokens:
            return 0.0

        m, n = len(ref_tokens), len(cand_tokens)
        dp = [[0] * (n + 1) for _ in range(m + 1)]

        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if ref_tokens[i-1] == cand_tokens[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])

        lcs = dp[m][n]
        precision = lcs / n if n > 0 else 0
        recall = lcs / m if m > 0 else 0

        if precision + recall == 0:
            return 0.0
        return 2 * precision * recall / (precision + recall)

    def calculate_semantic_similarity(self, reference: str, candidate: str) -> float:
        if self.embedder is None:
            return 0.0
        try:
            ref_emb = self.embedder.embed_text(reference)
            cand_emb = self.embedder.embed_text(candidate)
            sim = np.dot(ref_emb, cand_emb) / (np.linalg.norm(ref_emb) * np.linalg.norm(cand_emb))
            return float(sim)
        except:
            return 0.0

    def calculate_answer_relevance(self, question: str, answer: str, reference: str = None) -> Dict[str, float]:
        q_tokens = set(self.tokenize(question))
        a_tokens = set(self.tokenize(answer))

        results = {}
        results['keyword_coverage'] = len(q_tokens & a_tokens) / len(q_tokens) if q_tokens else 0.0

        if self.embedder:
            results['question_relevance'] = self.calculate_semantic_similarity(question, answer)
            if reference:
                results['reference_similarity'] = self.calculate_semantic_similarity(reference, answer)

        return results

    def evaluate(self, question: str, reference: str, candidate: str) -> Dict[str, Any]:
        results = {}
        results.update(self.calculate_bleu(reference, candidate))
        results.update(self.calculate_rouge(reference, candidate))

        if self.embedder:
            results['semantic_similarity'] = self.calculate_semantic_similarity(reference, candidate)

        results.update(self.calculate_answer_relevance(question, candidate, reference))
        results['word_count'] = len(self.tokenize(candidate))

        return results

    def evaluate_batch(self, questions: List[str], references: List[str], candidates: List[str]) -> Dict[str, Any]:
        individual = []
        for q, r, c in zip(questions, references, candidates):
            individual.append(self.evaluate(q, r, c))

        metric_keys = ['bleu_1', 'bleu_2', 'bleu_3', 'bleu_4', 'bleu_combined',
                       'rouge_1', 'rouge_2', 'rouge_l', 'semantic_similarity',
                       'keyword_coverage', 'question_relevance', 'reference_similarity']

        aggregates = {}
        for key in metric_keys:
            values = [r.get(key, 0) for r in individual if key in r]
            if values:
                aggregates[f'{key}_mean'] = np.mean(values)
                aggregates[f'{key}_std'] = np.std(values)

        return {
            'individual_results': individual,
            'aggregate_statistics': aggregates,
            'num_samples': len(individual)
        }
