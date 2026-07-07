# AI History Tutoring Bot

A conversational AI tutor for high-school history that combines a fine-tuned LLM with Retrieval-Augmented Generation (RAG) to give curriculum-grounded explanations, adaptive hints, and answer assessment through a chat interface.

## Overview

The system pairs a LoRA fine-tuned **Llama-3.1-8B-Instruct** model with a **FAISS-based RAG pipeline** built from curriculum PDFs and a custom Q&A dataset, so responses stay grounded in actual course material instead of relying purely on the base model's knowledge. On top of that sits a dialogue layer that detects user intent (question, quiz request, answer to assess, request for a hint, greeting, etc.) and an assessment engine that generates leveled hints, evaluates student answers, and flags common historical-reasoning errors (presentism, anachronism, oversimplification, false causation, lack of evidence).

The bot is served through a FastAPI backend with a lightweight web chat UI, and includes a full evaluation suite comparing the base model, base model + RAG, and fine-tuned model + RAG across BLEU/ROUGE, semantic similarity, source grounding, and hallucination rate.

## Features

- **RAG-grounded answers**: retrieves relevant chunks from curriculum PDFs and a Q&A dataset (FAISS + `sentence-transformers/all-MiniLM-L6-v2`) and injects them into the prompt.
- **Fine-tuned LLM**: LoRA/PEFT fine-tuning of Llama-3.1-8B-Instruct on a custom history Q&A dataset, with a separate script for dataset preparation and training.
- **Dialogue management**: intent detection (question, quiz request, greeting, help, repeat, thanks, etc.) drives which strategy the tutor uses to respond.
- **Adaptive hinting**: three hint levels (nudge, partial, full) so the tutor can guide a student toward an answer without giving it away.
- **Answer assessment**: checks student answers against curriculum context and flags specific reasoning errors rather than just marking right/wrong.
- **Quiz generation**: generates single, well-formed quiz questions on request, grounded in retrieved context where available.
- **Evaluation suite**: benchmarking scripts and reports comparing base / base+RAG / finetuned+RAG on BLEU, ROUGE, semantic similarity, source grounding, and hallucination rate (via NLI).
- **Web UI + API**: FastAPI backend (`/api/chat/message`, file upload endpoints) with a simple HTML/CSS/JS chat frontend.

## Architecture

```
User (Web UI) → FastAPI (api/main.py) → HistoryTutor (src/tutor.py)
                                              │
                    ┌─────────────────────────┼─────────────────────────┐
                    ▼                         ▼                         ▼
            Dialogue Manager            RAG Retriever              LLM Generator
        (intent detection, state)   (FAISS + embeddings)      (fine-tuned or base model)
                    │
                    ▼
            Assessment Engine
      (hints, answer grading, error analysis)
```

## Project Structure

| Path | Description |
|---|---|
| `api/main.py` | FastAPI app: chat endpoint, model switching, file upload for new source documents. |
| `src/tutor.py` | `HistoryTutor` orchestrator tying together the LLM, RAG, dialogue, and assessment layers. |
| `src/llm/` | Model loading (`model_loader.py`) and generation (`generator.py`), including prompt formatting and chat history. |
| `src/rag/` | Document processing/chunking, embedding, and FAISS retrieval (`document_processor.py`, `embedder.py`, `retriever.py`). |
| `src/dialogue/` | Intent detection and dialogue state management. |
| `src/assessment/` | Hint generation, answer assessment, and historical-reasoning error analysis. |
| `scripts/finetune.py` | LoRA fine-tuning script for the base LLM using `transformers` + `peft`. |
| `scripts/prepare_data.py` | Prepares training/validation data for fine-tuning. |
| `evaluation/` | Benchmarking, metrics (BLEU/ROUGE/semantic similarity), grounding analysis, and hallucination detection, with generated reports and plots. |
| `configs/config.yaml` | Central configuration: model choice, generation parameters, RAG settings, prompts, assessment behavior. |
| `data/` | Curriculum PDFs, the Q&A dataset, fine-tuning data, and the FAISS vector store. |
| `web/` | Static HTML/CSS/JS chat frontend served alongside the API. |

## Requirements

- Python 3.10+
- A CUDA-capable GPU is recommended for running/fine-tuning the 8B-parameter base model (configurable to run on CPU via `configs/config.yaml`).
- See `requirements.txt` for the full dependency list (key libraries: `transformers`, `peft`, `bitsandbytes`, `faiss-cpu`, `sentence-transformers`, `fastapi`, `torch`).

## Setup

```bash
pip install -r requirements.txt
```

Configure the model, RAG, and assessment behavior in `configs/config.yaml` (base model name, generation parameters, chunking/retrieval settings, hint levels, error categories to check for).

## Running

Start the API + web UI:

```bash
python -m api.main
```

Or run the tutor as an interactive CLI:

```bash
cd src
python tutor.py --use-rag
```

CLI commands: type a history question directly, `reset` to clear conversation state, `stats` to view session statistics, or `quit` to exit.

## Fine-Tuning

```bash
python scripts/prepare_data.py   # builds train/val JSONL from the Q&A dataset
python scripts/finetune.py       # LoRA fine-tunes the base model, saves adapter to models/finetuned/
```

## Evaluation

```bash
python evaluation/run_evaluation.py
```

Sample results comparing base model, base model + RAG, and fine-tuned model + RAG (see `evaluation/results/`, `evaluation/grounding_results/`, `evaluation/hallucination_results/` for full reports):

- **Answer quality**: fine-tuned + RAG scored highest overall across BLEU/ROUGE/semantic-similarity-based ranking.
- **Source grounding**: adding RAG raised grounded-sentence rate from 0% (base model, no sources) to ~82% of sentences classified as grounded against retrieved sources.
- **Hallucination rate**: adding RAG reduced the measured hallucination rate from ~1.2% to ~0.2% of sentences (NLI-based contradiction detection against reference answers).
