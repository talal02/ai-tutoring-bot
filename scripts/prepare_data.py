import json
import random
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent / "src"))
from rag.document_processor import DocumentProcessor
from utils.config import RAGConfig

def extract_pdf_training_data(pdf_directory: str) -> list:
    config = RAGConfig()
    doc_processor = DocumentProcessor(config)
    pdf_path = Path(pdf_directory)
    pdf_files = list(pdf_path.glob("*.pdf"))
    print(f"Found {len(pdf_files)} PDF files")

    training_examples = []

    for pdf_file in pdf_files:
        try:
            print(f"  Processing {pdf_file.name}...")
            document = doc_processor.load_pdf_file(str(pdf_file))
            chunks = doc_processor.process_documents([document], chunk=True)

            for i, chunk in enumerate(chunks):
                if len(chunk.text.strip()) < 100:
                    continue

                topic = chunk.metadata.get('source', 'History').replace('.pdf', '').replace('_', ' ')

                training_examples.append({
                    "messages": [
                        {
                            "role": "system",
                            "content": "You are a helpful and patient history tutor for high school students. Provide clear, accurate, and pedagogically sound explanations."
                        },
                        {
                            "role": "user",
                            "content": f"Can you explain about {topic}?"
                        },
                        {
                            "role": "assistant",
                            "content": chunk.text.strip()
                        }
                    ],
                    "topic": topic
                })

        except Exception as e:
            print(f"  Error processing {pdf_file.name}: {e}")
            continue

    return training_examples

def prepare_finetuning_data(
    input_file: str = "questions_dataset.json",
    pdf_directory: str = "data",
    output_dir: str = "data/finetuning",
    train_split: float = 0.9,
    include_pdfs: bool = True,
    seed: int = 42
) -> None:
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    print(f"Loaded {len(data)} Q&A pairs")

    random.seed(seed)
    random.shuffle(data)

    formatted_data = []

    for item in data:
        formatted_item = {
            "messages": [
                {
                    "role": "system",
                    "content": "You are a helpful and patient history tutor for high school students. Provide clear, accurate, and pedagogically sound explanations."
                },
                {
                    "role": "user",
                    "content": item["question"]
                },
                {
                    "role": "assistant",
                    "content": item["answer"]
                }
            ],
            "topic": item.get("topic", "General History")
        }
        formatted_data.append(formatted_item)

    if include_pdfs:
        print(f"\nExtracting content from PDFs in {pdf_directory}...")
        pdf_data = extract_pdf_training_data(pdf_directory)
        formatted_data.extend(pdf_data)
        print(f"Added {len(pdf_data)} training examples from PDFs")

    split_idx = int(len(formatted_data) * train_split)
    train_data = formatted_data[:split_idx]
    val_data = formatted_data[split_idx:]

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    train_file = output_path / "train.jsonl"
    val_file = output_path / "val.jsonl"

    with open(train_file, 'w', encoding='utf-8') as f:
        for item in train_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

    with open(val_file, 'w', encoding='utf-8') as f:
        for item in val_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

    print(f"Total: {len(formatted_data)}, Train: {len(train_data)}, Val: {len(val_data)}")
    print(f"Saved to: {train_file}, {val_file}")

if __name__ == "__main__":
    prepare_finetuning_data(include_pdfs=True)
