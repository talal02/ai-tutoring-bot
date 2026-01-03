import json
import random
from pathlib import Path

def prepare_finetuning_data(
    input_file: str = "questions_dataset.json",
    output_dir: str = "data/finetuning",
    train_split: float = 0.9,
    seed: int = 42
) -> None:
    """Convert Q&A dataset to training format for LoRA fine-tuning."""

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
    prepare_finetuning_data()
