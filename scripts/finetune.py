import os
import json
import torch
from pathlib import Path
from torch.utils.data import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
)
from peft import LoraConfig, get_peft_model, TaskType


# Ensure HF cache path exists
os.environ.setdefault("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


def get_local_model_path(model_name: str) -> str:
    hf_home = Path(os.environ["HF_HOME"])
    cache_path = hf_home / "hub" / f"models--{model_name.replace('/', '--')}"
    snapshots_dir = cache_path / "snapshots"

    if not snapshots_dir.exists():
        raise FileNotFoundError(f"Model cache not found at {cache_path}")

    snapshot_dirs = list(snapshots_dir.iterdir())
    if not snapshot_dirs:
        raise FileNotFoundError(f"No snapshot found in {snapshots_dir}")

    return str(snapshot_dirs[0])


class JSONLDataset(Dataset):
    def __init__(self, file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            self.data = [json.loads(line.strip()) for line in f]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class SimpleFinetuner:
    def __init__(self, model_name: str, output_dir: str):
        self.model_name = model_name
        self.output_dir = output_dir
        print(f"Model: {model_name}")

    def load_and_prepare_data(self, train_file: str, val_file: str):
        train_dataset = JSONLDataset(train_file)
        val_dataset = JSONLDataset(val_file)
        print(f"Loaded {len(train_dataset)} train, {len(val_dataset)} val samples")
        return {"train": train_dataset, "validation": val_dataset}

    def setup_model_and_tokenizer(self):
        local_model_path = get_local_model_path(self.model_name)
        print(f"Loading model from: {local_model_path}")

        self.tokenizer = AutoTokenizer.from_pretrained(
            local_model_path,
            trust_remote_code=True,
            local_files_only=True,
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            local_model_path,
            dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
            use_cache=False,
            local_files_only=True,
        )

        self.model.gradient_checkpointing_enable()
        self.model.enable_input_require_grads()

        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )

        self.model = get_peft_model(self.model, lora_config)
        self.model.print_trainable_parameters()

    def format_example(self, example):
        text = ""
        for msg in example["messages"]:
            role = msg["role"]
            content = msg["content"]
            text += f"<|{role}|>\n{content}<|end|>\n"
        return text

    def tokenize_dataset(self, dataset):
        tokenized = []
        for example in dataset:
            text = self.format_example(example)
            enc = self.tokenizer(
                text,
                truncation=True,
                max_length=1024,
                padding="max_length",
            )
            enc["labels"] = enc["input_ids"].copy()
            tokenized.append(enc)
        return tokenized

    def train(self, dataset, num_epochs=3, batch_size=2, learning_rate=2e-4):
        print("Tokenizing data...")
        train_data = self.tokenize_dataset(dataset["train"])
        val_data = self.tokenize_dataset(dataset["validation"])

        class TokenizedDataset(Dataset):
            def __init__(self, data):
                self.data = data

            def __len__(self):
                return len(self.data)

            def __getitem__(self, idx):
                return self.data[idx]

        training_args = TrainingArguments(
            output_dir=self.output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            gradient_accumulation_steps=4,
            learning_rate=learning_rate,
            bf16=True,
            logging_steps=10,
            eval_strategy="steps",
            eval_steps=100,
            save_strategy="steps",
            save_steps=200,
            save_total_limit=2,
            warmup_steps=50,
            logging_dir=f"{self.output_dir}/logs",
            report_to="none",
            remove_unused_columns=False,
            max_grad_norm=0.3,
            optim="adamw_torch",
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=TokenizedDataset(train_data),
            eval_dataset=TokenizedDataset(val_data),
            data_collator=DataCollatorForSeq2Seq(
                tokenizer=self.tokenizer,
                model=self.model,
                padding=True,
            ),
        )

        print("Starting training...")
        trainer.train()

        final_dir = f"{self.output_dir}/final"
        trainer.save_model(final_dir)
        self.tokenizer.save_pretrained(final_dir)
        print(f"Training complete. Model saved to: {final_dir}")


def main():
    CONFIG = {
        "model_name": "meta-llama/Meta-Llama-3.1-8B-Instruct",
        "train_file": "data/finetuning/train.jsonl",
        "val_file": "data/finetuning/val.jsonl",
        "output_dir": "models/finetuned_8b",
        "num_epochs": 3,
        "batch_size": 2,
        "learning_rate": 2e-4,
    }

    print("Fine-tuning Llama 3.1 8B")
    print("-" * 40)
    for k, v in CONFIG.items():
        print(f"{k}: {v}")
    print("-" * 40)

    finetuner = SimpleFinetuner(CONFIG["model_name"], CONFIG["output_dir"])
    dataset = finetuner.load_and_prepare_data(CONFIG["train_file"], CONFIG["val_file"])
    finetuner.setup_model_and_tokenizer()
    finetuner.train(
        dataset,
        CONFIG["num_epochs"],
        CONFIG["batch_size"],
        CONFIG["learning_rate"],
    )

    print("Done!")


if __name__ == "__main__":
    main()