import torch
import json
import os
from torch.utils.data import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType
import logging

# Set environment variable to reduce memory fragmentation
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)


class JSONLDataset(Dataset):
    """Custom Dataset for loading JSONL files without the datasets library"""
    def __init__(self, file_path):
        self.data = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                self.data.append(json.loads(line.strip()))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class SimpleFinetuner:
    def __init__(
        self,
        model_name: str = "meta-llama/Llama-3.2-3B-Instruct",
        output_dir: str = "./models/finetuned",
    ):
        self.model_name = model_name
        self.output_dir = output_dir
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        print(f"Model: {model_name}")
        print(f"Device: {self.device}")

    def load_and_prepare_data(self, train_file: str, val_file: str):
        train_dataset = JSONLDataset(train_file)
        val_dataset = JSONLDataset(val_file)
        print(f"Loaded {len(train_dataset)} train, {len(val_dataset)} val samples")
        return {'train': train_dataset, 'validation': val_dataset}

    def setup_model_and_tokenizer(self):
        print(f"Loading model: {self.model_name}")

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                quantization_config=None,
                device_map="auto",
                max_memory={0: "14GB", "cpu": "30GB"},  # Reserve 14GB GPU, use CPU for overflow
                trust_remote_code=True,
                torch_dtype=torch.float16,
                use_cache=False,  # Disable KV cache to save memory during training
                offload_state_dict=True  # Offload to CPU during loading
            )

        # Enable gradient checkpointing before LoRA
        self.model.gradient_checkpointing_enable()

        # For fp16 models, prepare for training
        self.model.enable_input_require_grads()

        # LoRA configuration
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]

        lora_config = LoraConfig(
            r=8,
            lora_alpha=16,
            target_modules=target_modules,
            lora_dropout=0.05,
            bias="none",
            task_type=TaskType.CAUSAL_LM
        )

        self.model = get_peft_model(self.model, lora_config)

        # Print trainable parameters
        self.model.print_trainable_parameters()

        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"Trainable: {trainable_params:,} / {total_params:,} ({100 * trainable_params / total_params:.2f}%)")

    def formatting_func(self, example):
        messages = example['messages']
        text = ""
        for msg in messages:
            role = msg['role']
            content = msg['content']
            if role == "system":
                text += f"<|system|>\n{content}<|end|>\n"
            elif role == "user":
                text += f"<|user|>\n{content}<|end|>\n"
            elif role == "assistant":
                text += f"<|assistant|>\n{content}<|end|>\n"
        return text

    def tokenize_function(self, examples):
        texts = [self.formatting_func(ex) for ex in examples]
        result = self.tokenizer(
            texts,
            truncation=True,
            max_length=1024,
            padding="max_length",
            return_tensors=None
        )
        result["labels"] = result["input_ids"].copy()
        return result

    def train(self, dataset, num_epochs: int = 3, batch_size: int = 8, learning_rate: float = 2e-4):
        print("Tokenizing data...")

        # Tokenize training data
        tokenized_train_data = []
        for example in dataset['train']:
            tokenized = self.tokenize_function([example])
            tokenized_train_data.append({
                'input_ids': tokenized['input_ids'][0],
                'attention_mask': tokenized['attention_mask'][0],
                'labels': tokenized['labels'][0]
            })

        # Tokenize validation data
        tokenized_val_data = []
        for example in dataset['validation']:
            tokenized = self.tokenize_function([example])
            tokenized_val_data.append({
                'input_ids': tokenized['input_ids'][0],
                'attention_mask': tokenized['attention_mask'][0],
                'labels': tokenized['labels'][0]
            })

        # Create simple dataset wrapper
        class TokenizedDataset(Dataset):
            def __init__(self, data):
                self.data = data
            def __len__(self):
                return len(self.data)
            def __getitem__(self, idx):
                return self.data[idx]

        tokenized_train = TokenizedDataset(tokenized_train_data)
        tokenized_val = TokenizedDataset(tokenized_val_data)

        training_args = TrainingArguments(
            output_dir=self.output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            gradient_accumulation_steps=2,  # Effective batch of 8 (4 * 2)
            gradient_checkpointing=False,  # Already enabled on model
            gradient_checkpointing_kwargs={"use_reentrant": False},  # Better for LoRA
            learning_rate=learning_rate,
            fp16=True,
            logging_steps=10,
            eval_strategy="steps",
            eval_steps=100,  # Less frequent eval to save memory
            save_strategy="steps",
            save_steps=200,  # Less frequent saves
            save_total_limit=1,  # Only keep 1 checkpoint to save disk and memory
            load_best_model_at_end=False,
            metric_for_best_model="eval_loss",
            warmup_steps=50,
            logging_dir=f"{self.output_dir}/logs",
            report_to="none",
            remove_unused_columns=False,
            max_grad_norm=0.3,
            optim="adamw_torch",
            ddp_find_unused_parameters=False,
        )

        data_collator = DataCollatorForSeq2Seq(
            tokenizer=self.tokenizer,
            model=self.model,
            padding=True
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=tokenized_train,
            eval_dataset=tokenized_val,
            data_collator=data_collator,
        )

        print("Starting training...")
        trainer.train()

        final_dir = f"{self.output_dir}/final"
        trainer.save_model(final_dir)
        self.tokenizer.save_pretrained(final_dir)

        print(f"Training complete. Model saved to: {final_dir}")


def main():
    # Clear GPU cache before starting
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    CONFIG = {
        "model_name": "meta-llama/Llama-3.2-3B-Instruct",  # 3B model fits comfortably in 16GB
        "train_file": "data/finetuning/train.jsonl",
        "val_file": "data/finetuning/val.jsonl",
        "output_dir": "models/finetuned",
        "num_epochs": 3,
        "batch_size": 4,  # Can use larger batch with 3B model
        "learning_rate": 2e-4
    }

    print("Fine-tuning Large Tutoring Model")
    print("-" * 50)

    finetuner = SimpleFinetuner(
        model_name=CONFIG["model_name"],
        output_dir=CONFIG["output_dir"],
    )

    dataset = finetuner.load_and_prepare_data(
        train_file=CONFIG["train_file"],
        val_file=CONFIG["val_file"]
    )

    finetuner.setup_model_and_tokenizer()

    finetuner.train(
        dataset=dataset,
        num_epochs=CONFIG["num_epochs"],
        batch_size=CONFIG["batch_size"],
        learning_rate=CONFIG["learning_rate"]
    )

    print("Done!")


if __name__ == "__main__":
    main()
