import torch
import argparse
import os
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, BitsAndBytesConfig, EarlyStoppingCallback
from peft import PeftModel, PeftConfig, LoraConfig, get_peft_model
from datasets import load_dataset
from trl import SFTTrainer,SFTConfig
from dotenv import load_dotenv
import sys 

# Set tokenizer parallelism to avoid fork warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Add the project root directory to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from sqlalchemy import create_engine
from datagen.loader import create_dataset


def parse_args():
    parser = argparse.ArgumentParser(description="Train a language model with LoRA fine-tuning")
    
    # Database and dataset arguments
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for dataset creation")
    parser.add_argument("--push_to_hub", action="store_true", help="Whether to push dataset to Hugging Face Hub")
    parser.add_argument("--hub_name", type=str, default=None, help="Hub name for dataset if pushing to Hub")
    
    # Model arguments
    parser.add_argument("--model_name", type=str, default="openlm-research/open_llama_7b", help="Base model to fine-tune")
    parser.add_argument("--load_in_8bit", action="store_true", default=False, help="Whether to load model in 8-bit quantization")
    
    # LoRA arguments
    parser.add_argument("--lora_r", type=int, default=32, help="LoRA attention dimension")
    parser.add_argument("--lora_alpha", type=int, default=64, help="LoRA alpha parameter")
    parser.add_argument("--lora_dropout", type=float, default=0.1, help="LoRA dropout probability")
    parser.add_argument("--target_modules", type=str, nargs="+", 
                        default=["q_proj", "k_proj", "v_proj", "o_proj","gate_proj","down_proj","up_proj"], 
                        help="Target modules for LoRA")
    
    # Training arguments
    parser.add_argument("--learning_rate", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--num_train_epochs", type=int, default=6, help="Number of training epochs")
    parser.add_argument("--per_device_train_batch_size", type=int, default=8, help="Per-device training batch size")
    parser.add_argument("--save_steps", type=int, default=500, help="Save checkpoint every X steps")
    parser.add_argument("--logging_steps", type=int, default=100, help="Log every X steps")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory for model checkpoints")
    
    # Dataset split arguments
    parser.add_argument("--eval_split_ratio", type=float, default=0.05, help="Ratio of dataset to use for evaluation (0.1 = 10%)")
    parser.add_argument("--random_seed", type=int, default=42, help="Random seed for dataset splitting")
    
    # Early stopping arguments
    parser.add_argument("--early_stopping_patience", type=int, default=3, help="Number of evaluations with no improvement after which training will be stopped")
    parser.add_argument("--early_stopping_threshold", type=float, default=0.001, help="Minimum change in loss to qualify as an improvement")
    
    return parser.parse_args()

def load_model_with_peft_support(args):
    """Load model with PEFT support, similar to learn.py logic"""
    
    # Load tokenizer first
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Configure quantization
    quantization_config = None
    num_gpus = torch.cuda.device_count()
    if args.load_in_8bit and num_gpus == 1:
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_threshold=6.0
        )
    elif args.load_in_8bit and num_gpus > 1:
        print("Warning: Disabling 8-bit quantization for multi-GPU training")
    
    try:
        # Try to load as existing PEFT model first
        peft_config = PeftConfig.from_pretrained(args.model_name)
        
        # Load base model with quantization
        base_model = AutoModelForCausalLM.from_pretrained(
            peft_config.base_model_name_or_path,
            quantization_config=quantization_config,
            torch_dtype=torch.float16,
            device_map=None if num_gpus > 1 else "auto",
            trust_remote_code=True
        )
        
        # Load the PEFT model on top
        model = PeftModel.from_pretrained(base_model, args.model_name, is_trainable=True)
        
        print(f"Successfully loaded existing PEFT model from {args.model_name}")
        print(f"Base model: {peft_config.base_model_name_or_path}")
        
    except Exception as e:
        print(f"Failed to load as existing PEFT model: {e}")
        print(f"Loading as base model and applying LoRA...")
        
        # Load base model
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            quantization_config=quantization_config,
            torch_dtype=torch.float16,
            device_map=None if num_gpus > 1 else "auto",
            trust_remote_code=True
        )
        
        # Apply LoRA configuration
        lora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            target_modules=args.target_modules,
            lora_dropout=args.lora_dropout,
            task_type="CAUSAL_LM"
        )
        model = get_peft_model(model, lora_config)
        
        print(f"Applied LoRA to base model {args.model_name}")
    
    return model, tokenizer

def main():
    
    args = parse_args()
    
    load_dotenv()
    
    # Check if multiple GPUs are available
    num_gpus = torch.cuda.device_count()
    print(f"Number of GPUs available: {num_gpus}")
    
    DATABASE_URL = f"postgresql://{os.getenv('PGUSER')}:{os.getenv('PGPASSWORD')}@{os.getenv('PGHOST')}:{os.getenv('PGPORT')}/{os.getenv('PGDATABASE')}?sslmode=require"
    engine = create_engine(DATABASE_URL)
    
    # Get dataset and split into train/eval
    full_dataset = load_dataset("parthh01/llamagm-bongcloud", split="train")
    
    # Split dataset into train and eval
    dataset_split = full_dataset.train_test_split(
        test_size=args.eval_split_ratio, 
        seed=args.random_seed,
        shuffle=True
    )
    training_dataset = dataset_split["train"]
    eval_dataset = dataset_split["test"]
    
    print(f"Training samples: {len(training_dataset)}")
    print(f"Evaluation samples: {len(eval_dataset)}")
    
    # Load model and tokenizer with PEFT support
    model, tokenizer = load_model_with_peft_support(args)
    
    # Set output directory if not specified
    if args.output_dir is None:
        args.output_dir = f"./{args.model_name.replace('/', '-')}-lora"
    
    # Set up training arguments for multi-GPU
    training_args = SFTConfig(
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_train_batch_size,  # Same batch size for eval
        save_steps=args.save_steps,
        logging_steps=args.logging_steps,
        report_to="wandb",
        gradient_accumulation_steps=1,
        warmup_steps=100,
        save_total_limit=3,
        logging_first_step=True,
        remove_unused_columns=False,
        completion_only_loss=True,
        # Early stopping configuration
        eval_strategy="steps",
        eval_steps=args.logging_steps,  # Evaluate at the same frequency as logging
        metric_for_best_model="eval_loss",  # Use eval_loss instead of train_loss
        greater_is_better=False,  # Lower loss is better
        load_best_model_at_end=True,
    )
    
    # Create early stopping callback
    early_stopping_callback = EarlyStoppingCallback(
        early_stopping_patience=args.early_stopping_patience,
        early_stopping_threshold=args.early_stopping_threshold
    )
    
    # Create trainer and train
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=training_dataset,
        eval_dataset=eval_dataset,  # Add eval dataset
        callbacks=[early_stopping_callback],
    )
    
    trainer.train()
    
    trainer.save_model(f"{args.output_dir}-final")

if __name__ == "__main__":
    main()
