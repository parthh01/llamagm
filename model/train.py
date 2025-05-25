import torch
import argparse
import os
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, BitsAndBytesConfig
from datasets import load_dataset
from trl import SFTTrainer,SFTConfig
from peft import LoraConfig, get_peft_model
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
    parser.add_argument("--num_train_epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--per_device_train_batch_size", type=int, default=8, help="Per-device training batch size")
    parser.add_argument("--save_steps", type=int, default=500, help="Save checkpoint every X steps")
    parser.add_argument("--logging_steps", type=int, default=100, help="Log every X steps")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory for model checkpoints")
    
    return parser.parse_args()

def main():
    
    args = parse_args()
    
    load_dotenv()
    
    # Check if multiple GPUs are available
    num_gpus = torch.cuda.device_count()
    print(f"Number of GPUs available: {num_gpus}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    DATABASE_URL = f"postgresql://{os.getenv('PGUSER')}:{os.getenv('PGPASSWORD')}@{os.getenv('PGHOST')}:{os.getenv('PGPORT')}/{os.getenv('PGDATABASE')}?sslmode=require"
    engine = create_engine(DATABASE_URL)
    
    # Get dataset and total rows
    training_dataset, total_rows = create_dataset(
        database_url=DATABASE_URL,
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        push_to_hub=args.push_to_hub,
        hub_name=args.hub_name,
        prompt_completion=True  # This creates combined text format
    )
    
    # Calculate max_steps based on total rows and number of GPUs
    max_steps = (total_rows // (args.per_device_train_batch_size * num_gpus)) * args.num_train_epochs
    print(f"Training for {max_steps} steps based on {total_rows} examples across {num_gpus} GPUs")
    
    # Configure quantization - disable for multi-GPU
    quantization_config = None
    if args.load_in_8bit and num_gpus == 1:
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_threshold=6.0
        )
    elif args.load_in_8bit and num_gpus > 1:
        print("Warning: Disabling 8-bit quantization for multi-GPU training")
    
    # Load model without device_map for multi-GPU
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        quantization_config=quantization_config,
        torch_dtype=torch.float16,  # Use fp16 for better multi-GPU performance
        device_map=None,  # Don't use device_map for multi-GPU
        trust_remote_code=True,
        #attn_implementation="flash_attention_2"
    )
    
    # Apply LoRA for efficient fine-tuning
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=args.target_modules,
        lora_dropout=args.lora_dropout,
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_config)
    
    # Set output directory if not specified
    if args.output_dir is None:
        args.output_dir = f"./{args.model_name.replace('/', '-')}-lora"
    
    # Set up training arguments for multi-GPU
    training_args = SFTConfig(
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        save_steps=args.save_steps,
        logging_steps=args.logging_steps,
        max_steps=max_steps,
        report_to="wandb",
        dataloader_num_workers=0,  # Set to 0 to avoid IterableDataset issues
        gradient_accumulation_steps=1,
        warmup_steps=100,
        save_total_limit=3,
        logging_first_step=True,
        remove_unused_columns=False,
        completion_only_loss=True,
        # Multi-GPU specific settings
        dataloader_pin_memory=False,  # Disable for IterableDataset
        save_safetensors=True,
        bf16=torch.cuda.is_bf16_supported(),  # Use bf16 if supported, otherwise fp16
        fp16=not torch.cuda.is_bf16_supported(),
        ddp_find_unused_parameters=False,
    )
    
    # Create trainer and train
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=training_dataset,
    )
    
    trainer.train()
    
    trainer.save_model(f"{args.output_dir}-final")

if __name__ == "__main__":
    main()
