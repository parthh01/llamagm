import os
import argparse
from huggingface_hub import HfApi, login
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig
import torch
from dotenv import load_dotenv

load_dotenv()

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained model directory")
    parser.add_argument("--repo_name", default="chess-llm-tournament", type=str, required=True, help="Name for the HF repository")
    parser.add_argument("--hf_token", type=str, help="Hugging Face token")
    parser.add_argument("--private", action="store_true",default=False, help="Whether to make the repository private")
    parser.add_argument("--commit_message", type=str, default="Upload chess GRPO trained model", help="Commit message for the upload")
    return parser.parse_args()

def upload_chess_model_to_hf(
    model_path: str,
    repo_name: str,
    hf_token: str = None,
    private: bool = False,
    commit_message: str = "Upload chess GRPO trained model"
):
    """
    Upload a trained chess model to Hugging Face Hub.
    
    Args:
        model_path: Path to the trained model directory
        repo_name: Name for the HF repository (e.g., "username/chess-model")
        hf_token: Hugging Face token (if not provided, will try from env)
        private: Whether to make the repository private
        commit_message: Commit message for the upload
    """
    
    # Get HF token
    if hf_token is None:
        hf_token = os.getenv('HUGGINGFACE_TOKEN')
    
    if hf_token is None:
        raise ValueError("Hugging Face token not provided. Set HUGGINGFACE_TOKEN environment variable or pass hf_token parameter.")
    
    # Login to Hugging Face
    login(token=hf_token)
    
    print(f"Uploading model from {model_path} to {repo_name}")
    
    try:
        # Try to load as PEFT model first
        peft_config = PeftConfig.from_pretrained(model_path)
        print(f"Detected PEFT model with base: {peft_config.base_model_name_or_path}")
        
        # Load the base model
        base_model = AutoModelForCausalLM.from_pretrained(
            peft_config.base_model_name_or_path,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        
        # Load the PEFT model
        model = PeftModel.from_pretrained(base_model, model_path)
        
        # Merge the LoRA weights into the base model for easier deployment
        print("Merging LoRA weights into base model...")
        model = model.merge_and_unload()
        
        # Load tokenizer from base model
        tokenizer = AutoTokenizer.from_pretrained(peft_config.base_model_name_or_path)
        
        upload_type = "PEFT (merged)"
        
    except Exception as e:
        print(f"Failed to load as PEFT model: {e}")
        print("Loading as regular model...")
        
        # Load as regular model
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        upload_type = "Regular model"
    
    # Ensure pad token is set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print(f"Model type: {upload_type}")
    print(f"Model parameters: {model.num_parameters():,}")
    
    # Create repository if it doesn't exist
    api = HfApi()
    try:
        api.create_repo(repo_id=repo_name, private=private, exist_ok=True)
        print(f"Repository {repo_name} created/verified")
    except Exception as e:
        print(f"Repository creation warning: {e}")
    
    # Upload model and tokenizer
    print("Uploading model...")
    model.push_to_hub(
        repo_name,
        commit_message=commit_message,
        private=private
    )
    
    print("Uploading tokenizer...")
    tokenizer.push_to_hub(
        repo_name,
        commit_message=commit_message,
        private=private
    )
    
    # Create and upload model card
    model_card_content = create_model_card(model_path, upload_type)
    
    with open("README.md", "w") as f:
        f.write(model_card_content)
    
    api.upload_file(
        path_or_fileobj="README.md",
        path_in_repo="README.md",
        repo_id=repo_name,
        commit_message="Add model card"
    )
    
    # Clean up temporary file
    os.remove("README.md")
    
    print(f"✅ Model successfully uploaded to https://huggingface.co/{repo_name}")
    
    return f"https://huggingface.co/{repo_name}"

def create_model_card(model_path: str, model_type: str) -> str:
    """Create a model card for the chess model"""
    
    return f"""---
license: apache-2.0
language:
- en
tags:
- chess
- reinforcement-learning
- grpo
- game-playing
pipeline_tag: text-generation
---

# Chess GRPO Trained Model

This model has been trained using Group Relative Policy Optimization (GRPO) to play chess. It was trained to generate chess moves in JSON format with reasoning.

## Model Details

- **Model Type**: {model_type}
- **Training Method**: GRPO (Group Relative Policy Optimization)
- **Task**: Chess move generation with evaluation reasoning
- **Source Path**: {model_path}


 """

if __name__ == "__main__":
    args = parse_args()

    upload_chess_model_to_hf(args.model_path, args.repo_name, args.hf_token, args.private, args.commit_message)