import torch
import json
import chess
import chess.engine
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel, PeftConfig, LoraConfig, get_peft_model
from trl import GRPOConfig, GRPOTrainer
from datasets import Dataset
import os
from dotenv import load_dotenv
from stockfish import Stockfish
import threading
import re

import sys 
# Set tokenizer parallelism to avoid fork warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Add the project root directory to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from datagen.gen import generate_grpo_games
import argparse
load_dotenv()

@dataclass
class ChessReward:
    """Reward structure for chess moves"""
    json_parse_penalty: float = -10.0  # Large negative for invalid JSON
    illegal_move_reward: float = 0.0   # Neutral for illegal moves
    legal_move_reward: float = 1.0    # Weak positive for legal moves
    reasoning_parse_reward: float = 1.0  # Weak positive for parseable reasoning
    value_accuracy_reward: float = 1.0   # Medium positive for accurate evaluation
    position_improvement_reward: float = 5.0  # Large positive for position improvement
    win_reward: float = 200.0  # Large positive for wins
    draw_reward: float = 100.0  
    loss_reward: float = -200.0  # Large negative for losses

class ChessGRPOEnvironment:
    """Environment for GRPO chess training"""
    
    def __init__(self, stockfish_path: str = None, skill_level: int = 3):
        self.stockfish_path = stockfish_path or os.getenv('STOCKFISH_PATH', '/usr/local/bin/stockfish')
        self.skill_level = skill_level
        self.reward_config = ChessReward()
        self.thread_local = threading.local()
        
    def get_stockfish(self):
        """Get thread-local Stockfish instance"""
        if not hasattr(self.thread_local, "stockfish"):
            stockfish = Stockfish(
                path=self.stockfish_path,
                depth=10,
                parameters={"Threads": 1, "Hash": 32}
            )
            stockfish.set_skill_level(self.skill_level)
            self.thread_local.stockfish = stockfish
        return self.thread_local.stockfish
    
    def parse_model_output(self, output: str) -> Tuple[Optional[str], Optional[str], bool]:
        """
        Parse model output to extract move and reasoning.
        Returns: (move, reasoning, is_valid_json)
        """
        try:
            parsed = json.loads(output)
            move = parsed.get("move")
            reasoning = parsed.get("reasoning", "")
            return move, reasoning, True
        except (json.JSONDecodeError, KeyError, IndexError):
            print('json decode error')
            print(output)
            return None, None, False
    
    def extract_value_from_reasoning(self, reasoning: str) -> Optional[float]:
        """
        Extract numerical value estimation from reasoning text.
        Returns value in centipawns or None if not found.
        Format: "eval: {value}" or "M{moves}" where positive is white advantage/mate, negative is black
        """
        if not reasoning:
            return None
        
        reasoning = reasoning.strip()
        
        # Handle mate patterns: M{moves} (e.g., "M5" for mate in 5, "M-3" for mate in -3)
        mate_pattern = r'^M([+-]?\d+)$'
        mate_match = re.match(mate_pattern, reasoning)
        if mate_match:
            mate_moves = int(mate_match.group(1))
            # Convert mate to large centipawn equivalent
            # Positive mate moves = white mate, negative = black mate
            return 9999 if mate_moves > 0 else -9999
        
        # Handle eval patterns: eval: {value} (e.g., "eval: 150", "eval: -200")
        eval_pattern = r'^eval:\s*([+-]?\d+)$'
        eval_match = re.match(eval_pattern, reasoning)
        if eval_match:
            try:
                return float(eval_match.group(1))
            except ValueError:
                pass
        
        return None
    
    def get_stockfish_evaluation(self, board: chess.Board) -> float:
        """Get Stockfish evaluation of position in centipawns"""
        stockfish = self.get_stockfish()
        stockfish.set_fen_position(board.fen())
        evaluation = stockfish.get_evaluation()
        
        if evaluation['type'] == 'cp':
            return evaluation['value']
        elif evaluation['type'] == 'mate':
            # Convert mate to large centipawn equivalent
            mate_moves = evaluation['value']
            return 9999 if mate_moves > 0 else -9999
        return 0.0
    
    def is_position_improving(self, board_before: chess.Board, board_after: chess.Board, 
                            is_white_move: bool) -> bool:
        """Check if the move improved the position for the moving player"""
        eval_before = self.get_stockfish_evaluation(board_before)
        eval_after = self.get_stockfish_evaluation(board_after)
        
        if is_white_move:
            return eval_after > eval_before
        else:
            return eval_after < eval_before
    
    def calculate_reward(self, board: chess.Board, model_output: str, prompt: str = None) -> Tuple[float, Dict]:
        """
        Calculate reward for a model output given the board state.
        Returns: (reward, info_dict)
        """
        info = {}
        reward = 0.0
        
        # Check if this is one of the first 2 white moves (opening moves we don't want to modify)
        move_history = [] 
        # Parse move history from prompt to determine LLM move count
        if prompt:
            try:
                # Extract JSON from prompt - find the last occurrence of {"moveHistory"
                json_start = prompt.rfind('{"moveHistory"')
                if json_start != -1:
                    # Find the first } after the json_start to close the JSON object
                    json_end = prompt.find('}', json_start) + 1
                    if json_end > json_start:
                        json_str = prompt[json_start:json_end]
                        position_data = json.loads(json_str)
                        move_history = position_data.get("moveHistory", [])
                        info['move_history_length'] = len(move_history)
            except (json.JSONDecodeError, KeyError, ValueError) as e:
                info['prompt_parse_error'] = str(e)
                # Fallback to using board state
        is_white_move = board.turn
        llm_move_count = len(move_history)
        # Skip rewarding first 2 white moves (LLM moves 0 and 1 when playing white)
        if is_white_move and llm_move_count <= 3:
            info['opening_move_skipped'] = True
            info['move_number'] = llm_move_count
            return 0.0, info
        
        # 1. Parse model output - large negative penalty for invalid JSON
        move_str, reasoning, is_valid_json = self.parse_model_output(model_output)
        info['valid_json'] = is_valid_json
        info['move_str'] = move_str
        info['reasoning'] = reasoning
        
        if not is_valid_json:
            reward += self.reward_config.json_parse_penalty
            info['json_parse_penalty'] = self.reward_config.json_parse_penalty
            return reward, info
        
        # 2. Check if move is legal - neutral reward for illegal moves
        legal_moves = list(board.legal_moves)
        legal_moves_san = [board.san(move) for move in legal_moves]
        
        if move_str not in legal_moves_san:
            reward += self.reward_config.illegal_move_reward  # 0.0
            info['illegal_move'] = True
            return reward, info
        
        # Parse the move
        try:
            move = board.parse_san(move_str)
            info['move'] = move
        except ValueError:
            reward += self.reward_config.illegal_move_reward  # 0.0
            info['move_parse_error'] = True
            return reward, info
        
        # 3. Legal move - weak positive reward
        reward += self.reward_config.legal_move_reward
        info['legal_move_reward'] = self.reward_config.legal_move_reward
        
        # 4. Check if reasoning can be parsed - weak positive reward
        estimated_eval = self.extract_value_from_reasoning(reasoning)
        if estimated_eval is not None:
            reward += self.reward_config.reasoning_parse_reward
            info['reasoning_parse_reward'] = self.reward_config.reasoning_parse_reward
            info['estimated_eval'] = estimated_eval
            
            # 5. Check accuracy of evaluation - medium positive reward
            board_copy = board.copy()
            board_copy.push(move)
            actual_eval = self.get_stockfish_evaluation(board_copy)
            info['actual_eval'] = actual_eval
            
            # Calculate accuracy reward based on how close the estimate is
            eval_error = abs(estimated_eval - actual_eval)
            # Use exponential decay for accuracy reward - closer estimates get higher rewards
            max_error = 500  # centipawns - beyond this, no accuracy reward
            if eval_error < max_error:
                accuracy_multiplier = float(np.exp(-eval_error / 200))  # Decay factor
                accuracy_reward = self.reward_config.value_accuracy_reward * accuracy_multiplier
                reward += accuracy_reward
                info['accuracy_reward'] = accuracy_reward
                info['eval_error'] = eval_error
        
        # 6. Position improvement - large positive reward
        eval_before = self.get_stockfish_evaluation(board)
        board_copy = board.copy()
        board_copy.push(move)
        eval_after = self.get_stockfish_evaluation(board_copy)
        info['eval_after'] = eval_after
        info['eval_before'] = eval_before
        
        # Calculate position improvement
        eval_diff = eval_after - eval_before
        if is_white_move:
            # For white, positive eval_diff is good (position improved)
            position_improvement = eval_diff
        else:
            # For black, negative eval_diff is good (position improved from black's perspective)
            position_improvement = -eval_diff
        
        # Scale the improvement reward
        if position_improvement > 0:
            # Positive improvement gets reward
            improvement_reward = min(position_improvement / 100, 5.0) * self.reward_config.position_improvement_reward
            reward += improvement_reward
            info['improvement_reward'] = improvement_reward
        else:
            # Negative improvement gets penalty (but not as harsh as illegal moves)
            improvement_penalty = max(position_improvement / 100, 0) * self.reward_config.position_improvement_reward
            reward += improvement_penalty
            info['improvement_penalty'] = improvement_penalty
        
        info['position_improvement'] = position_improvement
        info['eval_diff'] = eval_diff
        
        return reward, info
        
    
    def get_game_outcome_reward(self, board: chess.Board, max_moves_reached: bool = False) -> float:
        """Calculate final game outcome reward"""
        if board.is_checkmate():
            # Winner gets positive reward, loser gets negative
            if board.turn:  # Black wins (white is in checkmate)
                return self.reward_config.loss_reward  # From white's perspective
            else:  # White wins (black is in checkmate)
                return self.reward_config.win_reward
        elif board.is_stalemate() or board.is_insufficient_material() or board.is_seventyfive_moves():
            return self.reward_config.draw_reward
        elif max_moves_reached:
            # Determine winner by Stockfish evaluation
            final_eval = self.get_stockfish_evaluation(board)
            if abs(final_eval) < 2:  # Close game, call it a draw
                return self.reward_config.draw_reward
            elif final_eval > 0:  # White advantage
                return self.reward_config.win_reward
            else:  # Black advantage
                return self.reward_config.loss_reward
        
        return 0.0  # Game not finished

class ChessGRPOTrainer:
    """GRPO trainer for chess"""
    
    def __init__(self, model_name: str, output_dir: str, stockfish_skill_level: int = 3, load_in_8bit: bool = False,per_device_train_batch_size: int = 8):
        self.model_name = model_name
        self.output_dir = output_dir
        self.stockfish_skill_level = stockfish_skill_level
        self.per_device_train_batch_size = per_device_train_batch_size
        # Load model and tokenizer with PEFT support
        self.model, self.tokenizer = self._load_peft_model(model_name, load_in_8bit) #TODO: reuse function in train.py
        
        # Initialize environment
        self.env = ChessGRPOEnvironment(skill_level=stockfish_skill_level)
        
        # GRPO configuration with memory optimizations
        self.grpo_config = GRPOConfig(
            output_dir=output_dir,
            learning_rate=1e-5,
            logging_steps=10,
            fp16=True,
            # Critical memory optimizations for GRPO
            per_device_train_batch_size=self.per_device_train_batch_size,  # Start with 1, can increase if stable
            gradient_accumulation_steps=4,  # Accumulate to effective batch size
            gradient_checkpointing=True,
            report_to="wandb",
            save_total_limit=3,
            num_train_epochs=3,
            completion_only_loss=True
            # GRPO specific memory settings
        )
    
    def _load_peft_model(self, model_path: str, load_in_8bit: bool = False):
        """Load model with aggressive memory optimizations"""
        
        # Load tokenizer first
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Clear any existing CUDA cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        try:
            # Try to load as existing PEFT model first
            peft_config = PeftConfig.from_pretrained(model_path)
            
            # Configure quantization like in train.py
            quantization_config = None
            if load_in_8bit:
                quantization_config = BitsAndBytesConfig(
                    load_in_8bit=True,
                    llm_int8_threshold=6.0
                )
            
            # Load base model with quantization
            base_model = AutoModelForCausalLM.from_pretrained(
                peft_config.base_model_name_or_path,
                quantization_config=quantization_config,
                device_map="auto",
                trust_remote_code=True
            )
            
            # Load the PEFT model on top
            model = PeftModel.from_pretrained(base_model, model_path,is_trainable=True)
            
            print(f"Successfully loaded existing PEFT model from {model_path}")
            print(f"Base model: {peft_config.base_model_name_or_path}")
            
        except Exception as e:
            print(f"Failed to load as existing PEFT model: {e}")
            print(f"Loading as base model and applying LoRA...")
            
            # Configure quantization like in train.py
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_threshold=6.0
            )
            
            # Load base model with quantization
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                quantization_config=quantization_config,
                device_map="auto",
                trust_remote_code=True
            )
            
            # Apply LoRA configuration like in train.py
            lora_config = LoraConfig(
                r=16,
                lora_alpha=32,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                lora_dropout=0.05,
                task_type="CAUSAL_LM",
                inference_mode=False
            )
            model = get_peft_model(model, lora_config)
            
            print(f"Applied LoRA to base model {model_path}")
        
        # Enable gradient checkpointing
        if hasattr(model, 'gradient_checkpointing_enable'):
            model.gradient_checkpointing_enable()
        elif hasattr(model.base_model, 'gradient_checkpointing_enable'):
            model.base_model.gradient_checkpointing_enable()
        
        # Print memory usage for debugging
        if torch.cuda.is_available():
            print(f"GPU memory allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
            print(f"GPU memory reserved: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")
            
        return model, tokenizer
    
    def generate_game_data(self, num_games: int = 100) -> Dataset:
        """Generate training data by playing games"""
        
        # Generate games using the current model
        game_data = generate_grpo_games(
            model=self.model,
            tokenizer=self.tokenizer,
            env=self.env,
            num_games=num_games,
            max_moves_per_game=50
        )
        
        return Dataset.from_list(game_data)
    
    def chess_reward_function(self, completions: List[str], **kwargs) -> List[float]:
        """
        Reward function for GRPO training.
        Takes completions and returns rewards for each.
        """
        rewards = []
        # Get the current batch of board states from kwargs
        board_states = kwargs.get('board_state', [])
        
        if len(board_states) != len(completions): raise ValueError("Number of board states and completions must match")
        
        for completion, fen_string, prompt in zip(completions, board_states,kwargs.get('prompts')):
            try:
                # Construct chess board from FEN string
                board_state = chess.Board(fen_string)
                # Calculate reward using our existing environment
                reward, _ = self.env.calculate_reward(board_state, completion, prompt)
                rewards.append(reward)
            except Exception as e:
                # Fallback reward for any errors
                print(f"Error calculating reward: {e}")
                rewards.append(-10.0)  # Large negative reward for errors
        print('rewards: ', rewards) 
        return rewards
    
    def train(self, num_iterations: int = 10, games_per_iteration: int = 100):
        """Main training loop with memory management"""
        
        # Clear GPU cache before starting
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        for iteration in range(num_iterations):
            print(f"GRPO Iteration {iteration + 1}/{num_iterations}")
            
            # Generate new training data
            print("Generating game data...")
            dataset = self.generate_game_data(games_per_iteration)
            
            # Force garbage collection and clear cache
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Create GRPO trainer with memory-optimized settings
            trainer = GRPOTrainer(
                model=self.model,
                args=self.grpo_config,
                train_dataset=dataset,
                reward_funcs=self.chess_reward_function,
            )
            
            # Train for one iteration
            print("Training...")
            trainer.train()
            
            # Save checkpoint
            checkpoint_dir = f"{self.output_dir}/checkpoint-{iteration}"
            trainer.save_model(checkpoint_dir)
            
            # Optionally increase Stockfish difficulty
            if iteration % 3 == 2:  # Every 3 iterations
                self.env.skill_level = min(20, self.env.skill_level + 2)
                print(f"Increased Stockfish skill level to {self.env.skill_level}")

        print("Training completed!")
        trainer.save_model(f"{self.output_dir}-final")


def parse_args():
    parser = argparse.ArgumentParser(description="Run GRPO training for chess LLM")
    
    parser.add_argument("--model_path", type=str, required=True,
                       help="Path to the base model to train")
    parser.add_argument("--output_dir", type=str, default="./chess-grpo-output",
                       help="Output directory for trained models")
    parser.add_argument("--num_iterations", type=int, default=10,
                       help="Number of GRPO iterations")
    parser.add_argument("--games_per_iteration", type=int, default=50,
                       help="Number of games to generate per iteration")
    parser.add_argument("--initial_skill_level", type=int, default=3,
                       help="Initial Stockfish skill level")
    parser.add_argument("--progressive", action="store_true",
                       help="Use progressive training against increasing Stockfish levels")
    parser.add_argument("--load_in_8bit", action="store_true", default=False,
                       help="Whether to load model in 8-bit quantization")
    parser.add_argument("--lora_r", type=int, default=16,
                       help="LoRA attention dimension")
    parser.add_argument("--lora_alpha", type=int, default=32,
                       help="LoRA alpha parameter")
    parser.add_argument("--per_device_train_batch_size", type=int, default=8,
                       help="Per-device training batch size")
    
    return parser.parse_args()

def progressive_stockfish_training(model_path: str, output_dir: str, iterations: int = 10,games_per_iteration: int = 4,per_device_train_batch_size: int = 8):
    """
    Progressive training against increasingly difficult Stockfish opponents
    """
    
    # Stockfish skill progression
    skill_levels = [1,3, 6, 9, 11, 14, 17, 20]
    
    current_model_path = model_path
    
    for i, skill_level in enumerate(skill_levels):
        if i >= iterations:
            break
            
        print(f"Training iteration {i+1}: Stockfish skill level {skill_level}")
        
        trainer = ChessGRPOTrainer(
            model_name=current_model_path,
            output_dir=f"{output_dir}/skill_{skill_level}",
            stockfish_skill_level=skill_level,
            per_device_train_batch_size=per_device_train_batch_size
        )
        
        # Train for a few iterations at this skill level
        trainer.train(num_iterations=1, games_per_iteration=games_per_iteration)
        
        # Update model path for next iteration
        # After GRPO training, the model is saved as a regular model, not PEFT
        current_model_path = f"{output_dir}/skill_{skill_level}-final"
        
        print(f"Completed training against skill level {skill_level}")
    
    print("Progressive training completed!")

def main():
    args = parse_args()
    
    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)
    
    if args.progressive:
        print("Starting progressive GRPO training...")
        progressive_stockfish_training(
            model_path=args.model_path,
            output_dir=args.output_dir,
            iterations=args.num_iterations,
            games_per_iteration=args.games_per_iteration,
            per_device_train_batch_size=args.per_device_train_batch_size
        )
    else:
        print("Starting standard GRPO training...")
        trainer = ChessGRPOTrainer(
            model_name=args.model_path,
            output_dir=args.output_dir,
            stockfish_skill_level=args.initial_skill_level,
            load_in_8bit=args.load_in_8bit,
        )
        
        trainer.train(
            num_iterations=args.num_iterations,
            games_per_iteration=args.games_per_iteration,
            per_device_train_batch_size=args.per_device_train_batch_size
        )
    
    print("GRPO training completed!")

if __name__ == "__main__":
    main()
    #trainer = ChessGRPOTrainer(
    #    model_name="./train_output-final",
    #    output_dir="./chess-grpo-output",
    #    stockfish_skill_level=3
    #)
    #prompts = [
    #    "{\"moveHistory\": [\"e4\", \"e5\",\"Ke2\",\"d6\"]}",
    #    "{\"moveHistory\": [\"e4\", \"e5\",\"Ke2\",\"d6\"]}",
    #    "{\"moveHistory\": [\"e4\", \"e5\",\"Ke2\",\"d6\"]}",
    #    "{\"moveHistory\": [\"e4\", \"e5\",\"Ke2\",\"d6\"]}",
    #    "{\"moveHistory\": [\"e4\", \"e5\",\"Ke2\",\"d6\"]}",
    #    "{\"moveHistory\": [\"e4\", \"e5\",\"Ke2\",\"d6\"]}",
    #]
    #completions = [
    #    "{\"move\": \"Ke1\", \"reasoning\": \"eval: 100\"}", 
    #    "{\"move\": \"d4\", \"reasoning\": \"eval: 100\"}",
    #    "{\"move\": \"d3\", \"reasoning\": \"eval: 100\"}",
    #    "{\"move\": \"cxd4\", \"reasoning\": \"eval: 100\"}",
    #    "{\"move\": dkfndkajnfkdjan",
    #    "{\"move\": \"Ke2\', \"reasoning\": \"eval: 100\"}"

    #]
    #board_state = [
    #    "rnbqkbnr/ppp2ppp/3p4/4p3/4P3/8/PPPPKPPP/RNBQ1BNR w kq - 0 3",
    #    "rnbqkbnr/ppp2ppp/3p4/4p3/4P3/8/PPPPKPPP/RNBQ1BNR w kq - 0 3",
    #    "rnbqkbnr/ppp2ppp/3p4/4p3/4P3/8/PPPPKPPP/RNBQ1BNR w kq - 0 3",
    #    "rnbqkbnr/ppp2ppp/3p4/4p3/4P3/8/PPPPKPPP/RNBQ1BNR w kq - 0 3",
    #    "rnbqkbnr/ppp2ppp/3p4/4p3/4P3/8/PPPPKPPP/RNBQ1BNR w kq - 0 3",
    #    "rnbqkbnr/ppp2ppp/3p4/4p3/4P3/8/PPPPKPPP/RNBQ1BNR w kq - 0 3",
    #]
    #trainer.chess_reward_function(completions, prompts=prompts,board_state=board_state)
