import torch
import json
import chess
import chess.engine
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig
from trl import GRPOConfig, GRPOTrainer
from datasets import Dataset
import os
from dotenv import load_dotenv
from stockfish import Stockfish
import threading
import re
from datagen.gen import progressive_stockfish_training, generate_grpo_games
import argparse
load_dotenv()

@dataclass
class ChessReward:
    """Reward structure for chess moves"""
    json_parse_reward: float = -10.0  # Large negative for invalid JSON
    illegal_move_reward: float = -5.0  # Smaller negative for illegal moves
    value_parse_reward: float = -1.0   # Small negative for unparseable value
    value_accuracy_weight: float = 0.01  # Weight for value estimation accuracy
    position_improvement_weight: float = 2.0  # Weight for position improvement
    win_reward: float = 20.0  # Large positive for wins
    draw_reward: float = 0.0  # Neutral for draws
    loss_reward: float = -20.0  # Large negative for losses

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
            # Try to find JSON in the output
            json_match = re.search(r'\{.*\}', output, re.DOTALL)
            if json_match:
                json_str = json_match.group()
                parsed = json.loads(json_str)
                move = parsed.get("move")
                reasoning = parsed.get("reasoning", "")
                return move, reasoning, True
            else:
                return None, None, False
        except (json.JSONDecodeError, KeyError):
            return None, None, False
    
    def extract_value_from_reasoning(self, reasoning: str) -> Optional[float]:
        """
        Extract numerical value estimation from reasoning text.
        Returns value in centipawns or None if not found.
        """
        if not reasoning:
            return None
            
        # Look for patterns like "eval: +150", "advantage: -200", "mate in 3", etc.
        patterns = [
            r'eval[:\s]+([+-]?\d+)',
            r'advantage[:\s]+([+-]?\d+)',
            r'position[:\s]+([+-]?\d+)',
            r'([+-]?\d+)\s*centipawn',
            r'([+-]?\d+)\s*cp',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, reasoning.lower())
            if match:
                try:
                    return float(match.group(1))
                except ValueError:
                    continue
        
        # Look for mate patterns
        mate_pattern = r'mate\s+in\s+(\d+)'
        mate_match = re.search(mate_pattern, reasoning.lower())
        if mate_match:
            mate_moves = int(mate_match.group(1))
            # Convert mate to large centipawn equivalent
            return 9999 if "white" in reasoning.lower() else -9999
            
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
    
    def calculate_reward(self, board: chess.Board, model_output: str, 
                        move_made: Optional[chess.Move] = None) -> Tuple[float, Dict]:
        """
        Calculate reward for a model output given the board state.
        Returns: (reward, info_dict)
        """
        info = {}
        reward = 0.0
        
        # Parse model output
        move_str, reasoning, is_valid_json = self.parse_model_output(model_output)
        info['valid_json'] = is_valid_json
        info['move_str'] = move_str
        info['reasoning'] = reasoning
        
        # Reward for valid JSON parsing
        if not is_valid_json:
            reward += self.reward_config.json_parse_reward
            info['json_parse_penalty'] = self.reward_config.json_parse_reward
            return reward, info
        
        # Check if move is legal
        legal_moves = list(board.legal_moves)
        legal_moves_san = [board.san(move) for move in legal_moves]
        
        if move_str not in legal_moves_san:
            reward += self.reward_config.illegal_move_reward
            info['illegal_move_penalty'] = self.reward_config.illegal_move_reward
            return reward, info
        
        # Parse the move
        try:
            move = board.parse_san(move_str)
            info['move'] = move
        except ValueError:
            reward += self.reward_config.illegal_move_reward
            info['move_parse_error'] = True
            return reward, info
        
        # Extract value estimation from reasoning
        estimated_value = self.extract_value_from_reasoning(reasoning)
        info['estimated_value'] = estimated_value
        
        if estimated_value is None:
            reward += self.reward_config.value_parse_reward
            info['value_parse_penalty'] = self.reward_config.value_parse_reward
        
        # Make the move and evaluate position
        board_before = board.copy()
        is_white_move = board.turn
        board.push(move)
        
        # Get actual Stockfish evaluation after move
        actual_value = self.get_stockfish_evaluation(board)
        info['actual_value'] = actual_value
        
        # Reward for value estimation accuracy
        if estimated_value is not None:
            value_error = abs(estimated_value - actual_value)
            # Reward decreases with error, max reward when error is 0
            value_accuracy_reward = max(0, 100 - value_error) * self.reward_config.value_accuracy_weight
            reward += value_accuracy_reward
            info['value_accuracy_reward'] = value_accuracy_reward
        
        # Reward for position improvement
        if self.is_position_improving(board_before, board, is_white_move):
            improvement_reward = self.reward_config.position_improvement_weight
            reward += improvement_reward
            info['improvement_reward'] = improvement_reward
        else:
            improvement_penalty = -self.reward_config.position_improvement_weight / 2
            reward += improvement_penalty
            info['improvement_penalty'] = improvement_penalty
        
        info['total_reward'] = reward
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
            if abs(final_eval) < 50:  # Close game, call it a draw
                return self.reward_config.draw_reward
            elif final_eval > 0:  # White advantage
                return self.reward_config.win_reward
            else:  # Black advantage
                return self.reward_config.loss_reward
        
        return 0.0  # Game not finished

class ChessGRPOTrainer:
    """GRPO trainer for chess"""
    
    def __init__(self, model_name: str, output_dir: str, stockfish_skill_level: int = 3):
        self.model_name = model_name
        self.output_dir = output_dir
        self.stockfish_skill_level = stockfish_skill_level
        
        # Load model and tokenizer with PEFT support
        self.model, self.tokenizer = self._load_peft_model(model_name)
        
        # Initialize environment
        self.env = ChessGRPOEnvironment(skill_level=stockfish_skill_level)
        
        # GRPO configuration
        self.grpo_config = GRPOConfig(
            output_dir=output_dir,
            learning_rate=1e-5,
            logging_steps=10,
            fp16=True
        )
    
    def _load_peft_model(self, model_path: str):
        """Load PEFT model and tokenizer"""
        try:
            # Try to load as PEFT model first
            peft_config = PeftConfig.from_pretrained(model_path)
            base_model = AutoModelForCausalLM.from_pretrained(
                peft_config.base_model_name_or_path,
                device_map="auto"
            )
            model = PeftModel.from_pretrained(base_model, model_path)
            
            # Merge and unload for GRPO training (GRPO needs a regular model, not PEFT)
            model = model.merge_and_unload()
            model = model.bfloat16()
            
            # IMPORTANT: Enable gradients for all parameters after merge_and_unload
            for param in model.parameters():
                param.requires_grad = True
            
            # Load tokenizer from the PEFT model directory or base model
            try:
                tokenizer = AutoTokenizer.from_pretrained(model_path)
            except:
                tokenizer = AutoTokenizer.from_pretrained(peft_config.base_model_name_or_path)
            
            print(f"Successfully loaded PEFT model from {model_path}")
            print(f"Base model: {peft_config.base_model_name_or_path}")
            
        except Exception as e:
            print(f"Failed to load as PEFT model: {e}")
            print(f"Attempting to load as regular model...")
            
            # Fallback to regular model loading
            model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto")
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            
            # Ensure gradients are enabled for regular models too
            for param in model.parameters():
                param.requires_grad = True
        
        # Set pad token if not present
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
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
        
        for completion, fen_string in zip(completions, board_states):
            try:
                # Construct chess board from FEN string
                board_state = chess.Board(fen_string)
                # Calculate reward using our existing environment
                reward, _ = self.env.calculate_reward(board_state, completion)
                rewards.append(reward)
            except Exception as e:
                # Fallback reward for any errors
                print(f"Error calculating reward: {e}")
                rewards.append(-10.0)  # Large negative reward for errors
        print('rewards: ', rewards) 
        return rewards
    
    def train(self, num_iterations: int = 10, games_per_iteration: int = 100):
        """Main training loop"""
        for iteration in range(num_iterations):
            print(f"GRPO Iteration {iteration + 1}/{num_iterations}")
            
            # Generate new training data
            print("Generating game data...")
            dataset = self.generate_game_data(games_per_iteration)
            
            # Create GRPO trainer with reward function
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
        
        # Save final model
        trainer.save_model(f"{self.output_dir}/final")


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
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)
    
    if args.progressive:
        print("Starting progressive GRPO training...")
        progressive_stockfish_training(
            model_path=args.model_path,
            output_dir=args.output_dir,
            iterations=args.num_iterations
        )
    else:
        print("Starting standard GRPO training...")
        trainer = ChessGRPOTrainer(
            model_name=args.model_path,
            output_dir=args.output_dir,
            stockfish_skill_level=args.initial_skill_level
        )
        
        trainer.train(
            num_iterations=args.num_iterations,
            games_per_iteration=args.games_per_iteration
        )
    
    print("GRPO training completed!")

if __name__ == "__main__":
    main()
