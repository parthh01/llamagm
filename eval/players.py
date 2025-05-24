import chess 
import stockfish
from abc import ABC, abstractmethod
import os 
import random 
from transformers import AutoModelForCausalLM, AutoTokenizer
import json 
from constants import system_prompt
import threading
from peft import PeftModel, PeftConfig

#derived from https://lichess.org/forum/general-chess-discussion/what-elo-are-the-various-stockfish-levels and https://www.reddit.com/r/chess/comments/ltzzon/what_is_the_approximate_elo_rating_of_each_of_the/
#really we are just using this for relative strength so even if these elo ratings aren't quite accurate, that it's the idential ratings used for all evals allows for relative comparisons
stockfish_skill_elo_map = {
    850: {"Skill Level": 3,"Threads": 2,"Depth": 1},
    950: {"Skill Level": 6, "Threads": 2,"Depth": 2},
    1050: {"Skill Level": 9, "Threads": 2,"Depth": 3},
    1250: {"Skill Level": 11, "Threads": 2,"Depth": 4},
    1700: {"Skill Level": 14, "Threads": 2,"Depth": 6},
    1900: {"Skill Level": 17, "Threads": 1,"Depth": 8},
    2000: {"Skill Level": 20, "Threads": 1,"Depth": 10}
}

class IllegalMoveError(Exception):
    """Exception raised for illegal moves."""
    def __init__(self, message):
        self.message = message
        super().__init__(self.message)

class BasePlayer(ABC):
    @abstractmethod
    def get_move(self, board: chess.Board, time_left: int) -> chess.Move:
        pass

class StockfishPlayer(BasePlayer):
    def __init__(self, params:dict):
        self.stockfish = stockfish.Stockfish(path=os.getenv('STOCKFISH_PATH', '/usr/local/bin/stockfish'))
        self.stockfish.set_skill_level(params["Skill Level"])
        self.stockfish.update_engine_parameters({"Threads": params["Threads"]})
        self.stockfish.set_depth(params["Depth"])
        self.lock = threading.Lock()

    def get_move(self, board: chess.Board, time_left: int) -> chess.Move:
        self.stockfish.set_fen_position(board.fen())
        move = self.stockfish.get_best_move()
        return chess.Move.from_uci(move)
        
    def evaluate_position(self, board: chess.Board) -> float:
        """
        Evaluate the current position using Stockfish.
        Returns a float representing the evaluation in pawns from white's perspective.
        Positive values favor white, negative values favor black.
        """
        with self.lock:
            self.stockfish.set_fen_position(board.fen())
            evaluation = self.stockfish.get_evaluation()
            
            if evaluation['type'] == 'cp':
                # Convert centipawns to pawns
                return evaluation['value']
            elif evaluation['type'] == 'mate':
                # Return a large value for mate, with sign indicating which side is winning
                # Positive mate score means white is winning, negative means black is winning
                return 999 if evaluation['value'] > 0 else -999
            else:
                # Fallback for unexpected evaluation type
                return 0.5
    
    def __del__(self):
        # Clean up Stockfish process when the player is destroyed
        if hasattr(self, 'stockfish'):
            try:
                self.stockfish.__del__()
            except Exception as e:
                print(f"Error cleaning up Stockfish: {e}")

class RandomPlayer(BasePlayer):
    def get_move(self, board: chess.Board, time_left: int) -> chess.Move:
        return random.choice(list(board.legal_moves))

class LLMPlayer(BasePlayer):
    def __init__(self, dir_path: str):
        self.model, self.tokenizer = self._load_model(dir_path)
    
    def _load_model(self, model_path: str):
        """Load model with PEFT support"""
        try:
            # Try to load as PEFT model first
            peft_config = PeftConfig.from_pretrained(model_path)
            base_model = AutoModelForCausalLM.from_pretrained(
                peft_config.base_model_name_or_path,
                device_map="auto"
            )
            model = PeftModel.from_pretrained(base_model, model_path)
            
            # Load tokenizer
            try:
                tokenizer = AutoTokenizer.from_pretrained(model_path)
            except:
                tokenizer = AutoTokenizer.from_pretrained(peft_config.base_model_name_or_path)
            
            print(f"Successfully loaded PEFT model from {model_path}")
            
        except Exception as e:
            print(f"Failed to load as PEFT model: {e}")
            print(f"Attempting to load as regular model...")
            
            # Fallback to regular model loading
            model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto")
            tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        # Fix the padding token issue
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        return model, tokenizer

    def get_move_history_in_san(self, board: chess.Board) -> list:
        """Generate SAN move history by replaying the game from the beginning."""
        move_history = []
        temp_board = chess.Board()
        
        for move in board.move_stack:
            san_move = temp_board.san(move)
            move_history.append(san_move)
            temp_board.push(move)
            
        return move_history

    def get_move(self, board: chess.Board, time_left: int) -> chess.Move:
        position_input = {
            "moveHistory": self.get_move_history_in_san(board),
            "possibleMoves": [board.san(move) for move in board.legal_moves],
            "color": "w" if board.turn else "b"
        }
        prompt = f"[INST] {system_prompt}\n\n{json.dumps(position_input)} [/INST]" 
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        outputs = self.model.generate(inputs["input_ids"],max_new_tokens=40)
        assert len(outputs[0]) <= 40, f"LLM response too long: {len(outputs[0])}"
        response = self.tokenizer.decode(outputs[0],skip_special_tokens=True)
        model_response = response.split("[/INST]")[1].strip()
        response_json = json.loads(model_response)
        
        move_str = response_json["move"]
        # Check if the move is in the list of legal moves
        legal_moves_san = [board.san(move) for move in board.legal_moves]
        if move_str not in legal_moves_san:
            raise IllegalMoveError(f"Illegal move suggested by LLM: {move_str}")
            
        return board.parse_san(response_json["move"])


if __name__ == "__main__":
    board = chess.Board()
    #player = StockfishPlayer(stockfish_skill_elo_map[850])
    #player = RandomPlayer()
    print('starting game')
    player = LLMPlayer("./open_llama_7b-lora-final")
    move = player.get_move(board,1000)
    print(move)
    board.push(move)
    print(board.fen())
