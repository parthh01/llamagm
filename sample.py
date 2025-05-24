from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import os 
import json 
import torch 
import chess
import random
from constants import system_prompt
from dotenv import load_dotenv

load_dotenv()

# Import the StockfishPlayer from eval module
from eval.players import StockfishPlayer, stockfish_skill_elo_map

#model_name = "./open_llama_7b-lora-final"
model_name = "./chess-grpo-output/checkpoint-150"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    #quantization_config=BitsAndBytesConfig(load_in_8bit=True),
    torch_dtype="auto",
    device_map="auto",
    offload_folder="./offload_folder",
)

def get_random_move(board):
    """Get a random legal move."""
    try:
        legal_moves = list(board.legal_moves)
        if legal_moves:
            move = random.choice(legal_moves)
            move_san = board.san(move)
            return move, move_san
        else:
            return None, "NO_LEGAL_MOVES"
    except Exception as e:
        print(f"Random player error: {e}")
        return None, "ERROR"

def get_llm_move(board):
    """Get a move from the LLM given the current board position."""
    # Convert board to position data format
    move_history = []
    temp_board = chess.Board()
    
    # Reconstruct move history in algebraic notation
    for move in board.move_stack:
        move_history.append(temp_board.san(move))
        temp_board.push(move)
    
    # Get possible moves in algebraic notation
    possible_moves = [board.san(move) for move in board.legal_moves]
    
    position_data = {
        "moveHistory": move_history,
        "possibleMoves": possible_moves,
        "color": "w" if board.turn == chess.WHITE else "b"
    }
    
    # Format input with system prompt
    prompt = f"[INST] {system_prompt}\n\n{json.dumps(position_data)} [/INST]" 
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    # Generate response
    outputs = model.generate(
        inputs["input_ids"],
        max_new_tokens=40,
        temperature=0
    )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Parse JSON response
    try:
        # Extract response (after instruction)
        model_response = response.split("[/INST]")[1].strip()
        response_json = json.loads(model_response)
        
        # Get the suggested move
        suggested_move = response_json.get("move", "")
        
        # Convert algebraic notation to chess.Move
        try:
            move = board.parse_san(suggested_move)
            if move in board.legal_moves:
                return move, suggested_move
            else:
                print(f"LLM suggested illegal move: {suggested_move}")
                return None, suggested_move
        except:
            print(f"LLM suggested invalid move format: {suggested_move}")
            return None, suggested_move
            
    except Exception as e:
        print(f"Error parsing LLM response: {e}")
        print(f"Raw response: {response}")
        return None, "ERROR"

def get_stockfish_move(stockfish_player, board):
    """Get a move from Stockfish using StockfishPlayer."""
    try:
        with stockfish_player.lock:
            move = stockfish_player.get_move(board, 10000)  # 10 second time limit
            move_san = board.san(move)
            return move, move_san
    except Exception as e:
        print(f"Stockfish error: {e}")
        return None, "ERROR"

def evaluate_position_with_stockfish(board, depth=15):
    """Evaluate the current position using Stockfish and return a score."""
    try:
        # Create a temporary strong Stockfish instance for evaluation
        eval_params = {"Skill Level": 20, "Threads": 1, "Depth": 15}
        
        evaluator = StockfishPlayer(eval_params)
        
        eval_score = evaluator.evaluate_position(board)
        print(f"Stockfish evaluation: {eval_score:.2f} (positive favors White)")
        
        return eval_score
            
    except Exception as e:
        print(f"Error evaluating position: {e}")
        return 0.0  # Return draw if evaluation fails

def play_full_game(opponent="stockfish", llm_is_white=True, stockfish_elo=1250):
    """
    Play a full game between LLM and opponent with move-by-move output.
    Game truncates after 50 moves (25 per side) and winner is decided by Stockfish evaluation.
    
    Args:
        opponent: "stockfish" or "random"
        llm_is_white: Whether LLM plays as white
        stockfish_elo: ELO level for Stockfish (only used if opponent="stockfish")
    """
    
    # Initialize opponent
    stockfish_player = None
    if opponent == "stockfish":
        if stockfish_elo not in stockfish_skill_elo_map:
            print(f"Invalid Stockfish ELO: {stockfish_elo}. Available: {list(stockfish_skill_elo_map.keys())}")
            return
        
        stockfish_params = stockfish_skill_elo_map[stockfish_elo]
        stockfish_player = StockfishPlayer(stockfish_params)
        opponent_name = f"Stockfish (ELO {stockfish_elo})"
    elif opponent == "random":
        opponent_name = "Random Player"
    else:
        print(f"Unknown opponent: {opponent}")
        return
    
    board = chess.Board()
    move_count = 1
    max_moves = 50  # 25 moves per side
    
    print("=" * 60)
    print(f"CHESS GAME: {'LLM (White) vs ' + opponent_name + ' (Black)' if llm_is_white else opponent_name + ' (White) vs LLM (Black)'}")
    print(f"Game will truncate after {max_moves} moves if not finished")
    print("=" * 60)
    print(f"Starting position:\n{board}")
    print()
    
    try:
        while not board.is_game_over() and len(board.move_stack) < max_moves:
            current_player_is_llm = (board.turn == chess.WHITE) == llm_is_white
            
            if current_player_is_llm:
                print(f"Move {move_count}: LLM ({'White' if board.turn == chess.WHITE else 'Black'}) to move")
                print(f"Position: {board.fen()}")
                
                move, move_san = get_llm_move(board)
                
                if move is None:
                    print("LLM failed to provide valid move. Making random legal move.")
                    move, move_san = get_random_move(board)
                
                print(f"LLM plays: {move_san}")
                board.push(move)
                
            else:
                print(f"Move {move_count}: {opponent_name} ({'White' if board.turn == chess.WHITE else 'Black'}) to move")
                print(f"Position: {board.fen()}")
                
                if opponent == "stockfish":
                    move, move_san = get_stockfish_move(stockfish_player, board)
                elif opponent == "random":
                    move, move_san = get_random_move(board)
                
                if move is None:
                    print(f"{opponent_name} failed to provide move!")
                    break
                
                print(f"{opponent_name} plays: {move_san}")
                board.push(move)
            
            print(f"Board after move:\n{board}")
            print("-" * 40)
            
            if board.turn == chess.WHITE:
                move_count += 1
        
        # Game over - print result
        print("=" * 60)
        print("GAME OVER")
        print("=" * 60)
        
        # Determine game result
        if len(board.move_stack) >= max_moves and not board.is_game_over():
            # Game truncated - use Stockfish evaluation
            print(f"Game truncated after {max_moves} moves. Using Stockfish evaluation...")
            eval_score = evaluate_position_with_stockfish(board)
            print(f"Stockfish evaluation: {eval_score:.2f} (positive favors White)")
            
            if eval_score > 0.5:  # White is winning
                result = "1-0"
                winner = "White (by evaluation)"
            elif eval_score < -0.5:  # Black is winning
                result = "0-1"
                winner = "Black (by evaluation)"
            else:  # Close to equal
                result = "1/2-1/2"
                winner = "Draw (by evaluation)"
                
            print(f"Game ended by move limit and evaluation")
        else:
            # Normal game end
            result = board.result()
            if result == "1-0":
                winner = "White"
            elif result == "0-1":
                winner = "Black"
            else:
                winner = "Draw"
            
            if board.is_checkmate():
                print("Game ended by checkmate")
            elif board.is_stalemate():
                print("Game ended by stalemate")
            elif board.is_insufficient_material():
                print("Game ended by insufficient material")
            elif board.can_claim_draw():
                print("Game ended by draw claim")
        
        print(f"Result: {result} ({winner})")
        print(f"Final position:\n{board}")
        print(f"Total moves played: {len(board.move_stack)}")
        
        # Print full game in PGN format
        print("\nGame in PGN format:")
        game_pgn = ""
        temp_board = chess.Board()
        move_num = 1
        
        for i, move in enumerate(board.move_stack):
            if i % 2 == 0:
                game_pgn += f"{move_num}. "
            game_pgn += temp_board.san(move) + " "
            temp_board.push(move)
            if i % 2 == 1:
                move_num += 1
        
        game_pgn += result
        print(game_pgn)
        
    except KeyboardInterrupt:
        print("\nGame interrupted by user")
    except Exception as e:
        print(f"Error during game: {e}")
    finally:
        # Clean up Stockfish player if it was created
        if stockfish_player:
            try:
                del stockfish_player
            except:
                pass

if __name__ == "__main__":
    # Play a game with LLM as white against random player
    print("Playing game with LLM as White vs Random Player...")
    play_full_game(opponent="random", llm_is_white=True)
    
    print("\n" + "="*80 + "\n")
    
    # Play a game with LLM as white against Stockfish
    #print("Playing game with LLM as White vs Stockfish...")
    #play_full_game(opponent="stockfish", llm_is_white=True, stockfish_elo=1250)
    
    print("\n" + "="*80 + "\n")
    
    # Uncomment to play more games
    # print("Playing game with LLM as Black vs Random Player...")
    # play_full_game(opponent="random", llm_is_white=False)

