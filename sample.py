from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import os 
import json 
import torch 
import chess
import chess.engine
from constants import system_prompt

model_name = "./open_llama_7b-lora-final"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    #quantization_config=BitsAndBytesConfig(load_in_8bit=True),
    torch_dtype="auto",
    device_map="auto",
    offload_folder="./offload_folder",
)

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
        do_sample=True,
        temperature=0.7
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

def get_stockfish_move(board, engine, time_limit=1.0):
    """Get a move from Stockfish."""
    try:
        result = engine.play(board, chess.engine.Limit(time=time_limit))
        move_san = board.san(result.move)
        return result.move, move_san
    except Exception as e:
        print(f"Stockfish error: {e}")
        return None, "ERROR"

def play_full_game(llm_is_white=True, stockfish_time=1.0, stockfish_depth=10):
    """Play a full game between LLM and Stockfish with move-by-move output."""
    
    # Initialize Stockfish engine
    try:
        engine = chess.engine.SimpleEngine.popen_uci("/usr/games/stockfish")  # Adjust path as needed
        engine.configure({"Skill Level": 10})
    except:
        try:
            engine = chess.engine.SimpleEngine.popen_uci("stockfish")  # Try without path
            engine.configure({"Skill Level": 10, "Depth": stockfish_depth})
        except Exception as e:
            print(f"Could not initialize Stockfish: {e}")
            return
    
    board = chess.Board()
    move_count = 1
    
    print("=" * 60)
    print(f"CHESS GAME: {'LLM (White) vs Stockfish (Black)' if llm_is_white else 'Stockfish (White) vs LLM (Black)'}")
    print("=" * 60)
    print(f"Starting position:\n{board}")
    print()
    
    try:
        while not board.is_game_over() and move_count <= 100:  # Limit to 100 moves
            current_player_is_llm = (board.turn == chess.WHITE) == llm_is_white
            
            if current_player_is_llm:
                print(f"Move {move_count}: LLM ({'White' if board.turn == chess.WHITE else 'Black'}) to move")
                print(f"Position: {board.fen()}")
                
                move, move_san = get_llm_move(board)
                
                if move is None:
                    print("LLM failed to provide valid move. Making random legal move.")
                    move = list(board.legal_moves)[0]
                    move_san = board.san(move)
                
                print(f"LLM plays: {move_san}")
                board.push(move)
                
            else:
                print(f"Move {move_count}: Stockfish ({'White' if board.turn == chess.WHITE else 'Black'}) to move")
                print(f"Position: {board.fen()}")
                
                move, move_san = get_stockfish_move(board, engine, stockfish_time)
                
                if move is None:
                    print("Stockfish failed to provide move!")
                    break
                
                print(f"Stockfish plays: {move_san}")
                board.push(move)
            
            print(f"Board after move:\n{board}")
            print("-" * 40)
            
            if board.turn == chess.WHITE:
                move_count += 1
        
        # Game over - print result
        print("=" * 60)
        print("GAME OVER")
        print("=" * 60)
        
        result = board.result()
        if result == "1-0":
            winner = "White"
        elif result == "0-1":
            winner = "Black"
        else:
            winner = "Draw"
        
        print(f"Result: {result} ({winner})")
        
        if board.is_checkmate():
            print("Game ended by checkmate")
        elif board.is_stalemate():
            print("Game ended by stalemate")
        elif board.is_insufficient_material():
            print("Game ended by insufficient material")
        elif board.can_claim_draw():
            print("Game ended by draw claim")
        elif move_count > 100:
            print("Game ended by move limit")
        
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
        engine.quit()

if __name__ == "__main__":
    # Play a game with LLM as white
    print("Playing game with LLM as White...")
    play_full_game(llm_is_white=True, stockfish_time=1.0, stockfish_depth=10)
    
    print("\n" + "="*80 + "\n")
    
    # Uncomment to play another game with LLM as black
    # print("Playing game with LLM as Black...")
    # play_full_game(llm_is_white=False, stockfish_time=1.0, stockfish_depth=10)

