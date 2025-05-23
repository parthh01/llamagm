import os 
import uuid
from dotenv import load_dotenv
from sqlalchemy import create_engine, insert, ForeignKey, Column, String, Integer, UUID, Float, or_
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import Session
from stockfish import Stockfish
import chess 
import pandas as pd 
from tqdm import tqdm
import concurrent.futures
import threading
from functools import partial
import torch
import numpy as np
from typing import List, Dict
from constants import system_prompt
import json

load_dotenv()

DATABASE_URL = f"postgresql://{os.getenv('PGUSER')}:{os.getenv('PGPASSWORD')}@{os.getenv('PGHOST')}:{os.getenv('PGPORT')}/{os.getenv('PGDATABASE')}?sslmode=require"
engine = create_engine(DATABASE_URL, echo=False, pool_size=20, max_overflow=40, pool_timeout=30, pool_recycle=1800, pool_pre_ping=True)
Base = declarative_base()

# Thread-local storage for Stockfish instances
thread_local = threading.local()

class Game(Base):
    __tablename__ = 'games'
    id = Column(UUID, primary_key=True)
    source = Column(String)
    pgn = Column(String)
    move = Column(String)
    engine = Column(String)
    engine_rating = Column(Integer)
    value = Column(String) #will be either an integer showing centipawn advantage (+ve for white, -ve for black) or will start with an M denoting forced mate, (integer following M will be +ve if forced win for white else -ve)
    description = Column(String,nullable=True)
    category = Column(String) # opening, endgame, or midgame 

# Create all tables defined above
Base.metadata.create_all(engine)

def get_stockfish():
    """Get a thread-local Stockfish instance"""
    if not hasattr(thread_local, "stockfish"):
        stockfish = Stockfish(depth=18, path=os.getenv('STOCKFISH_PATH', '/usr/local/bin/stockfish'), 
                             parameters={"Threads": 1, "Hash": 32})
        stockfish.set_elo_rating(4000)
        thread_local.stockfish = stockfish
    return thread_local.stockfish

def evaluate_game(fen_string, engine):
    engine.set_fen_position(fen_string)
    evaluation = engine.get_evaluation() 
    value = f"M{evaluation['value']}" if evaluation['type'] == 'mate' else f"{evaluation['value']}"
    best_move = engine.get_best_move()
    move = chess.Move.from_uci(best_move)
    board = chess.Board(fen_string)
    return value, board.san(move)

def process_opening(row, batch_size=100):
    """Process a single opening and return records to insert"""
    opening_name = row['name']
    pgn_text = row['pgn']
    eco = row['eco']
    
    # Get thread-local Stockfish instance
    chess_engine = get_stockfish()
    
    # Create a new board
    board = chess.Board()
    
    # Split the PGN into moves
    moves_text = pgn_text.split()
    current_pgn = ""
    
    # Records to insert
    records_to_insert = []
    
    # First, check which positions already exist in the database
    positions_to_check = []
    
    # Process each move to build positions to check
    for move_text in moves_text:
        # Skip move numbers (like "1.", "2.", etc.)
        if '.' in move_text:
            continue
            
        # Parse the move
        try:
            move = board.parse_san(move_text)
            san_move = board.san(move)  # Get the standard algebraic notation
            
            # Update the current PGN
            if current_pgn:
                next_pgn = f"{current_pgn} {san_move}"
            else:
                next_pgn = san_move
            
            positions_to_check.append({
                'source': f"ECO {eco}",
                'pgn': current_pgn,
                'move': san_move,
                'next_pgn': next_pgn,
                'fen': board.fen()
            })
            
            # Make the move on the board
            board.push(move)
            current_pgn = next_pgn
            
        except ValueError as e:
            print(f"Error parsing move {move_text} in {opening_name}: {e}")
            continue
    
    # If no valid positions, return empty list
    if not positions_to_check:
        return []
    
    # Check which positions already exist in the database
    with Session(engine) as session:
        # Create a list of conditions to check
        conditions = []
        for pos in positions_to_check:
            conditions.append(
                (Game.source == pos['source']) & 
                (Game.pgn == pos['pgn']) & 
                (Game.move == pos['move'])
            )
        
        existing_positions = session.query(Game.source, Game.pgn, Game.move).filter(
            or_(*conditions)
        ).all()
        
        # Convert to set for faster lookup
        existing_set = {(ep.source, ep.pgn, ep.move) for ep in existing_positions}
    
    # Process positions that don't exist yet
    for pos in positions_to_check:
        key = (pos['source'], pos['pgn'], pos['move'])
        if key not in existing_set:
            # Evaluate the position
            value, best_move = evaluate_game(pos['fen'], chess_engine)
            
            # Create record
            records_to_insert.append({
                'id': uuid.uuid4(),
                'source': pos['source'],
                'pgn': pos['pgn'],
                'move': pos['move'],
                'engine': 'stockfish',
                'engine_rating': chess_engine.get_parameters()['UCI_Elo'],
                'value': value,
                'description': opening_name,
                'category': 'opening'
            })
    
    # Bulk insert records
    if records_to_insert:
        with Session(engine) as session:
            session.execute(insert(Game), records_to_insert)
            session.commit()
    
    return records_to_insert

def load_openings_parallel(df, max_workers=4, batch_size=100):
    """Process openings in parallel"""
    total_inserted = 0
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Process each opening in parallel
        future_to_row = {executor.submit(process_opening, row): idx for idx, row in df.iterrows()}
        
        # Show progress
        for future in tqdm(concurrent.futures.as_completed(future_to_row), total=len(future_to_row)):
            idx = future_to_row[future]
            try:
                records = future.result()
                total_inserted += len(records)
            except Exception as exc:
                print(f'Opening at index {idx} generated an exception: {exc}')
    
    print(f"Total new positions inserted: {total_inserted}")

def prune_openings(opening_trajectories, white_engine_params, black_engine_progression, max_moves_per_trajectory=50):
    """
    Prune existing white moves from openings and generate new trajectories.
    
    Args:
        opening_trajectories: List of lists, each containing a sequence of moves
        white_engine_params: Stockfish parameters for the white player (strongest engine)
        black_engine_progression: List of Stockfish parameters for black players (progressively stronger)
        max_moves_per_trajectory: Maximum number of moves to play out each trajectory
    """
    
    # Step 1: Remove existing white moves from the database
    print("Removing existing white moves from database...")
    with Session(engine) as session:
        # Query all games where it's white to move (pgn length is even or 0)
        all_games = session.query(Game).filter(Game.category == 'opening').all()
        white_move_ids = []
        
        for game in all_games:
            pgn_moves = game.pgn.split() if game.pgn else []
            # If pgn length is even (0, 2, 4, ...), it's white to move
            if len(pgn_moves) % 2 == 0:
                white_move_ids.append(game.id)
        
        # Delete white moves in batches
        batch_size = 1000
        for i in range(0, len(white_move_ids), batch_size):
            batch_ids = white_move_ids[i:i + batch_size]
            session.query(Game).filter(Game.id.in_(batch_ids)).delete(synchronize_session=False)
            session.commit()
        
        print(f"Removed {len(white_move_ids)} existing white moves from database")
    
    # Step 2: Generate new trajectories
    print("Generating new trajectories...")
    
    def play_trajectory_with_engines(trajectory, white_params, black_params_list):
        """Play out a single trajectory with different black engines"""
        records_to_insert = []
        
        for black_params in black_params_list:
            # Get thread-local engines
            white_engine = get_stockfish()
            black_engine = Stockfish(depth=18, path=os.getenv('STOCKFISH_PATH', '/usr/local/bin/stockfish'))
            
            # Configure engines
            white_engine.set_skill_level(white_params.get("Skill Level", 20))
            white_engine.update_engine_parameters({"Threads": white_params.get("Threads", 1)})
            white_engine.set_depth(white_params.get("Depth", 18))
            
            black_engine.set_skill_level(black_params.get("Skill Level", 20))
            black_engine.update_engine_parameters({"Threads": black_params.get("Threads", 1)})
            black_engine.set_depth(black_params.get("Depth", 18))
            
            # Initialize board and play opening moves
            board = chess.Board()
            current_pgn = ""
            
            # Play the opening trajectory
            for move_text in trajectory:
                try:
                    move = board.parse_san(move_text)
                    san_move = board.san(move)
                    
                    # Update PGN
                    if current_pgn:
                        current_pgn = f"{current_pgn} {san_move}"
                    else:
                        current_pgn = san_move
                    
                    board.push(move)
                except ValueError as e:
                    print(f"Error parsing opening move {move_text}: {e}")
                    break
            
            # Continue the game with engines
            moves_played = len(trajectory)
            while not board.is_game_over() and moves_played < max_moves_per_trajectory:
                try:
                    if board.turn == chess.WHITE:
                        # White's turn - use white engine
                        white_engine.set_fen_position(board.fen())
                        best_move_uci = white_engine.get_best_move()
                        
                        if best_move_uci is None:
                            break
                            
                        move = chess.Move.from_uci(best_move_uci)
                        san_move = board.san(move)
                        
                        # Evaluate position before making the move
                        evaluation = white_engine.get_evaluation()
                        value = f"M{evaluation['value']}" if evaluation['type'] == 'mate' else f"{evaluation['value']}"
                        
                        # Create record for white move
                        records_to_insert.append({
                            'id': uuid.uuid4(),
                            'source': f"Generated_Trajectory_Black_Skill_{black_params.get('Skill Level', 'Unknown')}",
                            'pgn': current_pgn,
                            'move': san_move,
                            'engine': 'stockfish',
                            'engine_rating': white_engine.get_parameters().get('UCI_Elo', 3200),
                            'value': value,
                            # 'description': f"White move vs Black skill level {black_params.get('Skill Level', 'Unknown')}",
                            'category': 'midgame'
                        })
                        
                        # Update PGN
                        if current_pgn:
                            current_pgn = f"{current_pgn} {san_move}"
                        else:
                            current_pgn = san_move
                        
                        board.push(move)
                        
                    else:
                        # Black's turn - use black engine
                        black_engine.set_fen_position(board.fen())
                        best_move_uci = black_engine.get_best_move()
                        
                        if best_move_uci is None:
                            break
                            
                        move = chess.Move.from_uci(best_move_uci)
                        san_move = board.san(move)
                        
                        # Update PGN (but don't store black moves in database)
                        if current_pgn:
                            current_pgn = f"{current_pgn} {san_move}"
                        else:
                            current_pgn = san_move
                        
                        board.push(move)
                    
                    moves_played += 1
                    
                except Exception as e:
                    print(f"Error during engine play: {e}")
                    break
            
            # Clean up black engine
            try:
                black_engine.__del__()
            except:
                pass
        
        return records_to_insert
    
    # Process trajectories in parallel
    all_records = []
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        future_to_trajectory = {
            executor.submit(play_trajectory_with_engines, trajectory, white_engine_params, black_engine_progression): idx 
            for idx, trajectory in enumerate(opening_trajectories)
        }
        
        for future in tqdm(concurrent.futures.as_completed(future_to_trajectory), total=len(future_to_trajectory)):
            trajectory_idx = future_to_trajectory[future]
            try:
                records = future.result()
                all_records.extend(records)
            except Exception as exc:
                print(f'Trajectory {trajectory_idx} generated an exception: {exc}')
    
    # Bulk insert all records
    if all_records:
        print(f"Inserting {len(all_records)} new white moves...")
        with Session(engine) as session:
            # Insert in batches to avoid memory issues
            batch_size = 1000
            for i in range(0, len(all_records), batch_size):
                batch = all_records[i:i + batch_size]
                session.execute(insert(Game), batch)
                session.commit()
        
        print(f"Successfully inserted {len(all_records)} new white moves")
    else:
        print("No new records to insert")

def generate_grpo_games(model, tokenizer, env, num_games: int = 100, max_moves_per_game: int = 50) -> List[Dict]:
    """
    Generate training data by having the model play games against Stockfish.
    Returns list of training examples with prompts and completions (no rewards).
    """
    game_data = []
    
    for game_idx in tqdm(range(num_games), desc="Generating GRPO games"):
        board = chess.Board()
        move_count = 0
        
        while not board.is_game_over() and move_count < max_moves_per_game:
            # Determine if it's the model's turn (let's say model plays white)
            if board.turn:  # White's turn (model)
                # Create prompt for current position
                move_history = []
                temp_board = chess.Board()
                for move in board.move_stack:
                    san_move = temp_board.san(move)
                    move_history.append(san_move)
                    temp_board.push(move)
                
                position_input = {
                    "moveHistory": move_history,
                    "possibleMoves": [board.san(move) for move in board.legal_moves],
                    "color": "w"
                }
                
                prompt = f"[INST] {system_prompt}\n\n{json.dumps(position_input)} [/INST]"
                
                # Generate model response
                inputs = tokenizer(prompt, return_tensors="pt")
                inputs = {k: v.to(model.device) for k, v in inputs.items()}
                
                with torch.no_grad():
                    outputs = model.generate(
                        inputs["input_ids"],
                        max_new_tokens=40,
                        pad_token_id=tokenizer.eos_token_id
                    )
                
                # Decode response
                response = tokenizer.decode(outputs[0], skip_special_tokens=True)
                try:
                    model_response = response.split("[/INST]")[1].strip()
                except Exception as e:
                    print('problem parsing the response from the llm: ',)
                    print('the response: ',response)
                    print(e)
                
                # Store the training example (prompt and completion only)
                game_data.append({
                    "prompt": prompt,
                    "completion": model_response,
                    "board_state": board.fen(), 
                    "game_id": game_idx,
                    "move_number": move_count + 1
                })
                
                # Try to make the move if it's legal
                try:
                    move_str, reasoning, is_valid_json = env.parse_model_output(model_response)
                    if is_valid_json and move_str:
                        legal_moves_san = [board.san(move) for move in board.legal_moves]
                        if move_str in legal_moves_san:
                            move = board.parse_san(move_str)
                            board.push(move)
                        else:
                            # Illegal move, make a random legal move to continue
                            random_move = np.random.choice(list(board.legal_moves))
                            board.push(random_move)
                    else:
                        # Invalid JSON, make a random legal move to continue
                        random_move = np.random.choice(list(board.legal_moves))
                        board.push(random_move)
                except Exception as e:
                    # Any error, make a random legal move
                    random_move = np.random.choice(list(board.legal_moves))
                    board.push(random_move)
                
            else:  # Black's turn (Stockfish)
                stockfish = env.get_stockfish()
                stockfish.set_fen_position(board.fen())
                best_move_uci = stockfish.get_best_move()
                
                if best_move_uci:
                    move = chess.Move.from_uci(best_move_uci)
                    board.push(move)
                else:
                    # Stockfish couldn't find a move, game over
                    break
            
            move_count += 1
    
    return game_data

def progressive_stockfish_training(model_path: str, output_dir: str, iterations: int = 10):
    """
    Progressive training against increasingly difficult Stockfish opponents
    """
    from model.learn import ChessGRPOTrainer
    
    # Stockfish skill progression
    skill_levels = [3, 6, 9, 11, 14, 17, 20]
    
    current_model_path = model_path
    
    for i, skill_level in enumerate(skill_levels):
        if i >= iterations:
            break
            
        print(f"Training iteration {i+1}: Stockfish skill level {skill_level}")
        
        trainer = ChessGRPOTrainer(
            model_name=current_model_path,
            output_dir=f"{output_dir}/skill_{skill_level}",
            stockfish_skill_level=skill_level
        )
        
        # Train for a few iterations at this skill level
        trainer.train(num_iterations=3, games_per_iteration=50)
        
        # Update model path for next iteration
        # After GRPO training, the model is saved as a regular model, not PEFT
        current_model_path = f"{output_dir}/skill_{skill_level}/final"
        
        print(f"Completed training against skill level {skill_level}")
    
    print("Progressive training completed!")

def seed_db():
    for f in tqdm([v for v in os.listdir('datagen/openings') if v.endswith('.tsv')]):
        df = pd.read_csv(f'datagen/openings/{f}', sep='\t')
        print(f"Original dataframe shape: {df.shape}")
        
        # Deduplicate the dataframe by keeping the row with the longest PGN for each opening name
        df['pgn_length'] = df['pgn'].str.len()
        df = df.sort_values('pgn_length', ascending=False)
        df = df.drop_duplicates(subset=['name'], keep='first')
        df = df.drop(columns=['pgn_length'])
        
        print(f"Deduplicated dataframe shape: {df.shape}")
        print(df.head())
        
        # Load openings into database using parallel processing
        load_openings_parallel(df, max_workers=8, batch_size=100)

        
if __name__ == "__main__":
    # Example of running progressive training
    # progressive_stockfish_training(
    #     model_path="./open_llama_7b-lora-final",
    #     output_dir="./progressive_chess_models",
    #     iterations=7
    # )
    
    #seed_db()
    stockfish_skill_elo_map = {
    850: {"Skill Level": 3,"Threads": 2,"Depth": 1},
    950: {"Skill Level": 6, "Threads": 2,"Depth": 2},
    1050: {"Skill Level": 9, "Threads": 2,"Depth": 3},
    1250: {"Skill Level": 11, "Threads": 2,"Depth": 4},
    1700: {"Skill Level": 14, "Threads": 2,"Depth": 6},
    1900: {"Skill Level": 17, "Threads": 1,"Depth": 8},
    2000: {"Skill Level": 20, "Threads": 1,"Depth": 10}
}
    #bongcloud opening trajectories
    opening_trajectories = [
        ['e4','e5','Ke2'],
        ['e4','e6','Ke2'],
        ['e4','d5','Ke2'],
        ['e4','d6','Ke2'],
        ['e4','c5','Ke2'],
        ['e4','c6','Ke2'],
        ['e4','g6','Ke2'],
        ['e4','g5','Ke2'],
        ['e4','f6','Ke2'],
        ['e4','f5','Ke2'],
        ['e4','b6','Ke2'],
        ['e4','b5','Ke2'],
        ['e4','a6','Ke2'],
        ['e4','a5','Ke2'],
        ['e4','h6','Ke2'],
        ['e4','h5','Ke2'],
        ['e4','Nf6','Ke2'],
        ['e4','Nc6','Ke2'],
    ]
    prune_openings(opening_trajectories, {"Skill Level": 20, "Threads": 1, "Depth": 18}, stockfish_skill_elo_map.values())
