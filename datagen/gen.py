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
    
    def play_trajectory_with_engines(trajectory, white_params, black_params_list, 
                                   max_exploration_moves=8, eval_threshold=150, max_branches=2):
        """Play out a single trajectory with adaptive branching exploration"""
        trajectory_str = " ".join(trajectory)
        print(f"\n=== Starting trajectory: {trajectory_str} ===")
        
        def play_single_game_with_branching(black_params):
            """Play a single game against one black engine configuration with adaptive branching"""
            black_skill = black_params.get("Skill Level", "Unknown")
            print(f"  Playing against black skill level: {black_skill}")
            
            # Get thread-local engines - reuse existing instances
            white_engine = get_stockfish()
            
            # Create black engine with proper initialization
            black_engine = Stockfish(
                path=os.getenv('STOCKFISH_PATH', '/usr/local/bin/stockfish'),
                parameters={"Threads": black_params.get("Threads", 1), "Hash": 32}
            )
            
            # Configure engines
            white_engine.set_skill_level(white_params.get("Skill Level", 20))
            white_engine.set_depth(white_params.get("Depth", 18))
            
            black_engine.set_skill_level(black_params.get("Skill Level", 20))
            black_engine.set_depth(black_params.get("Depth", 18))
            
            def explore_position(board, current_pgn, moves_played, depth=0, path_id=""):
                """Recursively explore positions with adaptive branching"""
                print(f"    Exploring position: moves={moves_played}, depth={depth}, path={path_id}, turn={'White' if board.turn else 'Black'}")
                print(f"      Current PGN: {current_pgn}")
                
                # Safety checks
                if moves_played >= max_moves_per_trajectory:
                    print(f"      Stopping: reached max moves ({max_moves_per_trajectory})")
                    return []
                
                if board.is_game_over():
                    print(f"      Stopping: game over")
                    return []
                
                if depth > 2:  # Strict depth limit to prevent explosion
                    print(f"      Stopping: max depth reached ({depth})")
                    return []
                
                records_to_insert = []
                
                if board.turn == chess.WHITE:
                    # White's turn - always play best move, no branching
                    print(f"      White's turn - finding best move...")
                    try:
                        white_engine.set_fen_position(board.fen())
                        best_move_uci = white_engine.get_best_move()
                        
                        if best_move_uci is None:
                            print(f"      White: No best move found")
                            return records_to_insert
                        
                        move = chess.Move.from_uci(best_move_uci)
                        san_move = board.san(move)
                        print(f"      White plays: {san_move}")
                        
                        # Evaluate position before making the move
                        evaluation = white_engine.get_evaluation()
                        value = f"M{evaluation['value']}" if evaluation['type'] == 'mate' else f"{evaluation['value']}"
                        
                        # Create record for white move
                        records_to_insert.append({
                            'id': uuid.uuid4(),
                            'source': f"Generated_Trajectory_Black_Skill_{black_params.get('Skill Level', 'Unknown')}_Branch_{depth}",
                            'pgn': current_pgn,
                            'move': san_move,
                            'engine': 'stockfish',
                            'engine_rating': white_engine.get_parameters().get('UCI_Elo', 3200),
                            'value': value,
                            'category': 'midgame'
                        })
                        
                        # Update PGN and continue
                        new_pgn = f"{current_pgn} {san_move}" if current_pgn else san_move
                        new_board = board.copy()
                        new_board.push(move)
                        
                        # Continue exploration
                        child_records = explore_position(new_board, new_pgn, moves_played + 1, depth, f"{path_id}W")
                        records_to_insert.extend(child_records)
                        print(f"      White move complete, generated {len(child_records)} child records")
                    
                    except Exception as e:
                        print(f"      Error in white move evaluation: {e}")
                        return records_to_insert
                    
                else:
                    # Black's turn - decide whether to branch
                    print(f"      Black's turn - checking if should branch...")
                    try:
                        black_engine.set_fen_position(board.fen())
                        
                        # Check if we should branch - much more restrictive
                        should_branch = (
                            moves_played < max_exploration_moves and 
                            depth < 2 and  # Reduced from 3
                            moves_played > 2  # Only branch after a few moves
                        )
                        
                        print(f"      Should branch: {should_branch} (moves={moves_played}, depth={depth})")
                        
                        if should_branch:
                            print(f"      Getting top moves for branching...")
                            # Use get_top_moves with timeout protection
                            try:
                                # Set a shorter depth for branching analysis
                                original_depth = black_engine.get_parameters().get("Depth", 8)
                                black_engine.set_depth(min(6, original_depth))  # Limit depth for faster analysis
                                
                                top_moves = black_engine.get_top_moves(max_branches + 1)
                                
                                # Restore original depth
                                black_engine.set_depth(original_depth)
                                
                                print(f"      Found {len(top_moves)} top moves")
                                
                                if len(top_moves) > 1:
                                    # Filter moves within threshold
                                    best_eval = top_moves[0]['Centipawn'] if 'Centipawn' in top_moves[0] else 0
                                    good_moves = []
                                    
                                    for i, move_info in enumerate(top_moves[:max_branches]):
                                        move_eval = move_info.get('Centipawn', 0)
                                        eval_diff = abs(move_eval - best_eval)
                                        print(f"        Move {i}: {move_info['Move']}, eval={move_eval}, diff={eval_diff}")
                                        
                                        if eval_diff <= eval_threshold:
                                            try:
                                                move = chess.Move.from_uci(move_info['Move'])
                                                good_moves.append(move)
                                                print(f"          Added to good moves")
                                            except ValueError as e:
                                                print(f"          Invalid move: {e}")
                                                continue
                                    
                                    # Branch on multiple moves if we have them
                                    if len(good_moves) > 1:
                                        print(f"      Branching on {len(good_moves)} moves")
                                        for i, move in enumerate(good_moves):
                                            san_move = board.san(move)
                                            new_pgn = f"{current_pgn} {san_move}" if current_pgn else san_move
                                            new_board = board.copy()
                                            new_board.push(move)
                                            
                                            print(f"        Branch {i}: {san_move}")
                                            
                                            # Continue exploration with increased depth
                                            branch_records = explore_position(new_board, new_pgn, moves_played + 1, depth + 1, f"{path_id}B{i}")
                                            records_to_insert.extend(branch_records)
                                            print(f"        Branch {i} complete, generated {len(branch_records)} records")
                                    else:
                                        print(f"      Not enough good moves for branching, falling back to single move")
                                        should_branch = False
                                else:
                                    print(f"      Not enough top moves for branching")
                                    should_branch = False
                            except Exception as e:
                                print(f"      Error getting top moves: {e}")
                                should_branch = False
                        
                        if not should_branch:
                            # Single best move
                            print(f"      Playing single best move...")
                            best_move_uci = black_engine.get_best_move()
                            if best_move_uci is None:
                                print(f"      Black: No best move found")
                                return records_to_insert
                            
                            move = chess.Move.from_uci(best_move_uci)
                            san_move = board.san(move)
                            print(f"      Black plays: {san_move}")
                            
                            new_pgn = f"{current_pgn} {san_move}" if current_pgn else san_move
                            new_board = board.copy()
                            new_board.push(move)
                            
                            # Continue exploration
                            child_records = explore_position(new_board, new_pgn, moves_played + 1, depth, f"{path_id}B")
                            records_to_insert.extend(child_records)
                            print(f"      Black single move complete, generated {len(child_records)} child records")
                    
                    except Exception as e:
                        print(f"      Error in black move evaluation: {e}")
                        return records_to_insert
                
                print(f"    Position exploration complete: {len(records_to_insert)} total records")
                return records_to_insert
            
            try:
                # Initialize board and play opening moves
                board = chess.Board()
                current_pgn = ""
                
                print(f"  Playing opening moves...")
                # Play the opening trajectory
                for i, move_text in enumerate(trajectory):
                    try:
                        move = board.parse_san(move_text)
                        san_move = board.san(move)
                        
                        # Update PGN
                        if current_pgn:
                            current_pgn = f"{current_pgn} {san_move}"
                        else:
                            current_pgn = san_move
                        
                        board.push(move)
                        print(f"    Opening move {i+1}: {san_move}")
                    except ValueError as e:
                        print(f"    Error parsing opening move {move_text}: {e}")
                        break
                
                print(f"  Opening complete. Starting exploration from: {current_pgn}")
                # Start exploration from the position after opening
                records = explore_position(board, current_pgn, len(trajectory), 0, "root")
                
                print(f"  Game complete against skill {black_skill}: {len(records)} records generated")
                return records
            
            finally:
                # Properly clean up black engine
                try:
                    if hasattr(black_engine, '_stockfish') and black_engine._stockfish:
                        black_engine._stockfish.kill()
                        black_engine._stockfish.wait()
                        print(f"  Cleaned up black engine for skill {black_skill}")
                except Exception as e:
                    print(f"  Error cleaning up black engine: {e}")
        
        # Run games sequentially for each black engine configuration
        all_records = []
        
        for i, black_params in enumerate(black_params_list):
            print(f"  Starting game {i+1}/{len(black_params_list)} for trajectory: {trajectory_str}")
            try:
                records = play_single_game_with_branching(black_params)
                all_records.extend(records)
                print(f"  Game {i+1} complete: {len(records)} records")
            except Exception as exc:
                print(f'  Game against black skill {black_params.get("Skill Level", "Unknown")} generated an exception: {exc}')
        
        print(f"=== Trajectory complete: {trajectory_str} - {len(all_records)} total records ===\n")
        return all_records
    
    # Process trajectories in parallel with reduced workers
    all_records = []
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:  # Reduced from 4
        future_to_trajectory = {
            executor.submit(play_trajectory_with_engines, trajectory, white_engine_params, black_engine_progression): idx 
            for idx, trajectory in enumerate(opening_trajectories)
        }
        
        for future in tqdm(concurrent.futures.as_completed(future_to_trajectory), total=len(future_to_trajectory)):
            trajectory_idx = future_to_trajectory[future]
            try:
                records = future.result()
                all_records.extend(records)
                print(f"MAIN: Trajectory {trajectory_idx} completed with {len(records)} records")
            except Exception as exc:
                print(f'MAIN: Trajectory {trajectory_idx} generated an exception: {exc}')
    
    # Bulk insert all records
    if all_records:
        print(f"MAIN: Inserting {len(all_records)} new white moves...")
        with Session(engine) as session:
            # Insert in batches to avoid memory issues
            batch_size = 1000
            for i in range(0, len(all_records), batch_size):
                batch = all_records[i:i + batch_size]
                session.execute(insert(Game), batch)
                session.commit()
                print(f"MAIN: Inserted batch {i//batch_size + 1}")
        
        print(f"MAIN: Successfully inserted {len(all_records)} new white moves")
    else:
        print("MAIN: No new records to insert")

def generate_grpo_games(model, tokenizer, env, num_games: int = 100, max_moves_per_game: int = 50) -> List[Dict]:
    """
    Generate training data by having the model play games against Stockfish.
    Returns list of training examples with prompts and completions (no rewards).
    """
    game_data = []
    
    # Put model in eval mode during generation
    model.eval()
    
    # Process games in smaller batches to manage memory
    batch_size = 10  # Process 10 games at a time
    
    for batch_start in range(0, num_games, batch_size):
        batch_end = min(batch_start + batch_size, num_games)
        
        # Clear cache at start of each batch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        for game_idx in tqdm(range(batch_start, batch_end), desc=f"Generating games {batch_start}-{batch_end}"):
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
                    
                    # Generate model response with memory optimizations
                    inputs = tokenizer(prompt, return_tensors="pt")
                    inputs = {k: v.to(model.device) for k, v in inputs.items()}
                    
                    with torch.no_grad():
                        # Use torch.cuda.amp for memory efficiency
                        with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                            outputs = model.generate(
                                inputs["input_ids"],
                                max_new_tokens=40,
                                pad_token_id=tokenizer.eos_token_id,
                                do_sample=True,  # Keep sampling for diversity
                                temperature=0.7,
                                top_p=0.9
                            )
                    
                    # Decode response
                    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
                    try:
                        model_response = response.split("[/INST]")[1].strip()
                    except Exception as e:
                        print('problem parsing the response from the llm: ')
                        print('the response: ', response)
                        print(e)
                        model_response = ""
                    
                    # Store the training example
                    game_data.append({
                        "prompt": prompt,
                        "completion": model_response,
                        "board_state": board.fen(), 
                        "game_id": game_idx,
                        "move_number": move_count + 1
                    })
                    
                    # Clean up tensors immediately
                    del inputs, outputs
                    
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
        
        # Clear cache after each batch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # Put model back in train mode
    model.train()
    
    return game_data

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

def extend_black_openings(white_engine_progression):
    """
    Query the database for all black openings and play them out with a white engine progression
    """
    print("Querying database for black opening positions...")
    
    # Query all opening positions and find the last position for each ECO opening
    with Session(engine) as session:
        # Get all opening games
        opening_games = session.query(Game).filter(Game.category == 'opening').all()
        
        # Group by source (ECO code) and find the position with longest PGN for each
        eco_positions = {}
        for game in opening_games:
            eco_code = game.source
            pgn_length = len(game.pgn.split()) if game.pgn else 0
            
            # Only consider positions where it's black to move (odd number of moves)
            if pgn_length % 2 == 1:
                if eco_code not in eco_positions or pgn_length > len(eco_positions[eco_code]['pgn'].split()):
                    eco_positions[eco_code] = {
                        'pgn': game.pgn,
                        'description': game.description,
                        'source': game.source
                    }
    
    print(f"Found {len(eco_positions)} unique ECO openings with black to move positions")
    
    # Convert to list of trajectories for processing
    opening_trajectories = []
    for eco_code, position_data in eco_positions.items():
        if position_data['pgn']:
            moves = position_data['pgn'].split()
            opening_trajectories.append({
                'moves': moves,
                'eco_code': eco_code,
                'description': position_data['description']
            })
        else:
            # Starting position
            opening_trajectories.append({
                'moves': [],
                'eco_code': eco_code,
                'description': position_data['description']
            })
    
    print(f"Processing {len(opening_trajectories)} opening trajectories...")
    
    def play_black_opening_trajectory(trajectory_data, white_params_list, max_moves_per_trajectory=50):
        """Play out a single black opening trajectory with white engine progression"""
        moves = trajectory_data['moves']
        eco_code = trajectory_data['eco_code']
        description = trajectory_data['description']
        
        trajectory_str = " ".join(moves) if moves else "Starting position"
        print(f"\n=== Processing ECO {eco_code}: {description} ===")
        print(f"Starting from: {trajectory_str}")
        
        def play_single_game_with_white_engine(white_params):
            """Play a single game with one white engine configuration"""
            white_skill = white_params.get("Skill Level", "Unknown")
            print(f"  Playing with white skill level: {white_skill}")
            
            # Get thread-local black engine (weaker, consistent opponent)
            black_engine = get_stockfish()
            
            # Create white engine with progression parameters
            white_engine = Stockfish(
                path=os.getenv('STOCKFISH_PATH', '/usr/local/bin/stockfish'),
                parameters={"Threads": white_params.get("Threads", 1), "Hash": 32}
            )
            
            
            white_engine.set_skill_level(white_params.get("Skill Level", 20))
            white_engine.set_depth(white_params.get("Depth", 18))
            
            try:
                # Initialize board and play opening moves
                board = chess.Board()
                current_pgn = ""
                
                # Play the opening trajectory
                for i, move_text in enumerate(moves):
                    try:
                        move = board.parse_san(move_text)
                        san_move = board.san(move)
                        
                        # Update PGN
                        if current_pgn:
                            current_pgn = f"{current_pgn} {san_move}"
                        else:
                            current_pgn = san_move
                        
                        board.push(move)
                        print(f"    Opening move {i+1}: {san_move}")
                    except ValueError as e:
                        print(f"    Error parsing opening move {move_text}: {e}")
                        return []
                
                print(f"  Opening complete. Continuing from: {current_pgn}")
                
                # Continue playing from this position
                records_to_insert = []
                moves_played = len(moves)
                
                while not board.is_game_over() and moves_played < max_moves_per_trajectory:
                    if board.turn == chess.BLACK:
                        # Black's turn (consistent opponent)
                        try:
                            black_engine.set_fen_position(board.fen())
                            best_move_uci = black_engine.get_best_move()
                            if best_move_uci is None:
                                print(f"      Black: No best move found")
                                break
                            
                            move = chess.Move.from_uci(best_move_uci)
                            san_move = board.san(move)

                            # Update PGN and board
                            new_pgn = f"{current_pgn} {san_move}" if current_pgn else san_move
                            board.push(move)
                            black_engine.set_fen_position(board.fen())
                            value = black_engine.get_evaluation()['value']
                            records_to_insert.append({
                                'id': uuid.uuid4(),
                                'source': f"Extended_{eco_code}_Black_Skill",
                                'pgn': current_pgn,
                                'move': san_move,
                                'engine': 'stockfish',
                                'engine_rating': white_engine.get_parameters().get('UCI_Elo', 3200),
                                'value': value,
                                'category': 'midgame'
                            }) 
                            current_pgn = new_pgn
                            moves_played += 1

                            
                        except Exception as e:
                            print(f"      Error in black move: {e}")
                            break

                    else:
                        # White's turn (engine progression)
                        try:
                            white_engine.set_fen_position(board.fen())
                            
                            # Evaluate position before making the move
                            evaluation = white_engine.get_evaluation()
                            value = f"M{evaluation['value']}" if evaluation['type'] == 'mate' else f"{evaluation['value']}"
                            
                            best_move_uci = white_engine.get_best_move()
                            
                            if best_move_uci is None:
                                print(f"      White: No best move found")
                                break
                            
                            move = chess.Move.from_uci(best_move_uci)
                            san_move = board.san(move)
                            

                            
                            # Update PGN and board
                            new_pgn = f"{current_pgn} {san_move}" if current_pgn else san_move
                            board.push(move)
                            current_pgn = new_pgn
                            moves_played += 1
                            
                        except Exception as e:
                            print(f"      Error in white move: {e}")
                            break
                
                print(f"  Game complete: {len(records_to_insert)} white moves generated")
                return records_to_insert
            
            finally:
                # Clean up white engine
                try:
                    if hasattr(white_engine, '_stockfish') and white_engine._stockfish:
                        white_engine._stockfish.kill()
                        white_engine._stockfish.wait()
                except Exception as e:
                    print(f"  Error cleaning up white engine: {e}")
        
        # Run games for each white engine configuration
        all_records = []
        
        for i, white_params in enumerate(white_params_list):
            print(f"  Starting game {i+1}/{len(white_params_list)} for ECO {eco_code}")
            try:
                records = play_single_game_with_white_engine(white_params)
                all_records.extend(records)
                print(f"  Game {i+1} complete: {len(records)} records")
            except Exception as exc:
                print(f'  Game with white skill {white_params.get("Skill Level", "Unknown")} generated an exception: {exc}')
        
        print(f"=== ECO {eco_code} complete: {len(all_records)} total records ===\n")
        return all_records
    
    # Process trajectories in parallel
    all_records = []
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        future_to_trajectory = {
            executor.submit(play_black_opening_trajectory, trajectory_data, white_engine_progression): idx 
            for idx, trajectory_data in enumerate(opening_trajectories)
        }
        
        for future in tqdm(concurrent.futures.as_completed(future_to_trajectory), total=len(future_to_trajectory)):
            trajectory_idx = future_to_trajectory[future]
            try:
                records = future.result()
                all_records.extend(records)
                print(f"MAIN: ECO trajectory {trajectory_idx} completed with {len(records)} records")
            except Exception as exc:
                print(f'MAIN: ECO trajectory {trajectory_idx} generated an exception: {exc}')
    
    # Bulk insert all records
    if all_records:
        print(f"MAIN: Inserting {len(all_records)} new white moves from black openings...")
        with Session(engine) as session:
            # Insert in batches to avoid memory issues
            batch_size = 1000
            for i in range(0, len(all_records), batch_size):
                batch = all_records[i:i + batch_size]
                session.execute(insert(Game), batch)
                session.commit()
                print(f"MAIN: Inserted batch {i//batch_size + 1}")
        
        print(f"MAIN: Successfully inserted {len(all_records)} new white moves from black openings")
    else:
        print("MAIN: No new records to insert")

if __name__ == "__main__":
    
    #seed_db()
    stockfish_skill_elo_map = {
    400: {"Skill Level": 1,"Threads": 1,"Depth": 1},
    850: {"Skill Level": 3,"Threads": 1,"Depth": 1},
    950: {"Skill Level": 6, "Threads": 1,"Depth": 2},
    1050: {"Skill Level": 9, "Threads": 1,"Depth": 3},
    1250: {"Skill Level": 11, "Threads": 1,"Depth": 4},
    1700: {"Skill Level": 14, "Threads": 1,"Depth": 6},
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
    #prune_openings(opening_trajectories, {"Skill Level": 20, "Threads": 1, "Depth": 18}, stockfish_skill_elo_map.values())
    extend_black_openings(stockfish_skill_elo_map.values())