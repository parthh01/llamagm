import os 
import uuid
from dotenv import load_dotenv
from sqlalchemy import create_engine, insert, ForeignKey, Column, String, Integer, UUID, Float
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import Session
from stockfish import Stockfish
import chess 
import pandas as pd 
from tqdm import tqdm
import concurrent.futures
import threading
from functools import partial
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
        
        # Query for existing positions
        from sqlalchemy import or_
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

if __name__ == "__main__":
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