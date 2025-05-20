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
load_dotenv()

DATABASE_URL = f"postgresql://{os.getenv('PGUSER')}:{os.getenv('PGPASSWORD')}@{os.getenv('PGHOST')}:{os.getenv('PGPORT')}/{os.getenv('PGDATABASE')}?sslmode=require"
engine = create_engine(DATABASE_URL, echo=True, pool_size=10, max_overflow=20, pool_timeout=5, pool_recycle=1800,pool_pre_ping=True)
Base = declarative_base()

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

def evaluate_game(fen_string,engine):
    engine.set_fen_position(fen_string)
    evaluation = engine.get_evaluation() 
    value = f"M{evaluation['value']}" if evaluation['type'] == 'mate' else f"{evaluation['value']}"
    best_move = engine.get_best_move()
    move = chess.Move.from_uci(best_move)
    board = chess.Board(fen_string)
    return value, board.san(move)

def load_openings(df, chess_engine):
    
    for _, row in df.iterrows():
        opening_name = row['name']
        pgn_text = row['pgn']
        eco = row['eco']
        
        # Create a new board
        board = chess.Board()
        
        # Split the PGN into moves
        moves_text = pgn_text.split()
        current_pgn = ""
        
        # Process each move
        for i, move_text in enumerate(moves_text):
            # Skip move numbers (like "1.", "2.", etc.)
            if '.' in move_text:
                continue
                
            # Parse the move
            try:
                move = board.parse_san(move_text)
                san_move = board.san(move)  # Get the standard algebraic notation
                # Get the current position before making the move
                fen_before_move = board.fen()
                # Evaluate the position
                value, best_move = evaluate_game(fen_before_move, chess_engine)
                # Create description
                description = f"{opening_name}"
                # Insert into database
                with Session(engine, expire_on_commit=False) as session:
                    new_game = Game(
                        id=uuid.uuid4(),
                        source=f"ECO {eco}",
                        pgn=current_pgn,
                        move=san_move,
                        engine='stockfish',
                        engine_rating=chess_engine.get_parameters()['UCI_Elo'],
                        value=value,
                        description=description,
                        category="opening"
                    )
                    session.add(new_game)
                    session.commit()
                
                # Update the current PGN
                if current_pgn:
                    current_pgn += f" {san_move}"
                else:
                    current_pgn = san_move
                
                # Make the move on the board
                board.push(move)
            

                

                    
                print(f"Added position after {san_move} in {opening_name}")
                
            except ValueError as e:
                print(f"Error parsing move {move_text} in {opening_name}: {e}")
                continue

if __name__ == "__main__":
    
    for f in tqdm([v for v in os.listdir('data/openings') if v.endswith('.tsv')]):
        df = pd.read_csv(f'data/openings/{f}', sep='\t')
        print(f"Original dataframe shape: {df.shape}")
        # Deduplicate the dataframe by keeping the row with the longest PGN for each opening name
        df['pgn_length'] = df['pgn'].str.len()
        df = df.sort_values('pgn_length', ascending=False)
        df = df.drop_duplicates(subset=['name'], keep='first')
        df = df.drop(columns=['pgn_length'])
        
        print(f"Deduplicated dataframe shape: {df.shape}")
        print(df.head())
        
        chess_engine = Stockfish(depth=18,path=os.getenv('STOCKFISH_PATH','/usr/local/bin/stockfish'),parameters={"Threads": 4, "Hash": 4096})
        chess_engine.set_elo_rating(4000)
        # Load openings into database
        load_openings(df, chess_engine)