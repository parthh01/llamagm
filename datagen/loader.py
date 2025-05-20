import os
import json
import chess
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from datasets import Dataset
import pandas as pd
from transformers import AutoTokenizer
from tqdm import tqdm

def get_total_rows(engine):
    """Get the total number of rows in the games table"""
    with engine.connect() as conn:
        result = conn.execute(text("SELECT COUNT(*) FROM games"))
        return result.scalar()

def get_move_history_from_pgn(pgn_text):
    """Convert PGN text to a list of moves"""
    if not pgn_text:
        return []
    return pgn_text.split()

def get_possible_moves(board):
    """Get a list of possible legal moves in SAN notation"""
    legal_moves = list(board.legal_moves)
    return [board.san(move) for move in legal_moves]

def get_color(board):
    """Return 'w' if it's white's turn, 'b' if it's black's turn"""
    return 'w' if board.turn else 'b'

def generate_reasoning(row, tokenizer):
    """Generate reasoning for the move based on evaluation and description"""
    # Add evaluation information (prioritize this)
    if row['value'].startswith('M'):
        mate_in = row['value'][1:]
        eval_info = f"Mate in {abs(int(mate_in))} for {'White' if int(mate_in) > 0 else 'Black'}."
    else:
        eval_value = int(row['value'])
        if abs(eval_value) < 50:
            eval_info = f"Position is roughly equal. Eval: {eval_value}"
        elif eval_value > 0:
            eval_info = f"White advantage: {eval_value}"
        else:
            eval_info = f"Black advantage: {abs(eval_value)}"
    
    # Start with just the evaluation info
    reasoning = eval_info
    
    # If there's no description, return just the eval info
    if not row['description']:
        return reasoning
    
    # Try to add description words while keeping under token limit
    words = row['description'].split()
    description = ""
    
    for word in words:
        test_text = f"{description} {word} {eval_info}".strip()
        # Count tokens in the test text
        token_count = len(tokenizer.encode(test_text))
        
        # If adding this word would exceed our limit, stop
        if token_count >= 40:
            break
            
        description += f"{word} "
    
    # Combine description and eval info if we have description
    if description:
        reasoning = f"{description.strip()} {eval_info}"
    
    return reasoning

def process_batch(batch_df, tokenizer):
    """Process a batch of rows from the database into the required format"""
    chess_data = []
    
    for _, row in batch_df.iterrows():
        try:
            # Create a chess board
            board = chess.Board()
            
            # Apply moves from PGN to get to the current position
            move_history = get_move_history_from_pgn(row['pgn'])
            for move in move_history:
                try:
                    board.push_san(move)
                except ValueError:
                    # Skip invalid moves
                    continue
            
            # Get possible moves
            possible_moves = get_possible_moves(board)
            
            # Skip if the actual move isn't in possible moves (shouldn't happen with valid data)
            if row['move'] not in possible_moves:
                continue
                
            # Create the input and output dictionaries
            input_dict = {
                "moveHistory": move_history,
                "possibleMoves": possible_moves,
                "color": get_color(board)
            }
            
            output_dict = {
                "move": row['move'],
                "reasoning": generate_reasoning(row, tokenizer)
            }
            
            # Add to the dataset
            chess_data.append({
                "input": json.dumps(input_dict),
                "output": json.dumps(output_dict)
            })
            
        except Exception as e:
            print(f"Error processing row: {e}")
            continue
    
    return chess_data

def stream_dataset(engine, tokenizer, batch_size=1000):
    """Stream the dataset in batches and yield processed examples"""
    total_rows = get_total_rows(engine)
    processed_rows = 0
    
    with tqdm(total=total_rows, desc="Processing rows") as pbar:
        # Stream data in batches
        for offset in range(0, total_rows, batch_size):
            query = text(f"SELECT * FROM games ORDER BY id LIMIT {batch_size} OFFSET {offset}")
            
            with engine.connect() as conn:
                batch_df = pd.read_sql(query, conn)
            
            # Process this batch
            chess_data_batch = process_batch(batch_df, tokenizer)
            
            # Update progress
            processed_rows += len(batch_df)
            pbar.update(len(batch_df))
            
            # Yield examples one by one
            for example in chess_data_batch:
                yield example

def create_dataset(engine, tokenizer, batch_size=1000, push_to_hub=False, hub_name=None):
    """Create a Hugging Face dataset using streaming"""
    # Create dataset using streaming to handle large data
    dataset = Dataset.from_generator(lambda: stream_dataset(engine, tokenizer, batch_size))
    
    # Optionally push to Hugging Face Hub
    if push_to_hub and hub_name:
        dataset.push_to_hub(hub_name)
    
    print(f"Dataset created with {len(dataset)} examples")
    print(f"Sample example: {dataset[0]}")
    
    return dataset

if __name__ == "__main__":
    # Example usage
    load_dotenv()
    DATABASE_URL = f"postgresql://{os.getenv('PGUSER')}:{os.getenv('PGPASSWORD')}@{os.getenv('PGHOST')}:{os.getenv('PGPORT')}/{os.getenv('PGDATABASE')}?sslmode=require"
    engine = create_engine(DATABASE_URL)
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    dataset = create_dataset(
        engine=engine,
        tokenizer=tokenizer,
        batch_size=32,
        push_to_hub=False,
        hub_name=None  # "your-username/chess-moves-dataset"
    )
    print(dataset[0])

