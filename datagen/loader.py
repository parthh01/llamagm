import os
import json
import chess
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from datasets import Dataset, IterableDataset, load_dataset
import pandas as pd
from transformers import AutoTokenizer
from tqdm import tqdm
from constants import system_prompt

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
    eval_info = f"eval: {row['value']}"
    
    # Calculate tokens for the JSON structure and move
    completion_template = json.dumps({
        "move": row['move'],
        "reasoning": ""
    })
    base_token_count = len(tokenizer.encode(completion_template))
    max_tokens = 40
    available_tokens = max_tokens - base_token_count
    
    # First ensure we can fit the eval_info
    eval_tokens = len(tokenizer.encode(eval_info))
    if eval_tokens > available_tokens:
        # Truncate eval_info if it's too long
        while eval_tokens > available_tokens:
            eval_info = eval_info[:-5]  # Remove 5 chars at a time
            eval_tokens = len(tokenizer.encode(eval_info))
        return eval_info
    
    # If there's no description, return just the eval info
    if not row['description']:
        return eval_info
    
    # Calculate remaining tokens for description
    remaining_tokens = available_tokens - eval_tokens
    
    # Try to add description words while keeping under token limit
    words = row['description'].split()
    description = ""
    
    for word in words:
        test_description = f"{description} {word}".strip()
        test_tokens = len(tokenizer.encode(test_description))
        
        # Check if adding this word would exceed our remaining tokens
        if test_tokens > remaining_tokens:
            break
            
        description = test_description
    
    # Combine description and eval info
    if description:
        reasoning = f"{description} {eval_info}"
    else:
        reasoning = eval_info
    
    # Final check to ensure we're under the token limit
    full_completion = json.dumps({
        "move": row['move'],
        "reasoning": reasoning
    })
    
    if len(tokenizer.encode(full_completion)) > max_tokens:
        # If somehow we're still over the limit, return just the eval_info
        return eval_info
    
    return reasoning

def process_batch(batch_df, tokenizer):
    """Process a batch of rows from the database into the required format"""
    prompts = []
    completions = []
    
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
                
            # Create the prompt and completion strings
            input_dict = {
                "moveHistory": move_history,
                "possibleMoves": possible_moves,
                "color": get_color(board)
            }
            
            # Format as prompt
            prompt = f"[INST] {system_prompt}\n\n{json.dumps(input_dict)} [/INST]"
            
            # Format completion as a JSON object with move and reasoning
            reasoning = generate_reasoning(row, tokenizer)
            completion = json.dumps({
                "move": row['move'],
                "reasoning": reasoning
            })
            
            # Add to the lists
            prompts.append(prompt)
            completions.append(completion)
            
        except Exception as e:
            print(f"Error processing row: {e}")
            continue
    
    return {"prompt": prompts, "completion": completions}

def stream_dataset(engine, tokenizer, batch_size=1000,prompt_completion=False):
    """Stream the dataset in batches and yield processed examples"""
    total_rows = get_total_rows(engine)
    offset = 0
    
    # Continue fetching batches until we've processed all rows
    while offset < total_rows:
        query = text(f"SELECT * FROM games ORDER BY id LIMIT {batch_size} OFFSET {offset}")
        
        with engine.connect() as conn:
            batch_df = pd.read_sql(query, conn)
        
        # If no rows returned, break the loop
        if len(batch_df) == 0:
            break
            
        # Process this batch
        batch_data = process_batch(batch_df, tokenizer)
        
        # Increment offset by number of rows returned
        offset += len(batch_df)
        
        # Yield the batch data as a single text field
        for prompt, completion in zip(batch_data["prompt"], batch_data["completion"]):
            # Combine prompt and completion into a single text field
            combined_text = f"{prompt} {completion}"
            yield {"text": combined_text} if prompt_completion else {"prompt": prompt, "completion": completion}

def create_dataset(database_url, tokenizer, batch_size=1000, push_to_hub=False, hub_name=None,prompt_completion=False):
    """Create a Hugging Face dataset using a single text field format"""
    # Get total rows count
    engine = create_engine(database_url)
    total_rows = get_total_rows(engine)
    def generator_fn(url):
        # Create a new engine inside the generator to avoid pickling issues
        local_engine = create_engine(url)
        yield from stream_dataset(local_engine, tokenizer, batch_size,prompt_completion)
    
    # Create an IterableDataset using the streaming generator
    dataset = IterableDataset.from_generator(lambda: generator_fn(str(database_url)))
    
    # Optionally push to Hugging Face Hub
    if push_to_hub and hub_name:
        dataset.push_to_hub(hub_name)
    
    print(f"Dataset created successfully with {total_rows} total examples")
    # Get a sample example to display
    sample_example = next(iter(dataset))
    print(f"Sample example: {sample_example}")
    
    return dataset, total_rows

def create_full_dataset_in_memory(database_url, tokenizer, push_to_hub=False, hub_name=None, prompt_completion=False):
    """Load all data into memory, process it, and create a Hugging Face dataset"""
    engine = create_engine(database_url)
    
    print("Loading all data from database into memory...")
    
    # Load all data at once
    query = text("SELECT * FROM games ORDER BY id")
    with engine.connect() as conn:
        all_data_df = pd.read_sql(query, conn)
    
    total_rows = len(all_data_df)
    print(f"Loaded {total_rows} rows from database")
    
    print("Processing all data...")
    
    # Process all data at once
    processed_data = process_batch(all_data_df, tokenizer)
    
    # Create the dataset format
    dataset_data = []
    for prompt, completion in zip(processed_data["prompt"], processed_data["completion"]):
        if prompt_completion:
            # Combine prompt and completion into a single text field
            combined_text = f"{prompt} {completion}"
            dataset_data.append({"text": combined_text})
        else:
            dataset_data.append({"prompt": prompt, "completion": completion})
    
    print(f"Successfully processed {len(dataset_data)} examples")
    
    # Create Hugging Face dataset from the processed data
    dataset = Dataset.from_list(dataset_data)
    
    # Optionally push to Hugging Face Hub
    if push_to_hub and hub_name:
        print(f"Pushing dataset to Hugging Face Hub: {hub_name}")
        dataset.push_to_hub(hub_name)
        print("Dataset successfully uploaded to Hugging Face Hub")
    
    print(f"Dataset created successfully with {len(dataset_data)} examples")
    
    # Display a sample
    if len(dataset_data) > 0:
        print(f"Sample example: {dataset_data[0]}")
    
    return dataset, len(dataset_data)

if __name__ == "__main__":
    # Example usage
    load_dotenv()
    DATABASE_URL = f"postgresql://{os.getenv('PGUSER')}:{os.getenv('PGPASSWORD')}@{os.getenv('PGHOST')}:{os.getenv('PGPORT')}/{os.getenv('PGDATABASE')}?sslmode=require"
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
    dataset, total_rows = create_full_dataset_in_memory(DATABASE_URL,tokenizer,push_to_hub=True,hub_name="llamagm-bongcloud")
    # dataset, total_rows = create_dataset(
    #     database_url=DATABASE_URL,
    #     tokenizer=tokenizer,
    #     batch_size=32,
    #     push_to_hub=False,
    #     hub_name=None  # "your-username/chess-moves-dataset"
    # )
    
    # # Print samples at specific intervals to check for corruption
    # check_intervals = [0, 3500, 6500, 10500]  # Beginning and suspected corruption points
    
    # print(f"\n=== Checking dataset samples at intervals: {check_intervals} ===")
    
    # for interval in check_intervals:
    #     if interval >= total_rows:
    #         print(f"\nInterval {interval} exceeds total rows ({total_rows}), skipping...")
    #         continue
            
    #     print(f"\n--- Samples around step {interval} ---")
        
    #     # Skip to the desired position and take 5 samples
    #     dataset_iter = iter(dataset)
        
    #     # Skip to the interval position
    #     for _ in range(interval):
    #         try:
    #             next(dataset_iter)
    #         except StopIteration:
    #             print(f"Reached end of dataset before step {interval}")
    #             break
        
    #     # Print 5 samples from this position
    #     for i in range(5):
    #         try:
    #             sample = next(dataset_iter)
    #             print(f"Sample {interval + i}: {sample}")
    #         except StopIteration:
    #             print(f"Reached end of dataset at sample {interval + i}")
    #             break

