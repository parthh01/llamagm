import chess
import math
import time
from typing import List, Dict, Tuple, Optional
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
from eval.players import BasePlayer, RandomPlayer, StockfishPlayer,LLMPlayer, stockfish_skill_elo_map, IllegalMoveError
import concurrent.futures
import threading
from dotenv import load_dotenv 

load_dotenv()

class ChessGauntlet:
    def __init__(self, player: BasePlayer, games_per_level: int = 10, starting_elo: int = 800, num_threads: int = 4):
        """
        Initialize the chess gauntlet for evaluating a player.
        
        Args:
            player: The player to evaluate
            games_per_level: Number of games to play against each opponent
            starting_elo: Initial ELO rating for the player
            num_threads: Number of parallel threads to use for evaluation
        """
        self.player = player
        self.games_per_level = games_per_level
        self.player_elo = starting_elo
        self.elo_history = []
        self.invalid_moves = 0
        self.illegal_moves = 0
        self.total_moves = 0
        self.num_threads = min(num_threads, games_per_level)  # Can't use more threads than games
        self.elo_lock = threading.Lock()  # Lock for updating ELO safely
        self.player_lock = threading.Lock()  # Lock for accessing the player safely
        self.stats_lock = threading.Lock()  # Lock for updating statistics
        
    def expected_score(self, player_elo: int, opponent_elo: int) -> float:
        """Calculate expected score based on ELO difference."""
        return 1 / (1 + 10 ** ((opponent_elo - player_elo) / 400))
    
    def update_elo(self, player_elo: int, opponent_elo: int, score: float, k: int = 32) -> int:
        """Update player's ELO based on game result."""
        expected = self.expected_score(player_elo, opponent_elo)
        new_elo = player_elo + k * (score - expected)
        return round(new_elo)
    
    def play_game(self, opponent: BasePlayer, opponent_elo: int, player_is_white: bool) -> Tuple[float, int, int, int]:
        """
        Play a single game between the player and opponent.
        
        Returns:
            Tuple of (score, invalid_moves, illegal_moves, moves_made)
            Score: 1 for win, 0.5 for draw, 0 for loss
        """
        board = chess.Board()
        invalid_moves = 0
        illegal_moves = 0
        moves_made = 0
        time_limit = 10000  # 10 seconds per move
        move_timeout = 30  # 5 seconds timeout for move generation
        # Use longer timeout for Stockfish players
        stockfish_timeout = 15 if isinstance(opponent, StockfishPlayer) else move_timeout
        
        # Add ply counter to limit game length
        ply_count = 0
        max_plies = 50  # 25 moves per side
        
        while not board.is_game_over() and ply_count < max_plies:
            ply_count += 1
            current_player = self.player if (board.turn == chess.WHITE) == player_is_white else opponent
            
            try:
                if current_player == self.player:
                    moves_made += 1
                    try:
                        with self.player_lock:  # Acquire lock when accessing the player
                            try:
                                move = self.player.get_move(board, time_limit)
                            except Exception as e:
                                if isinstance(e, IllegalMoveError):
                                    print(f"illegal move suggested by player: {e} defaulting to random move")
                                    illegal_moves += 1
                                else:
                                    print(f"Error during move generation: {e} defaulting to random move")
                                    invalid_moves += 1
                                move = RandomPlayer().get_move(board, time_limit)
                        # Check if move is legal
                        if move not in board.legal_moves:
                            illegal_moves += 1
                            # If illegal move, make a random legal move instead
                            move = RandomPlayer().get_move(board, time_limit)
                    except Exception as e:
                        invalid_moves += 1
                        # If invalid move (parsing error), make a random legal move
                        print("error parsing move: ", e)
                        move = RandomPlayer().get_move(board, time_limit)
                else:
                    # Also add timeout for opponent moves
                    current_timeout = stockfish_timeout if isinstance(opponent, StockfishPlayer) else move_timeout
                    try:
                        with concurrent.futures.ThreadPoolExecutor() as executor:
                            future = executor.submit(current_player.get_move, board, time_limit)
                            try:
                                move = future.result(timeout=current_timeout)
                            except concurrent.futures.TimeoutError:
                                print(f"Opponent move generation timed out after {current_timeout} seconds (position: {board.fen()})")
                                # If timeout, make a random legal move instead
                                move = RandomPlayer().get_move(board, time_limit)
                    except Exception as e:
                        print(f"Error during opponent move generation: {e}")
                        move = RandomPlayer().get_move(board, time_limit)
                
                board.push(move)
                
                # Check for game end conditions
                if board.is_checkmate() or board.is_stalemate() or board.is_insufficient_material() or board.can_claim_draw():
                    break
                    
            except Exception as e:
                print(f"Error during game: {e}")
                break
        
        # Determine game result
        if ply_count >= max_plies:
            # Game reached ply limit, use Stockfish to evaluate position
            try:
                # Create a strong Stockfish instance for evaluation
                evaluator = StockfishPlayer({"Skill Level": 20, "Threads": 1,"Depth": 10})
                eval_score = evaluator.evaluate_position(board)
                
                # Positive score favors white, negative favors black
                if eval_score > 0:  # White is winning
                    score = 1.0 if player_is_white else 0.0
                elif eval_score < 0:  # Black is winning
                    score = 0.0 if player_is_white else 1.0
                else:  # Draw-ish position
                    score = 0.5
                
                print(f"Game reached {max_plies} ply limit. Stockfish eval: {eval_score}. Score: {score}")
            except Exception as e:
                print(f"Error evaluating final position: {e}. Defaulting to draw.")
                score = 0.5
        else:
            # Normal game end
            result = board.result()
            if result == "1-0":
                score = 1.0 if player_is_white else 0.0
            elif result == "0-1":
                score = 0.0 if player_is_white else 1.0
            else:  # Draw
                score = 0.5
            
        return score, invalid_moves, illegal_moves, moves_made
    
    def play_game_thread(self, args):
        """Wrapper for play_game to be used with ThreadPoolExecutor."""
        opponent, opponent_elo, player_is_white, current_elo = args
        score, invalid, illegal, moves = self.play_game(opponent, opponent_elo, player_is_white)
        
        # Thread-safe ELO update
        with self.elo_lock:
            new_elo = self.update_elo(current_elo, opponent_elo, score)
            self.elo_history.append(new_elo)
         
        return score, invalid, illegal, moves, new_elo
    
    def run_sanity_check(self) -> bool:
        """Run sanity check against RandomPlayer."""
        print("Running sanity check against RandomPlayer...")
        random_player = RandomPlayer()
        wins = 0
        draws = 0
        losses = 0
        total_invalid = 0
        total_illegal = 0
        total_moves = 0
        
        # Create a thread pool
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.num_threads) as executor:
            # Prepare arguments for each game
            game_args = []
            for i in range(self.games_per_level):
                player_is_white = i % 2 == 0
                game_args.append((random_player, 0, player_is_white, self.player_elo))
            
            # Execute games in parallel and collect results
            futures = [executor.submit(self.play_game_thread, args) for args in game_args]
            
            # Process results as they complete
            for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
                score, invalid, illegal, moves, _ = future.result()
                if score == 1.0:
                    wins += 1
                elif score == 0.5:
                    draws += 1
                else:
                    losses += 1
                total_invalid += invalid
                total_illegal += illegal
                total_moves += moves
        
        win_rate = (wins + 0.5 * draws) / self.games_per_level
        
        # Thread-safe update of statistics
        with self.stats_lock:
            self.invalid_moves += total_invalid
            self.illegal_moves += total_illegal
            self.total_moves += total_moves
        
        invalid_pct = (total_invalid / total_moves) * 100 if total_moves > 0 else 0
        illegal_pct = (total_illegal / total_moves) * 100 if total_moves > 0 else 0
        
        print(f"Sanity check results:")
        print(f"Played {wins+draws+losses} games. {wins} wins, {draws} draws, {losses} losses")
        print(f"Invalid moves: {invalid_pct:.2f}%")
        print(f"Illegal moves: {illegal_pct:.2f}%")
        
        return win_rate > 0.5
    
    def run_gauntlet(self,levels) -> Dict:
        """Run the full gauntlet evaluation."""
        results = {
            "final_elo": self.player_elo,
            "elo_history": [],
            "highest_level_reached": None,
            "invalid_move_pct": 0,
            "illegal_move_pct": 0,
            "level_results": {}  # Store detailed results for each level
        }
        
        # First run sanity check
        if not self.run_sanity_check():
            print("Failed sanity check against RandomPlayer. Stopping evaluation.")
            return results
        
        print("\nStarting gauntlet evaluation...")
        self.elo_history.append(self.player_elo)
        
        # Run through each Stockfish level
        for level_idx, level in enumerate(levels):
            opp_elo,opp = level
            print(f"\nEvaluating against Opponent of (ELO: {opp_elo})...")
            
            wins = 0
            draws = 0
            losses = 0
            level_invalid = 0
            level_illegal = 0
            level_moves = 0
            
            # Create a thread pool
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.num_threads) as executor:
                
                # Prepare arguments for each game
                game_args = []
                for i in range(self.games_per_level):
                    player_is_white = i % 2 == 0
                    # Use the appropriate Stockfish instance based on thread index
                    stockfish_idx = i % self.num_threads
                    game_args.append((opp, opp_elo, player_is_white, self.player_elo))
                
                # Execute games in parallel and collect results
                futures = [executor.submit(self.play_game_thread, args) for args in game_args]
                
                # Process results as they complete
                for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
                    score, invalid, illegal, moves, new_elo = future.result()
                    # Update player ELO (the last completed game's ELO will be used)
                    self.player_elo = new_elo
                    
                    if score == 1.0:
                        wins += 1
                    elif score == 0.5:
                        draws += 1
                    else:
                        losses += 1
                    
                    level_invalid += invalid
                    level_illegal += illegal
                    level_moves += moves
                
                # Thread-safe update of statistics
                with self.stats_lock:
                    self.invalid_moves += level_invalid
                    self.illegal_moves += level_illegal
                    self.total_moves += level_moves
            
            win_rate = (wins + 0.5 * draws) / self.games_per_level
            invalid_pct = (level_invalid / level_moves) * 100 if level_moves > 0 else 0
            illegal_pct = (level_illegal / level_moves) * 100 if level_moves > 0 else 0
            
            # Store detailed results for this level
            results["level_results"][opp_elo] = {
                "wins": wins,
                "draws": draws,
                "losses": losses,
                "win_rate": win_rate,
                "invalid_move_pct": invalid_pct,
                "illegal_move_pct": illegal_pct
            }
            
            print(f"Results against Opponent ELO {opp_elo}:")
            print(f"Wins: {wins}, Draws: {draws}, Losses: {losses}")
            print(f"Win rate: {win_rate:.2f}")
            print(f"Current player ELO: {self.player_elo}")
            print(f"Invalid moves: {invalid_pct:.2f}%")
            print(f"Illegal moves: {illegal_pct:.2f}%")
            
            results["highest_level_reached"] = opp_elo
            
            # Stop if win rate is below 50%
            if win_rate < 0.5:
                print(f"Player failed to achieve >50% win rate against Opponent ELO {opp_elo}.")
                break
            
            # If this is the last level and player passed, note that they completed the gauntlet
            if level_idx == len(levels) - 1:
                print("Player successfully completed the entire gauntlet!")
        
        # Calculate final statistics
        results["final_elo"] = self.player_elo
        results["elo_history"] = self.elo_history
        results["invalid_move_pct"] = (self.invalid_moves / self.total_moves) * 100 if self.total_moves > 0 else 0
        results["illegal_move_pct"] = (self.illegal_moves / self.total_moves) * 100 if self.total_moves > 0 else 0
        
        return results
    
    def plot_elo_progression(self, results: Dict,show=False):
        """Plot the ELO progression throughout the gauntlet."""
        plt.figure(figsize=(12, 6))
        plt.plot(results["elo_history"])
        plt.title("Player ELO Progression During Gauntlet")
        plt.xlabel("Games Played")
        plt.ylabel("ELO Rating")
        plt.grid(True)
        plt.savefig("elo_progression.png")
        if show: plt.show()
        
        # Print summary of results
        print(f"Final ELO: {results['final_elo']}")
        print(f"Highest level reached: Opponent ELO {results['highest_level_reached']}")
        print(f"Invalid move percentage: {results['invalid_move_pct']:.2f}%")
        print(f"Illegal move percentage: {results['illegal_move_pct']:.2f}%")
        
        # Print detailed breakdown by level
        print("\nDetailed results by level:")
        for elo, level_result in results["level_results"].items():
            print(f"\nOpponent ELO {elo}:")
            print(f"  Wins: {level_result['wins']}, Draws: {level_result['draws']}, Losses: {level_result['losses']}")
            print(f"  Win rate: {level_result['win_rate']:.2f}")
            print(f"  Invalid moves: {level_result['invalid_move_pct']:.2f}%")
            print(f"  Illegal moves: {level_result['illegal_move_pct']:.2f}%")


if __name__ == "__main__":
    
    # Initialize your LLM player
    player = LLMPlayer("./train_output-final")
    #player = RandomPlayer()
    #player = StockfishPlayer(stockfish_skill_elo_map[1700])
    # Create and run the gauntlet
    levels = []
    for elo in sorted(stockfish_skill_elo_map.keys()):
        levels.append((elo,StockfishPlayer(stockfish_skill_elo_map[elo])))
    #levels.append((780,LLMPlayer("meta-llama/Llama-3.2-1B")))
    gauntlet = ChessGauntlet(player, games_per_level=10, starting_elo=800, num_threads=4)
    results = gauntlet.run_gauntlet(levels)
    
    # Plot results
    gauntlet.plot_elo_progression(results)
    
    # Save results to file
    import json
    with open("gauntlet_results.json", "w") as f:
        json.dump({
            "final_elo": results["final_elo"],
            "highest_level_reached": results["highest_level_reached"],
            "invalid_move_pct": results["invalid_move_pct"],
            "illegal_move_pct": results["illegal_move_pct"],
            "level_results": results["level_results"]
        }, f, indent=2)
