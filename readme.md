# Chess LLM with Stockfish Distribution Training

Chess LLM training system SFT'ed to play a specific opening trajectory as white, GRPO'ed to elicit generalized efficient play as black. 

TODO: implementing stockfish evaluation distribution learning rather than traditional best-move supervision. Built around the hypothesis that training on move evaluation distributions improves sample efficiency over state-best move pairs.

## Technical Implementation

**Multi-Stage Data Generation Pipeline:**
- **Opening**: Theory-based recall training with a very small generated set of trajectories, specialized bongcloud opening memorization

- **Midgame**: Stockfish evaluation labeling with centipawn values (`eval: {value}`) and mate notation (`M{moves}`)  
- **Endgame**: Forced win calculation with move count reasoning (didn't end up adding any games for this) but this would be good to add

**Training Architecture:**
- Base model: Llama with LoRA adapters (r=16, alpha=32)
- SFT on trajectory datasets with JSON move/reasoning format
- GRPO post-training with custom reward shaping:
  - Legal move reward: +1.0
  - Position improvement: +5.0  
  - Evaluation accuracy: exponential decay based on centipawn error
  - Win/loss terminal rewards: ±200.0

**GRPO Environment:**
- Thread-local Stockfish instances for parallel training
- Progressive skill level increases (1→20) during training
- Reward calculation based on move legality, position evaluation, and reasoning accuracy
- Memory-optimized batch generation with CUDA autocast

**Evaluation Suite:**
- ELO-based gauntlet system against Stockfish skill levels
- Parallel game execution with thread-safe statistics
- Bongcloud opening recall verification
- Invalid/illegal move percentage tracking

## Decisions: 

**Reward Shaping:** The GRPO reward function combines multiple signals - JSON parsing, move legality, evaluation accuracy, and position improvement. Evaluation accuracy uses exponential decay: `accuracy_reward = base_reward * exp(-error/200)` where error is in centipawns.

**Data Quality:** Rather than training on random games containing bad moves, the system generates positions where all candidate moves are evaluated by Stockfish, creating a cleaner training signal. Though this would be a good idea to add, KTO training (https://huggingface.co/papers/2402.01306) is a method that implements negative sampling. 

**Memory Management:** GRPO training uses batch processing with explicit CUDA cache clearing and autocast for memory efficiency during self-play generation.

**Progressive GRPO env:** Stockfish opponent difficulty increases periodically, ensuring the model can learn progressively better play. adapting the gauntlet to be truly reactive env would be another good further improvement, it would ensure it's allowing for the model to robustly stay in the dense reward regime while incrementally getting better. the current method to ensure this is primitive. 

## Scaling experiment results

- Model successfully memorizes bongcloud opening sequences while maintaining general chess ability (plays the opening >=90% of the time)
- GRPO demonstrably improves play quality (GRPO'd model achieves higher ELO in test suite, than equivalent training steps SFT model) 
- Lower train/eval loss correlates with better self-play performance, and against a random move player. (correlation examined via ELO in test suite) 
- Progressive Stockfish training shows continued improvement against stronger opponents (ELO rating trajectory gradient continually positive) 

training logs: https://api.wandb.ai/links/critique-labs-ai/im2tdep8

## Prerequisites
- CUDA + NVIDIA GPU's
- Stockfish engine
- PostgreSQL database
- HuggingFace API key
- wandb account 

## Architecture Files
- `datagen/gen.py`: Multi-threaded data generation with Stockfish evaluation
- `model/learn.py`: GRPO training environment and reward calculation  
- `eval/suite.py`: ELO-based evaluation gauntlet
- `constants.py`: System prompt and JSON response format


rationale.md contains the running diary of sorts i was keeping while building this. quite a few parts of the code were vibe coded, so it's quite messy and inefficient, there's duplication all over the place, but this was meant to be done in a couple days for this AI chess competition hosted by some company that never ended up getting back to me about how to submit, and also has a very dodgy telegram channel so I think the whole company might be a scam. 

dataset used for SFT: https://huggingface.co/parthh01/chess-llm-tournament
final model: https://huggingface.co/parthh01/chess-llm-tournament
training logs: <wanbd report here >
