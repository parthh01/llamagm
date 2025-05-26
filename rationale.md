https://x.com/auraonchain/status/1923468963582759071

LLM chess tournament. Presumably everyone's going to SFT on stockfish games. The question is what can we do that's better. 

learning an obscure/terrible opening for white but very well is an interesting idea (ie trained on all the lines for a given opening) with the idea that most people are pretraining on stockfish and hopefully not be able to generalize to it. But then for black it would have to learn all the main openings anyway since other llms will be using those. 

perhaps learning the 'anti-llm' opening might be an interesting task best left to post SFT RL. 

one thing feels clear: Despite the guide's insinuations, we are optimizing for optimal move next token prediction and not reasoning by any stretch. It remains to be proven whether or not single token prediction meaningfully abstracts it (reasoning), and even the models that approximate it are implementing CoT, something that seems not allowed in this competition. 
As such, SFT/PT on a sequence of good moves is the only way to maximize the likelihood the model will produce them. Basically the move the model will be trained on outputting should only be good moves. This means the games used cannot be just random games, they'll contain bad moves. One way to identify good moves is obviously ones with a high stockfish valuation in the resulting game state, but that seems too simplistic.

This isn't a trivial assertion, because a good move is independent of a good game. Training on grandmaster games and games between engines seeemingly contain good moves, but they are good moves against good opponents. essentially the model will learn to play well when it's opponent does. Wayward or random moves can still throw it off. we need a more general way to sample good moves. 


idk if i'll have time to implement this, but i suspect the optimal strategy is to SFT the model on the distribution of moves of stockfish eval instead of state-best move pairs. (stockfish has an eval for each possible move) the most scalable thing the model can learn is exactly this softmax distribution, it should jack sample efficiency right up and prove far more scalable than solely training on the state-best move pair. 

prereqs:
 - cuda  
 - stockfish 
 - python 
 - huggingface api key 


Sources for ideas, code, and other help: 

 - https://arxiv.org/html/2501.17186v2#:~:text=Dataset,data%20by%20350%20Elo%20points
 - https://github.com/lichess-org/chess-openings/tree/master


I think the tournament turned out to be some kind of scam, there doesn't seem to be any way to actually submit a model. in any case, I don't care to burn any more money on GPU's behind this, though early scaling experiments appear to validate the idea I've come up with. The scaling experiments I've ran so far prove the following: 

- the model learns to play the specific (bongcloud opening) with high recall when SFT'd on the dataset of likely trajectories, while still being able to play all other openings as black. 
- GRPO does indeed improve the model's level of play, while maintaining the ability to play the opening committed to memory via SFT. (albeit with specific reward shaping)
- the lower the train (and eval) loss on the dataset, the better the model is able to play against itself (trained to a higher loss)
- the longer GRPO is trained, the better the model is able to play against itself (not GRPO trained or less GRPO trained)
