
Strat: 

 - generate a dataset categorizing the game in three stages, each of these will have a separate labeling format: 
  - opening: this is easy it should just mention that this is theory and no calculation required as such, here we are just looking for recall
  - midgame: we can label the reasoning string to be stockfish's evaluation of the position by win%, thereby we can evaluate the model's reasoning ability by how well it's reasoning matches stockfish's evaluation. 
  - endgame: this should be straightforward like the opening. Reasoning here should just lay out forced win in how many moves. 

DATA generation is split into three parts: 
 - Opening Datagen
  prereqs: 
  - https://github.com/lichess-org/chess-openings all the tsv files in here. 
 - Middlegame Datagen - characterized as any game state not covered in opening and does not have a forced win. 
 - Endgame Datagen - any game state with a forced win. 
