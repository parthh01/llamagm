from transformers import AutoModelForCausalLM, AutoTokenizer,BitsAndBytesConfig
import os 
import json 
import torch 

model_name = "./open_llama_7b-lora-final"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
model_name,
#quantization_config=BitsAndBytesConfig(load_in_8bit=True),
torch_dtype="auto",
device_map="cpu", # "auto"
offload_folder="./offload_folder",
)

system_prompt = """You are a chess player.
You will be given a list of previous moves of a game in algebraic
notation, the color of your pieces, and a list of possible moves
in the format below.
{moveHistory: string[],
possibleMoves: string[],
color: "b" | "w" }
The move history will be alternating moves in algebraic starting
with white.
It is your job to pick the best possible move to win the game.
The move returned MUST BE PART OF THE SET OF POSSIBLE MOVES
GIVEN.
Here is an example set of possible moves at the start of the game
for white pieces:
['a3', 'a4', 'b3', 'b4', 'c3', 'c4', 'd3', 'd4', 'e3', 'e4',
'f3', 'f4', 'g3', 'g4', 'h3', 'h4', 'Na3', 'Nc3', 'Nf3', 'Nh3'].
Your response must be a JSON object with the key "move" which
will be your move, and the key "reasoning" which will be your
reasoning for making that move.
For example, a valid response would be {"move": "Nh3",
"reasoning": "I moved my knight to h3 to control the center of
the board."}
Your reasoning must be limited to 10 words or less."""


def test_chess_model(position_data):
	# Format input with system prompt
	prompt = f"[INST] {system_prompt}\n\n{json.dumps(position_data)} [/INST]" 
	inputs = tokenizer(prompt, return_tensors="pt").to(model.device)  # Move inputs to same device as model
	# Generate response
	outputs = model.generate(inputs["input_ids"],
	max_new_tokens=40)
	response = tokenizer.decode(outputs[0],
	skip_special_tokens=True)
	# Parse JSON response
	try:
		# Extract response (after instruction)
		model_response = response.split("[/INST]")[1].strip()
		response_json = json.loads(model_response)
		return response_json
	except:
		print(response)
		return {"error": "Invalid response format"}

if __name__ == "__main__":
	pos_data = {
"moveHistory": ["e4", "e5"],
"possibleMoves": ["Nf3", "Nc3", "Bb5", "Bc4", "Ng5", "d4"], 
"color": "w"
}
	response = test_chess_model(pos_data)
	print(response)
