from transformers import AutoModelForCausalLM, AutoTokenizer,BitsAndBytesConfig
import os 
import json 
import torch 
from constants import system_prompt
model_name = "./openlm-research/open_llama_7b-lora/checkpoint-500"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
model_name,
#quantization_config=BitsAndBytesConfig(load_in_8bit=True),
torch_dtype="auto",
device_map= "auto",
offload_folder="./offload_folder",
)




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

