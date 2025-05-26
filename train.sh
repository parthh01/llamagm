#python -m model.train --model_name meta-llama/Llama-3.2-1B --output_dir ./train_output --per_device_train_batch_size 16
#make sure to run accelerate config before running this script
accelerate launch model/train.py --model_name meta-llama/Llama-3.2-1B --output_dir ./train_output --per_device_train_batch_size 16 
