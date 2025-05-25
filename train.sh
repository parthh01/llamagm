#python -m model.train --model_name meta-llama/Llama-3.2-1B --output_dir ./train_output --per_device_train_batch_size 16
accelerate launch --config-file ./accel_config.yaml model/train.py --model_name ./checkpoint-2000 --output_dir ./train_output --per_device_train_batch_size 16 

