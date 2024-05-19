python ../src/models/train_vsft.py \
    --dataset_name="HuggingFaceH4/llava-instruct-mix-vsft" \
    --model_name_or_path="llava-hf/llava-1.5-7b-hf" \
    --report_to="wandb" \
    --learning_rate=1.4e-5 \
    --per_device_train_batch_size=1 \
    --gradient_accumulation_steps=1 \
    --output_dir="data/vsft-llava-1.5-7b-hf" \
    --logging_steps=5 \
    --num_train_epochs=1 \
    --push_to_hub \
    --gradient_checkpointing \
    --remove_unused_columns=False \
    --torch_dtype=float16 \
    --fp16=True \
    --use_peft=True \
    --lora_r=64 \
    --lora_alpha=16 \
    --lora_target_modules="all-linear"

# python /home/firqaaa/Python/VLM/train_vsft.py \
#     --dataset_name="HuggingFaceH4/llava-instruct-mix-vsft" \
#     --model_name_or_path="llava-hf/llava-1.5-7b-hf" \
#     --report_to="wandb" \
#     --learning_rate=1.4e-5 \
#     --per_device_train_batch_size=8 \
#     --gradient_accumulation_steps=1 \
#     --output_dir="data/vsft-llava-1.5-7b-hf" \
#     --logging_steps=5 \
#     --num_train_epochs=1 \
#     --push_to_hub \
#     --gradient_checkpointing \
#     --remove_unused_columns=False \
#     --torch_dtype=float16 \
#     --fp16=True