import logging
import os
from contextlib import nullcontext

TRL_USE_RICH = os.environ.get("TRL_USE_RICH", False)

from trl.commands.cli_utils import init_zero_verbose, SFTScriptArguments, TrlParser

if TRL_USE_RICH:
    init_zero_verbose()
    FORMAT = "%(message)s"

    from rich.console import Console
    from rich.logging import RichHandler

import torch
from accelerate import Accelerator
from datasets import load_dataset

from tqdm.rich import tqdm
from transformers import AutoTokenizer, AutoProcessor, LlavaForConditionalGeneration

from trl import (
    ModelConfig,
    RichProgressCallback,
    SFTTrainer,
    SFTConfig,
    get_peft_config,
    get_quantization_config,
    get_kbit_device_map
)

tqdm.pandas()

if TRL_USE_RICH:
    logging.basicConfig(format=FORMAT, datefmt="[%X]", handlers=[RichHandler()], level=logging.INFO)


if __name__ == "__main__":
    parser = TrlParser((SFTScriptArguments, SFTConfig, ModelConfig))
    sft_script_args, training_args, model_config = parser.parse_args_and_config()
    training_args.gradient_checkpointing_kwargs = dict(use_reentrant=False)
    # Force use our print callback
    if TRL_USE_RICH:
        training_args.disable_tqdm = True
        console = Console()
    
    ################
    # Model, Tokenizer & Processor
    ################
    LLAVA_CHAT_TEMPLATE = """{% if not add_generation_prompt is defined %}{% set add_generation_prompt = false %}{% endif %}A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. {% for message in messages %}{% if message['role'] == 'user' %}USER: {% else %}ASSISTANT: {% endif %}{% for item in message['content'] %}{% if item['type'] == 'text' %}{{ item['text'] }}{% elif item['type'] == 'image' %}<image>{% endif %}{% endfor %}{% if message['role'] == 'user' %} {% else %}{{eos_token}}{% endif %}{% endfor %}{% if add_generation_prompt %}ASSISTANT: {% endif %}"""

    torch_dtype = (
        model_config.torch_dtype
        if model_config.torch_dtype in ["auto", None]
        else getattr(torch, model_config.torch_dtype)
    )
    quantization_config = get_quantization_config(model_config)
    model_kwargs = dict(
        revision=model_config.model_revision,
        trust_remote_code=model_config.trust_remote_code,
        attn_implementation=model_config.attn_implementation,
        torch_dtype=torch_dtype,
        device_map=get_kbit_device_map() if quantization_config is not None else None,
        quantization_config=quantization_config
    )
    tokenizer = AutoTokenizer.from_pretrained(model_config.model_name_or_path, use_fast=True)
    tokenizer.chat_template = LLAVA_CHAT_TEMPLATE
    processor = AutoProcessor.from_pretrained(model_config.model_name_or_path)
    processor.tokenizer = tokenizer

    model = LlavaForConditionalGeneration.from_pretrained(model_config.model_name_or_path,  load_in_8bit=True, **model_kwargs)
    print(model)
    # print(model.named_modules())
    ################
    # Create a data collator to encode text and image pairs
    ################
    class LLavaDataCollator:
        def __init__(self, processor):
            self.processor = processor
        
        def __call__(self, examples):
            texts = []
            images = []
            for example in examples:
                image = example["images"]
                if not isinstance(image, list):
                    image = [image]
                if len(image) > 1:
                    raise ValueError("This collator only supports one image per example")
                
                messages = example["messages"]
                text = self.processor.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=False
                )
                texts.append(text)
                images.append(image[0])
            
            batch = self.processor(texts, images, return_tensors="pt", padding=True)

            labels = batch["input_ids"].clone()
            if self.processor.tokenizer.pad_token_id is not None:
                labels[labels == self.processor.tokenizer.pad_token_id] = -100
            batch["labels"] = labels

            return batch
    
    data_collator = LLavaDataCollator(processor)

    ################
    # Dataset
    ################
    raw_datasets = load_dataset(sft_script_args.dataset_name)
    train_dataset = raw_datasets[sft_script_args.dataset_train_split]
    eval_dataset = raw_datasets[sft_script_args.dataset_test_split]

    ###############
    # Optional rich context managers
    ###############

    init_context = nullcontext() if not TRL_USE_RICH else console.status("[bold green]Initializing the SFTTrainer...")
    save_context = (
        nullcontext()
        if not TRL_USE_RICH
        else console.status(f"[bold green]Training completed! Saving the model to {training_args.output_dir}")
    )

    ################
    # Training
    ################
    with init_context:
        trainer = SFTTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            dataset_text_field="text",
            tokenizer=tokenizer,
            peft_config=get_peft_config(model_config),
            callbacks=[RichProgressCallback] if TRL_USE_RICH else None,
            data_collator=data_collator,
            dataset_kwargs={"skip_prepare_dataset": True}
        )
    
    trainer.train()

    with save_context:
        trainer.save_model(training_args.output_dir)
        trainer.push_to_hub()
        if Accelerator().is_main_process:
            processor.push_to_hub(training_args.hub_model_id)
    


