
import os
from collections.abc import Mapping
import torch
import numpy as np
from PIL import Image
import huggingface_hub
from cfm.dataset.utils import process_vision_info
from cfm.dataset.data_collator_qwen import prompt_with_special_token, prompt_without_special_token, INSTRUCTION
from cfm.utils.parser import ModelConfig, PEFTLoraConfig, TrainingConfig, DataConfig, parse_args_with_yaml
from cfm.train import create_model_and_processor
from pathlib import Path

_MODEL_CONFIG_PATH = Path(__file__).parent / f"config/"

class CFMRewardInferencer():
    def __init__(self, config_path=None, checkpoint_path=None, device='cuda', differentiable=False):
        if config_path is None:
                config_path = os.path.join(_MODEL_CONFIG_PATH, 'CFM_7B.yaml')


        (data_config, training_args, model_config, peft_lora_config), config_path = (
            parse_args_with_yaml(
                (DataConfig, TrainingConfig, ModelConfig, PEFTLoraConfig), config_path, is_train=False
            )
        )
        training_args.output_dir = os.path.join(
            training_args.output_dir, config_path.split("/")[-1].split(".")[0]
        )
        model, processor, peft_config = create_model_and_processor(
            model_config=model_config,
            peft_lora_config=peft_lora_config,
            training_args=training_args,
            differentiable=differentiable,
        )

        self.device = device
        self.use_special_tokens = model_config.use_special_tokens

        state_dict = torch.load(checkpoint_path , map_location="cpu")
        if "model" in state_dict:
            state_dict = state_dict["model"]
        model.load_state_dict(state_dict, strict=False)
        model.eval()

        self.model = model
        self.processor = processor

        self.model.to(self.device)
        self.data_config = data_config

    def _pad_sequence(self, sequences, attention_mask, max_len, padding_side='right'):
        """
        Pad the sequences to the maximum length.
        """
        assert padding_side in ['right', 'left']
        if sequences.shape[1] >= max_len:
            return sequences, attention_mask
        
        pad_len = max_len - sequences.shape[1]
        padding = (0, pad_len) if padding_side == 'right' else (pad_len, 0)

        sequences_padded = torch.nn.functional.pad(sequences, padding, 'constant', self.processor.tokenizer.pad_token_id)
        attention_mask_padded = torch.nn.functional.pad(attention_mask, padding, 'constant', 0)

        return sequences_padded, attention_mask_padded
    
    def _prepare_input(self, data):
        """
        Prepare `inputs` before feeding them to the model, converting them to tensors if they are not already and
        handling potential state.
        """
        if isinstance(data, Mapping):
            return type(data)({k: self._prepare_input(v) for k, v in data.items()})
        elif isinstance(data, (tuple, list)):
            return type(data)(self._prepare_input(v) for v in data)
        elif isinstance(data, torch.Tensor):
            kwargs = {"device": self.device}
            ## TODO: Maybe need to add dtype
            # if self.is_deepspeed_enabled and (torch.is_floating_point(data) or torch.is_complex(data)):
            #     # NLP models inputs are int/uint and those get adjusted to the right dtype of the
            #     # embedding. Other models such as wav2vec2's inputs are already float and thus
            #     # may need special handling to match the dtypes of the model
            #     kwargs.update({"dtype": self.accelerator.state.deepspeed_plugin.hf_ds_config.dtype()})
            return data.to(**kwargs)
        return data
    
    def _prepare_inputs(self, inputs):
        """
        Prepare `inputs` before feeding them to the model, converting them to tensors if they are not already and
        handling potential state.
        """
        inputs = self._prepare_input(inputs)
        if len(inputs) == 0:
            raise ValueError
        return inputs
    
    def prepare_batch(self, image_paths, prompts):
        max_pixels = 256 * 28 * 28
        min_pixels = 256 * 28 * 28
        message_list = []
        for text, image in zip(prompts, image_paths):
            out_message = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "image": image,
                            "min_pixels": max_pixels,
                            "max_pixels": max_pixels,
                        },
                        {
                            "type": "text",
                            "text": (
                                INSTRUCTION.format(text_prompt=text)
                                + prompt_with_special_token
                                if self.use_special_tokens
                                else prompt_without_special_token
                            ),
                        },
                    ],
                }
            ]

            message_list.append(out_message)

        image_inputs, _ = process_vision_info(message_list)

        batch = self.processor(
            text=self.processor.apply_chat_template(message_list, tokenize=False, add_generation_prompt=True),
            images=image_inputs,
            padding=True,
            return_tensors="pt",
            videos_kwargs={"do_rescale": True},
        )
        batch = self._prepare_inputs(batch)
        return batch

    def reward(self, image_paths, prompts):
        
        batch = self.prepare_batch(image_paths, prompts)
        rewards = self.model(
            return_dict=True,
            **batch
        )["logits"]

        return rewards
