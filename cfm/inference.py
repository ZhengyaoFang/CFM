
import os
from collections.abc import Mapping
import torch
import huggingface_hub
from .dataset.utils import process_vision_info
from .dataset.data_collator_qwen import prompt_with_special_token, prompt_without_special_token, INSTRUCTION
from .utils.parser import ModelConfig, PEFTLoraConfig, TrainingConfig, DataConfig, parse_args_with_yaml
from .train import create_model_and_processor
from pathlib import Path

_MODEL_CONFIG_PATH = Path(__file__).parent / f"config/"

class CFMRewardInferencer():
    def __init__(self, config_path=None, checkpoint_path=None, device='cuda', differentiable=False):
        if config_path is None:
                config_path = os.path.join(_MODEL_CONFIG_PATH, 'CFM_7B.yaml')

        if checkpoint_path is None:
            # Use snapshot_download to download the entire model repository
            checkpoint_dir = huggingface_hub.snapshot_download(
                "Nineve/CFM_7B",
                repo_type='model',
                allow_patterns="*.safetensors"
            )
            # Find all safetensors files in the downloaded directory
            safetensors_files = sorted([f for f in os.listdir(checkpoint_dir) if f.endswith('.safetensors')])
            if not safetensors_files:
                raise FileNotFoundError("No safetensors file found in the downloaded model repository")
        
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

        # Load state_dict from checkpoint path or directory
        if checkpoint_path is not None and checkpoint_path.endswith('.safetensors'):
            # Single safetensors file
            import safetensors.torch
            state_dict = safetensors.torch.load_file(checkpoint_path, device="cpu")
        elif checkpoint_path is not None:
            # Single checkpoint file
            state_dict = torch.load(checkpoint_path, map_location="cpu")
        else:
            # Multiple safetensors files - merge them
            import safetensors.torch
            state_dict = {}
            for safetensors_file in safetensors_files:
                file_path = os.path.join(checkpoint_dir, safetensors_file)
                partial_state_dict = safetensors.torch.load_file(file_path, device="cpu")
                state_dict.update(partial_state_dict)

        if "model" in state_dict:
            state_dict = state_dict["model"]
        model.load_state_dict(state_dict, strict=True)
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

    @torch.inference_mode()
    def reward(self, prompts, image_paths, output_attn=False):
        batch = self.prepare_batch(image_paths, prompts)
        input_ids = batch["input_ids"]
        rewards = self.model(
            return_dict=True,
            **batch,
        )
        if not output_attn:
            return rewards["logits"] / self.model.scale_factor
        else:
            hidden_states = rewards["hidden_states"] # [batch, seq_len, dim]
            i = 0
            j = 28
            img_token_indexs = torch.where((input_ids[i] == self.model.config.image_token_id))
            text_token_indexs = torch.where((input_ids[i] != self.model.config.image_token_id))
            text_hidden = hidden_states[j][i][text_token_indexs]  # [text_len, dim]
            img_hidden = hidden_states[j][i][img_token_indexs]    # [img_len, dim]
            # 归一化
            text_hidden = torch.nn.functional.normalize(text_hidden, p=2, dim=-1)
            img_hidden = torch.nn.functional.normalize(img_hidden, p=2, dim=-1)
            # 计算text_hidden和img_hidden的注意力权重
            attn_weights = torch.matmul(text_hidden, img_hidden.T)  # [text_len, img_len]
            temperature = 10
            attn_weights = torch.softmax(attn_weights / temperature, dim=-1)  # [text_len, img_len]   
            # 将attn_weights的值约束在原本值的80%大小以下，超出部分按80%计算
            
            # 计算图像的attention map
            img_attention_map = torch.mean(attn_weights, dim=0)  # [img_len]
            # 可视化注意力图
            h_grid = batch['image_grid_thw'][i][1] // 2
            w_grid = batch['image_grid_thw'][i][2] // 2
            attn_map = img_attention_map.reshape(h_grid, w_grid).to(torch.float32)
            attn_map = (attn_map - attn_map.min()) / (attn_map.max() - attn_map.min() + 1e-8)
            return rewards["logits"], attn_map
