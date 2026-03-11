import pdb
from dataclasses import dataclass, field
from typing import Optional, List, Union
import numpy as np
import pandas as pd
import torch
from cfm.dataset.utils import process_vision_info
from torch.utils.data import Dataset
import torchvision.transforms.functional as F

INSTRUCTION = """
You are tasked with evaluating a generated image based on Relative Color Realism, which focuses on the rationality and harmony of colors in the context of the specific scene described by the textual prompt. Please provide a rating from 0 to 10, with 0 being the worst and 10 being the best.

**Color Realism:**  
Evaluate the authenticity and appropriateness of colors in the image relative to the scenario, environment, and context specified in the textual prompt. The following sub-dimensions should be considered:
- **Object-Scene Color Consistency:** Assess whether the colors of key objects in the image align with their typical or logically implied appearance within the prompt’s scenario.
- **Lighting-Context Color Harmony:** Evaluate if the color cast, brightness, and contrast of the image match the lighting conditions implied by the prompt. 
- **Saturation Fit to Scenario:** Consider whether color saturation levels are appropriate for the prompt’s context.
- **Color Transition Contextual Logic:** Check if color transitions between elements (e.g., objects, background, shadows) are natural within the prompt’s scenario.
- **Overall Color Scheme Coherence with Prompt:** Assess whether the entire color palette of the image conveys the atmosphere, mood, or setting described in the prompt. For example, a "haunted house at midnight" should have a color scheme that feels eerie and dark (e.g., deep blues, dim grays, faint candlelight yellows), aligning with the spooky, nighttime context, rather than bright, cheerful colors that clash with the scenario.

Textual prompt - {text_prompt}

"""

INSTRUCTION_debug = """
{text_prompt}
"""

prompt_with_special_token = """
Please provide the overall ratings of this image: <|Reward|>

END
"""

prompt_without_special_token = """
Please provide the overall ratings of this image: 
"""


class QWen2VLDataCollator:
    def __init__(
        self,
        processor,
        with_instruction=True,
        max_pixels=256 * 28 * 28,  # Default max pixels
        min_pixels=256 * 28 * 28,  # Default min pixels
        use_special_tokens=True,
    ):
        self.processor = processor
        self.with_instruction = with_instruction
        self.max_pixels = max_pixels
        self.min_pixels = min_pixels
        self.use_special_tokens = use_special_tokens

    def _clean_message(
        self,
        texts,
        images,
        max_pixels=256 * 28 * 28,
        min_pixels=256 * 28 * 28,
        with_instruction=True,
        use_special_tokens=True,
    ):
        """
        remove unnecessary keys from message(very very necessary)
        """
        message_list = []
        for text, image in zip(texts, images):
            out_message = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "image": image,
                            "min_pixels": min_pixels,
                            "max_pixels": max_pixels,
                        },
                        {
                            "type": "text",
                            "text": (
                                INSTRUCTION.format(text_prompt=text)
                                + prompt_with_special_token
                                if use_special_tokens
                                else prompt_without_special_token
                            ),
                        },
                    ],
                }
            ]

            message_list.append(out_message)

        return message_list

    def _pad_sequence(self, sequences, attention_mask, max_len, padding_side="right"):
        """
        Pad the sequences to the maximum length.
        """
        assert padding_side in ["right", "left"]
        if sequences.shape[1] >= max_len:
            return sequences, attention_mask

        pad_len = max_len - sequences.shape[1]
        padding = (0, pad_len) if padding_side == "right" else (pad_len, 0)

        sequences_padded = torch.nn.functional.pad(
            sequences, padding, "constant", self.processor.tokenizer.pad_token_id
        )
        attention_mask_padded = torch.nn.functional.pad(
            attention_mask, padding, "constant", 0
        )

        return sequences_padded, attention_mask_padded

    def __call__(self, inputs, with_instruction=True):
        """
        Preprocess inputs to token sequences and return a batch
        """

        # 扩展 image_3 ~ image_7
        images_1, images_2, images_3, images_4, images_5, images_6, images_7 = [], [], [], [], [], [], []
        texts_1, texts_2 = [], []

        for idx, batch in enumerate(inputs):
            texts_1.append(batch["text_1"])
            texts_2.append(batch["text_2"])
            images_1.append(batch["image_1"])
            images_2.append(batch["image_2"])
            # 如果存在扩展图片则添加，否则填None或空
            for i in range(3, 8):
                key = f"image_{i}"
                if key in batch:
                    locals()[f"images_{i}"].append(batch[key])
                else:
                    locals()[f"images_{i}"].append(None)

        # ========= 构造 messages ==========
        messages_batch_1 = self._clean_message(
            texts_1,
            images_1,
            max_pixels=self.max_pixels,
            min_pixels=self.min_pixels,
            with_instruction=self.with_instruction,
            use_special_tokens=self.use_special_tokens,
        )
        messages_batch_2 = self._clean_message(
            texts_2,
            images_2,
            max_pixels=self.max_pixels,
            min_pixels=self.min_pixels,
            with_instruction=self.with_instruction,
            use_special_tokens=self.use_special_tokens,
        )

        # 所有 3~7 的 text 都用 text_1
        messages_batch_3 = self._clean_message(
            texts_1, images_3,
            max_pixels=self.max_pixels, min_pixels=self.min_pixels,
            with_instruction=self.with_instruction, use_special_tokens=self.use_special_tokens,
        )
        messages_batch_4 = self._clean_message(
            texts_1, images_4,
            max_pixels=self.max_pixels, min_pixels=self.min_pixels,
            with_instruction=self.with_instruction, use_special_tokens=self.use_special_tokens,
        )
        messages_batch_5 = self._clean_message(
            texts_1, images_5,
            max_pixels=self.max_pixels, min_pixels=self.min_pixels,
            with_instruction=self.with_instruction, use_special_tokens=self.use_special_tokens,
        )
        messages_batch_6 = self._clean_message(
            texts_1, images_6,
            max_pixels=self.max_pixels, min_pixels=self.min_pixels,
            with_instruction=self.with_instruction, use_special_tokens=self.use_special_tokens,
        )
        messages_batch_7 = self._clean_message(
            texts_1, images_7,
            max_pixels=self.max_pixels, min_pixels=self.min_pixels,
            with_instruction=self.with_instruction, use_special_tokens=self.use_special_tokens,
        )

        # ========= 处理视觉输入 ==========
        def _proc(messages):
            imgs, _ = process_vision_info(messages)
            imgs = [np.array(i) / 255.0 for i in imgs]
            return imgs

        image_inputs_1 = _proc(messages_batch_1)
        image_inputs_2 = _proc(messages_batch_2)
        image_inputs_3 = _proc(messages_batch_3)
        image_inputs_4 = _proc(messages_batch_4)
        image_inputs_5 = _proc(messages_batch_5)
        image_inputs_6 = _proc(messages_batch_6)
        image_inputs_7 = _proc(messages_batch_7)

        do_rescale = False

        def _make_batch(messages, imgs):
            return self.processor(
                text=self.processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                ),
                images=imgs,
                videos=None,
                padding=True,
                return_tensors="pt",
                images_kwargs={"do_rescale": do_rescale},
            )

        batch_1 = _make_batch(messages_batch_1, image_inputs_1)
        batch_2 = _make_batch(messages_batch_2, image_inputs_2)
        batch_3 = _make_batch(messages_batch_3, image_inputs_3)
        batch_4 = _make_batch(messages_batch_4, image_inputs_4)
        batch_5 = _make_batch(messages_batch_5, image_inputs_5)
        batch_6 = _make_batch(messages_batch_6, image_inputs_6)
        batch_7 = _make_batch(messages_batch_7, image_inputs_7)

        # ========= Padding 同步 ==========
        max_len = max(
            batch_1["input_ids"].shape[1],
            batch_2["input_ids"].shape[1],
            batch_3["input_ids"].shape[1],
            batch_4["input_ids"].shape[1],
            batch_5["input_ids"].shape[1],
            batch_6["input_ids"].shape[1],
            batch_7["input_ids"].shape[1],
        )

        def _pad(b):
            b["input_ids"], b["attention_mask"] = self._pad_sequence(
                b["input_ids"], b["attention_mask"], max_len, "right"
            )
            return b

        batch_1 = _pad(batch_1)
        batch_2 = _pad(batch_2)
        batch_3 = _pad(batch_3)
        batch_4 = _pad(batch_4)
        batch_5 = _pad(batch_5)
        batch_6 = _pad(batch_6)
        batch_7 = _pad(batch_7)

        # ========= 汇总 ==========
        batch = {
            "batch_1": batch_1,
            "batch_2": batch_2,
            "batch_3": batch_3,
            "batch_4": batch_4,
            "batch_5": batch_5,
            "batch_6": batch_6,
            "batch_7": batch_7,
            "choice_dist": torch.stack([batch["choice_dist"] for batch in inputs]),
            "text_1": texts_1,
            "text_2": texts_2,
            "image_1": image_inputs_1,
            "image_2": image_inputs_2,
            "image_3": image_inputs_3,
            "image_4": image_inputs_4,
            "image_5": image_inputs_5,
            "image_6": image_inputs_6,
            "image_7": image_inputs_7,
        }

        return batch
