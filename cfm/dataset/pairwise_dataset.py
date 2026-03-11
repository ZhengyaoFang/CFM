import os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import io
import random
import json
from tqdm import tqdm

class PairwiseOriginalDataset(Dataset):
    def __init__(
        self,
        json_list,
        soft_label=False,
        confidence_threshold=None,
    ):
        self.samples = []
        for json_file in json_list:
            with open(json_file, "r") as f:
                data = json.load(f)
            self.samples.extend(data)
        self.soft_label = soft_label
        self.confidence_threshold = confidence_threshold
        self.num_ranks = 6
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx % len(self.samples)]
        sample = {
            "imgs": [sample["real.jpg"]] + [sample[f"rank{i}.jpg"] for i in range(self.num_ranks)],
            "txt": sample["txt"],
            "choice_dist": sample.get("choice_dist", [1.0, 0.0]),
            "confidence": sample.get("confidence", 1.0),
        }
        return self.get_single_item(sample)
    
    def get_single_item(self, sample):
        # 从sample中获取必要的内容
        img_choices = random.sample([i for i in range(len(sample["imgs"]))], 2)
        min_choice = min(img_choices)
        max_choice = max(img_choices)
        image_1 = sample["imgs"][min_choice]
        image_2 = sample["imgs"][max_choice]
        text_1 = sample["txt"]
        text_2 = text_1  # 假设文本是相同的，如果有变化根据需要修改
        
        # Process Label
        if self.soft_label:
            choice_dist = sorted(sample.get("choice_dist", [1.0, 0.0]), reverse=True)
            label = torch.tensor(choice_dist[0]) / torch.sum(torch.tensor(choice_dist))
        else:
            label = torch.tensor(1).float()

        # 返回符合训练的字典
        return {
            "image_1": sample["imgs"][0],
            "image_2": sample["imgs"][1],
            "image_3": sample["imgs"][2],
            "image_4": sample["imgs"][3],
            "image_5": sample["imgs"][4],
            "image_6": sample["imgs"][5],
            "image_7": sample["imgs"][6],
            "text_1": text_1,
            "text_2": text_2,
            "label": label,
            "confidence": sample.get("confidence", 1.0),
            "choice_dist": torch.tensor(sample.get("choice_dist", [1.0, 0.0])),
        }
    

# class PairwiseWebDataset(IterableDataset):
#     def __init__(self, webdataset_path, soft_label=False, confidence_threshold=None):
#         # 读取webdataset路径
#         self.webdataset_path = webdataset_path
#         self.soft_label = soft_label
#         self.confidence_threshold = confidence_threshold
        
#         shuffle_buffer = 100
#         self.num_ranks = 5
#         self.shard_reader = (
#             wds.WebDataset(self.webdataset_path, shardshuffle=True)
#             .shuffle(shuffle_buffer)  # 打乱数据顺序
#             .decode("pil")  # 解码图像为 PIL 对象
#         )
#         self.iterator = iter(self.shard_reader)


#     def _apply_confidence_threshold(self):
#         # 根据confidence_threshold过滤样本
#         new_samples = []
#         for sample in tqdm(self.samples, desc="Filtering samples according to confidence threshold"):
#             if sample.get("confidence", float("inf")) >= self.confidence_threshold:
#                 new_samples.append(sample)
#         self.samples = new_samples

#     def __len__(self):
#         return 18280

#     def __iter__(self):
#         # 获取单个样本，确保符合格式
#         sample = next(self.iterator)
#         try:
#             return self.get_single_item(sample)
#         except Exception as e:
#             print(f"Error processing sample: {e}")
#             return self.get_single_item(self.__iter__())

#     def get_single_item(self, sample):
#         # 从sample中获取必要的内容
        
#         image_1_data = sample["real.jpg"]
#         image_1 = Image.open(io.BytesIO(image_1_data)).convert("RGB")
        

#         rank_choice = random.randint(0, self.num_ranks-1)
#         rank_choice = f"rank{rank_choice}.jpg"
#         image_2_data = sample[rank_choice]  # 选择一个rank图像，通常需要根据实验调整
#         image_2 = Image.open(io.BytesIO(image_2_data)).convert("RGB")

#         text_1 = sample["txt"].decode("utf-8")
#         text_2 = text_1  # 假设文本是相同的，如果有变化根据需要修改
        
#         # Process Label
#         if self.soft_label:
#             choice_dist = sorted(sample.get("choice_dist", [1.0, 0.0]), reverse=True)
#             label = torch.tensor(choice_dist[0]) / torch.sum(torch.tensor(choice_dist))
#         else:
#             label = torch.tensor(1).float()

#         # 返回符合训练的字典
#         return {
#             "image_1": image_1,
#             "image_2": image_2,
#             "text_1": text_1,
#             "text_2": text_2,
#             "label": label,
#             "confidence": sample.get("confidence", 1.0),
#             "choice_dist": torch.tensor(sample.get("choice_dist", [1.0, 0.0])),
#         }