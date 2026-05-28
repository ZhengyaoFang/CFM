import torch
from torch.utils.data import Dataset
import random
import json

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
        img_choices = random.sample([i for i in range(len(sample["imgs"]))], 2)
        min_choice = min(img_choices)
        max_choice = max(img_choices)
        image_1 = sample["imgs"][min_choice]
        image_2 = sample["imgs"][max_choice]
        text_1 = sample["txt"]
        text_2 = text_1
        
        # Process label
        if self.soft_label:
            choice_dist = sorted(sample.get("choice_dist", [1.0, 0.0]), reverse=True)
            label = torch.tensor(choice_dist[0]) / torch.sum(torch.tensor(choice_dist))
        else:
            label = torch.tensor(1).float()

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
