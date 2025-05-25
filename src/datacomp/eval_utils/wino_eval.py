"""Evaluate on WinoGAViL dataset."""

import os

import datasets
import numpy as np
import open_clip
import torch

# from collections import Counter
from sklearn.metrics import jaccard_score
from tqdm import tqdm

from .wds_eval import create_model

# from transformers import CLIPModel, CLIPProcessor


class WinoDataset(torch.utils.data.Dataset):
    def __init__(self, hf_dataset, transform=None, text_transform=None):
        super().__init__()
        self._dataset = hf_dataset
        self.transform = (lambda x: x) if transform is None else transform
        self.text_transform = (
            (lambda x: x) if text_transform is None else text_transform
        )

    def __len__(self):
        return len(self._dataset)

    def __getitem__(self, index: int):
        example = self._dataset[index]
        return (
            self.transform(example["candidate_images"]),
            self.text_transform(example["cue"]),
            np.isin(example["candidates"], example["associations"]),
        )


def evaluate_winogavil_dataset(
    train_info, data_root=None, num_workers=4, batch_size=None, model_dict=None
):
    if train_info is not None:
        model_arch = train_info["scale_config"]["model"]
        model_path = train_info["checkpoint"]
        model, transform, device = create_model(model_arch, model_path)
        tokenizer = open_clip.get_tokenizer(model_arch)
    elif model_dict is not None:
        model, transform, device, tokenizer = model_dict["model"], model_dict["transform"], model_dict["device"], model_dict["tokenizer"]
    else:
        raise ValueError("Either train_info or model_dict is required.")

    # Load data
    dataset = WinoDataset(
        datasets.load_dataset(
            "nlphuji/winogavil",
            split="test",
            trust_remote_code=True,
            cache_dir=os.path.join(data_root, "hf_cache")
            if data_root is not None
            else None,
        ),
        transform=lambda imgs: torch.stack([transform(img) for img in imgs]),
        text_transform=lambda text: tokenizer([get_clip_prompt(text)]),
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=lambda batch: batch[0],
    )

    all_groups = []
    all_scores = []

    # Iterate WinoGAViL Instances
    for idx, (images, text, y_true) in enumerate(tqdm(dataloader)):
        # Get example
        n_images = len(images)
        n_assoc = y_true.sum()
        # Featurize
        with torch.no_grad(), torch.amp.autocast('cuda'):
            image_features = model.encode_image(images.to(device))
            text_features = model.encode_text(text.to(device))
            # Compute similarities
            image_logits = (text_features @ image_features.T).squeeze(0).cpu().numpy()
        # Select topk
        topk_indices = np.argsort(image_logits)[-n_assoc:]
        y_pred = np.isin(np.arange(n_images), topk_indices)

        # Evaluate with Jaccard
        score = jaccard_score(y_true, y_pred)
        all_scores.append(score)
        all_groups.append(n_images)

        if idx > 0 and idx % 100 == 0:
            print(f"idx: {idx}, current Jaccard index average: {np.mean(all_scores)}")

    all_groups = np.array(all_groups)
    all_scores = np.array(all_scores)
    return {
        "avg_jaccard_score": all_scores.mean(),
        "jaccard_score_5": all_scores[all_groups == 5].mean(),
        "jaccard_score_6": all_scores[all_groups == 6].mean(),
        "jaccard_score_10": all_scores[all_groups == 10].mean(),
        "jaccard_score_12": all_scores[all_groups == 12].mean(),
        "jaccard_score_5-6": all_scores[all_groups <= 6].mean(),
        "jaccard_score_10-12": all_scores[all_groups >= 10].mean(),
    }

def get_clip_prompt(item):
    item = item.lower()
    vowels = ["a", "e", "i", "o", "u"]
    if item[0] in vowels:
        clip_txt = f"An {item}"
    else:
        clip_txt = f"A {item}"
    return clip_txt
