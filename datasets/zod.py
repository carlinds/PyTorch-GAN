import os
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
from typing import List
from tqdm import tqdm


class ZOD(Dataset):
    def __init__(
        self,
        datalist: str,
        feature_cache_root: str,
        img_size: int = 1920,
        validate_paths: bool = False,
    ):
        self.img_size = img_size
        self.datalist = datalist
        self.feature_cache_root = feature_cache_root

        with open(self.datalist, "r") as f:
            self.image_paths = [p.strip() for p in f.readlines()]
        self.feature_paths = [
            os.path.join(
                self.feature_cache_root, os.path.basename(p).split("_")[0] + ".pt"
            )
            for p in self.image_paths
        ]

        # Check that all files exist
        if validate_paths:
            for ip, fp in tqdm(
                zip(self.image_paths, self.feature_paths), desc="Validating image paths"
            ):
                assert os.path.exists(ip), f"File {ip} does not exist."
                assert os.path.exists(fp), f"File {fp} does not exist."

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = torch.from_numpy(self.load_image(idx))
        feature = torch.load(self.feature_paths[idx])
        return img, feature

    @staticmethod
    def collate_fn(batch):
        imgs, features = zip(*batch)

        return torch.stack(imgs, 0), torch.stack(features, 0)

    def load_image(self, idx):
        img_path = self.image_paths[idx]
        img = cv2.imread(img_path)
        assert img is not None, f"Image {img_path} not found."

        h0, w0 = img.shape[:2]
        r = self.img_size / max(h0, w0)
        img = img[
            int((h0 - h0 * r) / 2) : int((h0 + h0 * r) / 2),
            int((w0 - w0 * r) / 2) : int((w0 + w0 * r) / 2),
            :,
        ]

        return img


if __name__ == "__main__":
    zod = ZOD(
        "zod/zod_full_original_minival.txt",
        "/staging/agp/datasets/zod_feature_cache/val",
        validate_paths=True,
    )

    dataloader = DataLoader(
        zod, batch_size=2, shuffle=False, num_workers=0, collate_fn=ZOD.collate_fn
    )
    for img, feature in dataloader:
        print(img.shape)
        print(feature.shape)
        break
