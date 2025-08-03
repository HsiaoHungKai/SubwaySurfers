import os
import matplotlib.pyplot as plt
import torch


class DepthDataset(torch.utils.data.Dataset):
    def __init__(self, images_dir, depth_maps_dir, transform=None):
        self.images_dir = images_dir
        self.depth_maps_dir = depth_maps_dir
        self.transform = transform

        all_image_files = [
            f for f in os.listdir(images_dir) if f.lower().endswith(".png")
        ]
        self.files = []
        for image_file in all_image_files:
            depth_file_path = os.path.join(depth_maps_dir, image_file)
            if os.path.exists(depth_file_path):
                self.files.append(image_file)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        image_path = os.path.join(self.images_dir, self.files[idx])
        depth_map_path = os.path.join(self.depth_maps_dir, self.files[idx])

        image = plt.imread(image_path)
        depth_map = plt.imread(depth_map_path)[:, :, 0]

        if self.transform:
            image = self.transform(image)
            depth_map = self.transform(depth_map)

        return image, depth_map
