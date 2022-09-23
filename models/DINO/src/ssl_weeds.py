import os
from typing import Callable, Tuple
from torch.utils.data import Dataset
import numpy as np
from PIL import Image

def read_rgb(file_path: str) -> np.ndarray:
    if not os.path.exists(file_path):
        raise ValueError(f"The path {file_path} does not exist")
    image = Image.open(file_path).convert("RGB")
    image = np.array(image)
    return image

class SSLWEEDS(Dataset):
    
    CLASSES = {
        1: "Crop",
        2: "Other",
        3: "Weed"
    }
    
    def __init__(
        self,
        root: str, 
        train: bool,
        transform: Callable = None
    ) -> None:
        """Self Supervised Weed Dataset support

        Args:
            root (str): data dir
            train (bool): if True unlabeled data is provided, else test data.
            transform (Callable, optional): self-sup transform function. Defaults to None.
        """
        if not os.path.exists(root):
            print(f"{root} does not exists. Quitting.")
            quit()
        
        self.data_dir = os.path.join(root, "train" if train else "val")
        #if train:
        #    self.img_paths = [os.path.join(self.data_dir, f) for f in os.listdir(self.data_dir) if not f.startswith(".")]
        #    self.targets = None
        if train:
            classes = [c for c in os.listdir(self.data_dir) if not c.startswith(".")]
            self.targets = None
            self.img_paths = []
            for c_i, c in enumerate(classes):
                c_dir = os.path.join(self.data_dir, c)
                for f in os.listdir(c_dir):
                    if not f.startswith("."):
                        self.img_paths.append(os.path.join(self.data_dir, c, f))
        else:
            classes = [c for c in os.listdir(self.data_dir) if not c.startswith(".")]
            self.targets = []
            self.img_paths = []
            self.labels = []
            for c_i, c in enumerate(classes):
                c_dir = os.path.join(self.data_dir, c)
                for f in os.listdir(c_dir):
                    if not f.startswith("."):
                        self.img_paths.append(os.path.join(self.data_dir, c, f))
                        self.targets.append(c_i)
                        self.labels.append(c)
        self.transform = transform
        
    def __getitem__(self, index) -> Tuple:
        
        img_path = self.img_paths[index]
        if self.targets is not None:
            label = self.targets[index]
        else:
            label = 0 # just to keep collate_fn simple
            
        img = read_rgb(img_path)
        views = None
        if self.transform:
            #img, views = self.transform(img)
            img = self.transform(img)

        return img, views, label
    
    def __len__(self) -> int:
        return len(self.img_paths)
        