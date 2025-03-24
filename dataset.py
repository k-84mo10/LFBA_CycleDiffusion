from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image
import os

class ImagePairDataset(Dataset):
    def __init__(self, dir_A, dir_B, size=256):
        self.files_A = sorted([os.path.join(dir_A, f) for f in os.listdir(dir_A)])
        self.files_B = sorted([os.path.join(dir_B, f) for f in os.listdir(dir_B)])

        self.transform = transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return min(len(self.files_A), len(self.files_B))

    def __getitem__(self, idx):
        image_A = self.transform(Image.open(self.files_A[idx]).convert("RGB"))
        image_B = self.transform(Image.open(self.files_B[idx]).convert("RGB"))
        return image_A, image_B
