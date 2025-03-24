import torch
from torch.utils.data import DataLoader
from model import CycleDiffusion
from dataset import ImagePairDataset

# データパス（照明あり / 照明なし など）
data_A = "back_data/0000"  # 明るい画像
data_B = "back_data/0100"  # 暗い画像

dataset = ImagePairDataset(data_A, data_B)
loader = DataLoader(dataset, batch_size=4, shuffle=True)

model = CycleDiffusion(device="cuda")
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

for epoch in range(100):
    for i, (img_A, img_B) in enumerate(loader):
        img_A = img_A.cuda()
        img_B = img_B.cuda()

        loss = model(img_A, img_B)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if i % 10 == 0:
            print(f"Epoch {epoch} Step {i} Loss: {loss.item():.4f}")

torch.save(model.unet_B2A.state_dict(), "unet_B2A.pth")
