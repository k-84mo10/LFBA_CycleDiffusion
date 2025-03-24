import torch
import torch.nn.functional as F
from diffusers import UNet2DConditionModel, DDPMScheduler, AutoencoderKL
from transformers import CLIPTextModel, CLIPTokenizer

class CycleDiffusion(torch.nn.Module):
    def __init__(self, device="cuda"):
        super().__init__()
        self.device = device

        # テキストエンコーダー（今回は使わないが汎用性のために残す）
        self.tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
        self.text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14").to(device)

        # 拡散モデル本体（UNet）
        self.unet_A2B = UNet2DConditionModel.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="unet").to(device)
        self.unet_B2A = UNet2DConditionModel.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="unet").to(device)

        # スケジューラー（ノイズの加減）
        self.scheduler = DDPMScheduler.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="scheduler")

        self.vae = AutoencoderKL.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="vae").to(device)


    def forward(self, image_A, image_B):
        # Diffusion step（例：50ステップ）
        timesteps = torch.randint(0, self.scheduler.config.num_train_timesteps, (image_A.shape[0],), device=self.device)

        # ① テキストエンコーダを使ってダミー入力を作成
        dummy_text = [""] * image_A.size(0)
        inputs = self.tokenizer(dummy_text, padding="max_length", max_length=77, return_tensors="pt").to(self.device)
        text_embeddings = self.text_encoder(**inputs).last_hidden_state

        # ② latentに変換
        latent_A = self.vae.encode(image_A).latent_dist.sample() * 0.18215
        latent_B = self.vae.encode(image_B).latent_dist.sample() * 0.18215

        # ③ A → B
        noise = torch.randn_like(latent_A)
        noisy_A = self.scheduler.add_noise(latent_A, noise, timesteps)
        pred_noise_B = self.unet_A2B(noisy_A, timesteps, encoder_hidden_states=text_embeddings).sample

        # ④ B → A
        noise_B = torch.randn_like(latent_B)
        noisy_B = self.scheduler.add_noise(latent_B, noise_B, timesteps)
        pred_noise_A = self.unet_B2A(noisy_B, timesteps, encoder_hidden_states=text_embeddings).sample

        # ⑤ 損失
        loss_cycle_A = F.mse_loss(pred_noise_A, noise)
        loss_cycle_B = F.mse_loss(pred_noise_B, noise_B)

        return loss_cycle_A + loss_cycle_B

