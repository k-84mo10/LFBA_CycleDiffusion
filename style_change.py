from PIL import Image
import numpy as np
import cv2

# 画像の読み込み
input_path = "input.jpg"
style_path = "style.jpg"
output_path = "stylized_input.jpg"

input_img = cv2.imread(input_path)
style_img = cv2.imread(style_path)

# サイズを揃える
style_img = cv2.resize(style_img, (input_img.shape[1], input_img.shape[0]))

# --- カラースタイルマッチング（色分布を style_img に合わせる） ---
# 色空間を Lab に変換
input_lab = cv2.cvtColor(input_img, cv2.COLOR_BGR2Lab).astype(np.float32)
style_lab = cv2.cvtColor(style_img, cv2.COLOR_BGR2Lab).astype(np.float32)

# 各チャンネルごとの平均と標準偏差を計算
for i in range(3):
    input_mean, input_std = input_lab[:,:,i].mean(), input_lab[:,:,i].std()
    style_mean, style_std = style_lab[:,:,i].mean(), style_lab[:,:,i].std()
    input_lab[:,:,i] = (input_lab[:,:,i] - input_mean) / (input_std + 1e-5)
    input_lab[:,:,i] = input_lab[:,:,i] * style_std + style_mean

# BGR に戻して保存
output_img = cv2.cvtColor(np.clip(input_lab, 0, 255).astype(np.uint8), cv2.COLOR_Lab2BGR)
cv2.imwrite(output_path, output_img)

print("カラースタイル転送完了 →", output_path)


from diffusers import StableDiffusionImg2ImgPipeline
import torch
from PIL import Image

pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16
).to("cuda")

prompt = "a photo of a room with soft, white lighting, low contrast, evenly lit"
image = Image.open("stylized_input.jpg").convert("RGB").resize((512, 512))

result = pipe(
    prompt=prompt,
    image=image,
    strength=0.25,  # 構図を維持
    guidance_scale=7.5
).images[0]

result.save("final_result.jpg")
print("完成！")
