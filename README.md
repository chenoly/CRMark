## 📝 Introduction

**CRMark: Cover-Recoverable Watermark**, a robust and reversible invisible image watermarking method. CRMark enables perfect reconstruction of the original cover image in lossless channels and robust watermark extraction in lossy channels.

CRMark leverages an **Integer Invertible Watermark Network (iIWN)** to achieve lossless and invertible mapping between cover-watermark pairs and stego images. It addresses the trade-off between robustness and reversibility in traditional robust reversible watermarking methods, offering significant improvements in robustness, visual quality, and computational efficiency.

Key features:
- **Robustness**: Enhanced against distortions through an Encoder-Noise Layer-Decoder framework.
- **Reversibility**: Ensures lossless recovery of both the cover image and the watermark in lossless channel.
- **Efficiency**: Reduces time complexity and auxiliary bitstream length.

---
## 🚀 Usage
```bash
pip install crmark==0.1.3
```

code
```bash
import os
import random
import string
import numpy as np
from PIL import Image
from crmark import CRMark

# Create output directory if not exists
os.makedirs("images", exist_ok=True)

# Initialize CRMark in color mode
crmark = CRMark()


# Generate a random string of length 3 (total 24 bits)
def generate_random_string(n: int) -> str:
    characters = string.ascii_letters + string.digits
    return ''.join(random.choices(characters, k=n))


# Calculate PSNR between two images
def calculate_psnr(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    PIXEL_MAX = 255.0
    psnr = 20 * np.log10(PIXEL_MAX / np.sqrt(mse))
    return psnr


# Random string message
str_data = generate_random_string(5)
str_data = "hello"
print(str_data)

# Define image paths
cover_path = "realflow_compare/image_11.png"
# cover_path = "images/cover_color.png"
rec_cover_path = "images/rec_color_cover.png"
attack_rec_cover_path = "images/attack_rec_color_cover.png"
stego_path_clean = "images/color_stego_clean.png"
stego_path_attacked = "images/color_stego_attacked.png"
residual_path_clean = "images/color_residual.png"

# === Case 1: Without attack ===
# Encode string into image
cover_image = np.float32(Image.open(cover_path).resize((256, 256)))
success, stego_image = crmark.encode(cover_image, str_data)
print("psnr:", calculate_psnr(np.float32(stego_image), cover_image))

if success:
    stego_image.save(stego_path_clean)
    stego_clean_image = np.float32(stego_image)
    residual = np.abs((stego_clean_image - cover_image) * 10.) + 127.5
    Image.fromarray(np.uint8(np.clip(residual, 0, 255))).save(residual_path_clean)

    # Recover cover and message from clean image
    stego_clean_image = np.float32(Image.open(stego_path_clean))
    is_attacked_clean, rec_cover_clean, rec_message_clean = crmark.recover(stego_clean_image)
    is_decoded, extracted_message_clean = crmark.decode(stego_clean_image)
    rec_cover_clean.save(rec_cover_path)

    # Compute pixel difference between original and recovered cover
    cover = np.float32(Image.open(cover_path).resize((256, 256)))
    rec_clean = np.float32(rec_cover_clean)
    diff_clean = np.sum(np.abs(cover - rec_clean))

    # === Case 2: With attack ===
    # Slightly modify the image to simulate attack
    stego = np.float32(Image.open(stego_path_clean))
    H, W, C = stego.shape
    rand_y = random.randint(0, H - 1)
    rand_x = random.randint(0, W - 1)
    rand_c = random.randint(0, C - 1)

    # Apply a small perturbation (±1)
    perturbation = random.choice([-1, 1])
    stego[rand_y, rand_x, rand_c] = np.clip(stego[rand_y, rand_x, rand_c] + perturbation, 0, 255)

    stego = np.clip(stego + np.random.randint(0, 1, size=stego.shape), 0, 255)

    Image.fromarray(np.uint8(stego)).save(stego_path_attacked)

    # Recover from attacked image
    stego_attacked_image = np.float32(Image.open(stego_path_attacked))
    is_attacked, rec_cover_attacked, rec_message_attacked = crmark.recover(stego_attacked_image)
    is_attacked_flag, extracted_message_attacked = crmark.decode(stego_attacked_image)
    rec_cover_attacked.save(attack_rec_cover_path)

    rec_attacked = np.float32(rec_cover_attacked)
    diff_attacked = np.sum(np.abs(cover - rec_attacked))

    # === Print results ===
    print("=== Without Attack ===")
    print("Original Message:", str_data)
    print("Recovered Message:", rec_message_clean)
    print("Extracted Message:", extracted_message_clean)
    print("Was Attacked:", is_attacked_clean)
    print("L1 Pixel Difference:", diff_clean)

    print("\n=== With Attack ===")
    print("Recovered Message:", rec_message_attacked)
    print("Extracted Message:", extracted_message_attacked)
    print("Was Attacked:", is_attacked)
    print("L1 Pixel Difference:", diff_attacked)

else:
    print("Emebdding failed!")





```

🚀 Training
You can train the models used in the paper with the following commands, which apply common distortion layers:
'''bash
python train_color.py
'''
'''bash
python train_gray.py
'''
For more complex distortions, use the following command:
'''bash
python train_more.py
'''
## 🚀 WatermarkLab

For evaluation and distortion layers, we utilize **WatermarkLab**, a comprehensive watermarking evaluation framework. WatermarkLab provides a wide range of distortion simulations and evaluation metrics to rigorously test the robustness and performance of watermarking algorithms.

You can find the WatermarkLab repository here: [github/chenoly/watermarklab](https://github.com/chenoly/watermarklab)

---

## ⚠️ Note  
When using CRMark, keep the following considerations in mind:

Training with Complex Distortions: Training CRMark on more complex distortions may result in an excessively large norm of the latent variable z, increasing the number of bits required to encode each value in the dependent variable. To address this, CRMark provides a parameter freq_bits_len during initialization to specify the encoding bit length. Setting a longer freq_bits_len can resolve this issue; otherwise, recovery may fail.

Pre-trained Model Characteristics: The watermarks in the provided pre-trained models are generated through random sampling. As a result, these models exhibit strong generalization for randomly sampled watermarks but weaker generalization for watermarks with added error-correcting codes. For practical applications, consider incorporating error-correcting codes into the watermark during training to enhance the robustness of the iIWN model.



our preprint paper:  

```
@article{chen2024drrw,
  title={Deep Robust Reversible Watermarking},
  author={Jiale Chen and Wei Wang and Chongyang Shi and Li Dong and Yuanman Li and Xiping Hu},
  journal={arXiv preprint arXiv:2503.02490},
  year={2024}
}
```
