import os
import random
import numpy as np
from PIL import Image
from crmark import CRMark

# Create output directory if not exists
os.makedirs("images", exist_ok=True)

# Initialize CRMark in color mode
crmark = CRMark(model_mode="color_256_100", float64=False)


# Calculate PSNR between two images
def calculate_psnr(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    PIXEL_MAX = 255.0
    psnr = 20 * np.log10(PIXEL_MAX / np.sqrt(mse))
    return psnr


# Generate random 64-bit binary message
watermark = [random.randint(0, 1) for _ in range(100)]

# Define image paths
cover_path = "images/color_cover.png"
rec_cover_path = "images/rec_color_cover.png"
stego_path_clean = "images/color_stego_clean.png"
stego_path_attacked = "images/color_stego_attacked.png"

# === Case 1: Without attack ===
# Encode string into image
cover_image = np.float32(Image.open(cover_path))
success, stego_image = crmark.encode_bits(cover_image, watermark)
stego_image.save(stego_path_clean)

# Recover cover and message from clean image
stego_clean_image = np.float32(Image.open(stego_path_clean))
is_attacked_clean, rec_cover_clean, rec_message_clean = crmark.recover_bits(stego_clean_image)
extracted_message_clean = crmark.decode_bits(stego_clean_image)
rec_cover_clean.save(rec_cover_path)

# Compute pixel difference between original and recovered cover
cover = np.float32(Image.open(cover_path))
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
Image.fromarray(np.uint8(stego)).save(stego_path_attacked)

# Recover from attacked image
stego_attacked_image = np.float32(Image.open(stego_path_attacked))
is_attacked, rec_cover_attacked, rec_message_attacked = crmark.recover_bits(stego_attacked_image)
extracted_message_attacked = crmark.decode_bits(stego_attacked_image)

rec_attacked = np.float32(rec_cover_attacked)
diff_attacked = np.sum(np.abs(cover - rec_attacked))

# === Print results ===
print("=== Without Attack ===")
print("Original Message:", watermark)
print("Recovered Message:", rec_message_clean)
print("Extracted Message:", extracted_message_clean)
print("Is Attacked:", is_attacked_clean)
print("L1 Pixel Difference:", diff_clean)

print("\n=== With Attack ===")
print("Recovered Message:", rec_message_attacked)
print("Extracted Message:", extracted_message_attacked)
print("Is Attacked:", is_attacked)
print("L1 Pixel Difference:", diff_attacked)
