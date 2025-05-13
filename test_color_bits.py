import os
import random
import numpy as np
from PIL import Image
from crmark import CRMark

# Create output directory
os.makedirs("images", exist_ok=True)

# Initialize CRMark for color image, using 64-bit message
crmark = CRMark(model_mode="color_256_100", float64=False)

# Generate random 64-bit binary message
watermark = [random.randint(0, 1) for _ in range(100)]

# Define image paths
cover_path = "images/color_cover.png"
stego_path_clean = "images/color_stego_clean.png"
stego_path_attacked = "images/color_stego_attacked.png"

# === Case 1: Without attack ===
# Encode watermark
success, stego_image = crmark.encode_bits(cover_path, watermark)
stego_image.save(stego_path_clean)

# Recover without attack
is_attacked_clean, rec_cover_clean, rec_message_clean = crmark.recover_bits(stego_path_clean)
ext_bits_clean = crmark.decode_bits(stego_path_clean)

# Compare cover and recovered cover
cover = np.float32(Image.open(cover_path))
rec_clean = np.float32(rec_cover_clean)
diff_clean = np.sum(np.abs(cover - rec_clean))

# === Case 2: With attack ===
# Slightly modify one pixel to simulate distortion
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
is_attacked, rec_cover_attacked, rec_message_attacked = crmark.recover_bits(stego_path_attacked)
ext_bits_attacked = crmark.decode_bits(stego_path_attacked)

rec_attacked = np.float32(rec_cover_attacked)
diff_attacked = np.sum(np.abs(cover - rec_attacked))

# === Print results ===
print("=== Without Attack ===")
print("Attacked:", is_attacked_clean)
print("Recovered Watermark:", rec_message_clean)
print("Extracted Watermark:", ext_bits_clean)
print("L1 Difference:", diff_clean)

print("\n=== With Attack ===")
print("Attacked:", is_attacked)
print("Recovered Watermark:", rec_message_attacked)
print("Extracted Watermark:", ext_bits_attacked)
print("L1 Difference:", diff_attacked)
