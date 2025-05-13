import os
import random
import numpy as np
from PIL import Image
from crmark import CRMark

# Ensure the output directory exists
os.makedirs("images", exist_ok=True)

# Initialize CRMark instance (gray mode, float32 precision)
crmark = CRMark(model_mode="gray_512_256", float64=False)

# Generate a random 256-bit binary watermark
watermark = [random.randint(0, 1) for _ in range(256)]

# Paths
cover_path = "images/gray_cover.png"
stego_path_clean = "images/gray_stego_clean.png"
stego_path_attacked = "images/gray_stego_attacked.png"

# === Case 1: Without attack ===
# Embed watermark into the cover image
success, stego_image = crmark.encode_bits(cover_path, watermark)
stego_image.save(stego_path_clean)

# Recover from clean stego image
is_attacked_clean, rec_cover_clean, rec_message_clean = crmark.recover_bits(stego_path_clean)
ext_bits_clean = crmark.decode_bits(stego_path_clean)

# Compute difference between original and recovered image
cover_img = np.float32(Image.open(cover_path))
rec_img_clean = np.float32(rec_cover_clean)
diff_clean = np.sum(np.abs(cover_img - rec_img_clean))

# === Case 2: With attack ===
# Slightly modify one pixel to simulate attack
stego = np.float32(Image.open(stego_path_clean))
H, W = stego.shape
rand_y = random.randint(0, H - 1)
rand_x = random.randint(0, W - 1)

# Apply a small perturbation (±1)
perturbation = random.choice([-1, 1])
stego[rand_y, rand_x] = np.clip(stego[rand_y, rand_x] + perturbation, 0, 255)
Image.fromarray(np.uint8(stego)).save(stego_path_attacked)

# Recover from attacked stego image
is_attacked, rec_cover_attacked, rec_message_attacked = crmark.recover_bits(stego_path_attacked)
ext_bits_attacked = crmark.decode_bits(stego_path_attacked)

# Compute difference between original and attacked-recovered image
rec_img_attacked = np.float32(rec_cover_attacked)
diff_attacked = np.sum(np.abs(cover_img - rec_img_attacked))

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
