import os
import random
import numpy as np
from PIL import Image
from crmark import CRMark

# Create the output directory if not exists
os.makedirs("images", exist_ok=True)

# Initialize CRMark in grayscale mode
crmark = CRMark(model_mode="gray_512_256", float64=False)

# Message to embed
message = "CRMark: Hide&Recover"

# Define file paths
cover_path = "images/gray_cover.png"
rec_cover_path = "images/rec_gray_cover.png"
stego_path_clean = "images/gray_stego_clean.png"
stego_path_attacked = "images/gray_stego_attacked.png"

# === Case 1: Without attack ===
# Encode message into grayscale image
success, stego_image = crmark.encode(cover_path, message)
stego_image.save(stego_path_clean)

# Decode and recover from clean stego image
is_attacked_clean, rec_cover_clean, rec_message_clean = crmark.recover(stego_path_clean)
is_decoded_clean, ext_message_clean = crmark.decode(stego_path_clean)
rec_cover_clean.save(rec_cover_path)

# Compute L1 pixel difference
cover = np.float32(Image.open(cover_path))
rec_clean = np.float32(rec_cover_clean)
diff_clean = np.sum(np.abs(cover - rec_clean))

# === Case 2: With attack ===
# Apply slight modification to simulate attack
stego = np.float32(Image.open(stego_path_clean))
H, W = stego.shape
rand_y = random.randint(0, H - 1)
rand_x = random.randint(0, W - 1)

# Apply a small perturbation (±1)
perturbation = random.choice([-1, 1])
stego[rand_y, rand_x] = np.clip(stego[rand_y, rand_x] + perturbation, 0, 255)
Image.fromarray(np.uint8(stego)).save(stego_path_attacked)

# Decode and recover from attacked image
is_attacked, rec_cover_attacked, rec_message_attacked = crmark.recover(stego_path_attacked)
is_decoded, ext_message_attacked = crmark.decode(stego_path_attacked)

rec_attacked = np.float32(rec_cover_attacked)
diff_attacked = np.sum(np.abs(cover - rec_attacked))

# === Print results ===
print("=== Without Attack ===")
print("Original Message:", message)
print("Recovered Message:", rec_message_clean)
print("Extracted Message:", ext_message_clean)
print("Is Attacked:", is_attacked_clean)
print("L1 Pixel Difference:", diff_clean)

print("\n=== With Attack ===")
print("Recovered Message:", rec_message_attacked)
print("Extracted Message:", ext_message_attacked)
print("Is Attacked:", is_attacked)
print("L1 Pixel Difference:", diff_attacked)
