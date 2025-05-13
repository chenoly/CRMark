import os
import re
import string

import torch
import lpips
import bchlib
import kornia
import hashlib
import numpy as np
from torch import nn, Tensor


class LPIPSLoss(nn.Module):
    def __init__(self, ):
        super().__init__()
        self.lpips = lpips.LPIPS(net="alex", verbose=False)

    def __call__(self, input_image, target_image):
        if input_image.shape[1] == 1:
            input_image = input_image.repeat(1, 3, 1, 1)
            target_image = target_image.repeat(1, 3, 1, 1)
        normalized_input = input_image.clamp(0, 1.) * 2 - 1
        normalized_encoded = target_image.clamp(0, 1.) * 2 - 1
        lpips_loss = self.lpips(normalized_input, normalized_encoded).mean()
        return lpips_loss


# Stochastic Round class for stochastic rounding operations
class StochasticRound:
    def __init__(self, scale=1 / 255.):
        """
        Initializes the Stochastic Round operation.

        Parameters:
            scale (float): The scaling factor for the rounding operation. Default is 1/255.
        """
        super().__init__()
        self.scale = scale

    def __call__(self, x, hard_round):
        """
        Perform stochastic rounding on the input tensor.

        Parameters:
            x (Tensor): Input tensor to be rounded.
            hard_round (bool): Whether to use hard rounding or soft rounding.

        Returns:
            Tensor: The rounded tensor.
        """
        # Scale the input by the defined scaling factor
        scale_x = x / self.scale
        # Perform the rounding operation
        round_out = scale_x + (torch.round(scale_x) - scale_x).detach()
        out = round_out + torch.rand_like(x) - 0.5  # Add noise for stochastic rounding
        if hard_round:
            return round_out * self.scale  # Return scaled result for hard rounding
        return out * self.scale  # Return original tensor if no rounding is performed


# Penalty loss class to compute penalties for overflow pixels
class PenalityLoss(nn.Module):
    def __init__(self, max_value=1.):
        """
        Initializes the Penalty Loss for overflow pixels.

        Parameters:
            max_value (float): Maximum allowable pixel value (default is 1).
        """
        super().__init__()
        self.max_value = max_value
        # self.MSE = nn.MSELoss(reduce=True, size_average=False)
        self.MSE = nn.MSELoss(reduce=True)

    def __call__(self, input_tensor):
        """
        Computes the penalty loss for pixels that overflow the allowable range.

        Parameters:
            input_tensor (Tensor): Input tensor to compute penalty loss on.

        Returns:
            Tensor: The penalty loss value.
        """
        # Calculate the penalty for pixels below 0
        loss_0 = self.MSE(torch.relu(-input_tensor), torch.zeros_like(input_tensor))
        # Calculate the penalty for pixels above max_value
        loss_255 = self.MSE(torch.relu(input_tensor - self.max_value), torch.zeros_like(input_tensor))
        # Total penalty loss is the sum of both losses
        loss = loss_0 + loss_255
        return loss


# Normalize function to scale image tensor to [0, 1] range
def normalize(input_image):
    """
    Normalize the input image tensor to the range [0, 1].

    Parameters:
        input_image (Tensor): Input image tensor to normalize.

    Returns:
        Tensor: The normalized image tensor.
    """
    min_vals = input_image.amin(dim=(1, 2, 3), keepdim=True)
    max_vals = input_image.amax(dim=(1, 2, 3), keepdim=True)
    normalized_img = (input_image - min_vals) / (max_vals - min_vals + 1e-5)  # Prevent division by zero
    return normalized_img


# Function to extract the accuracy of a secret image
def extract_accuracy(ext_secret, secret, max_value=1.):
    """
    Extracts the accuracy of a secret image by comparing it to the expected secret.

    Parameters:
        ext_secret (Tensor): The extracted secret image.
        secret (Tensor): The ground truth secret image.

    Returns:
        float: The accuracy value.

    Parameters
    ----------
    secret
    ext_secret
    max_value
    """
    acc = 1.0 - (torch.abs(torch.round(ext_secret.clamp(0., max_value)) - secret).mean())
    return acc.item()


# Function to calculate the number of overflow pixels in a stego image
def overflow_num(stego, mode, min_value=0., max_value=1.):
    """
    Calculate the number of overflow pixels in the stego image.

    Parameters:
        stego (Tensor): The stego image tensor.
        mode (int): The overflow mode (0 for below min_value, 255 for above max_value).
        min_value (float): The minimum allowed pixel value (default is 0).
        max_value (float): The maximum allowed pixel value (default is 1).

    Returns:
        float: The average overflow pixel count.
    """
    assert mode in [0, 255]
    if mode == 0:
        overflow_pixel_n = torch.sum(StochasticRound()(stego, True) < min_value, dim=(1, 2, 3)).float().mean()
    else:
        overflow_pixel_n = torch.sum(StochasticRound()(stego, True) > max_value, dim=(1, 2, 3)).float().mean()
    return overflow_pixel_n.item()


# Function to compute the PSNR between the input and target images
def compute_psnr(input_image, target_image, max_value=1.):
    """
    Compute the Peak Signal-to-Noise Ratio (PSNR) between two images.

    Parameters:
        input_image (Tensor): The input image tensor.
        target_image (Tensor): The target image tensor.
        max_value (float): The maximum allowable pixel value (default is 1).

    Returns:
        float: The average PSNR value.
    """
    # Apply stochastic rounding and clamp images to the range [0, 1]
    input_image = StochasticRound()(input_image.clamp(0., 1.), True)
    target_image = StochasticRound()(target_image.clamp(0., 1.), True)
    average_psnr = kornia.metrics.psnr(input_image, target_image, max_value).mean()
    return average_psnr.item()


def quantize_image(input_image):
    """
    Quantize the input image using stochastic rounding.

    Parameters:
        input_image (Tensor): Input image tensor with values between 0 and 1.

    Returns:
        Tensor: The quantized image tensor after applying stochastic rounding.
    """
    # Apply stochastic rounding to the input image, ensuring the values are within the [0, 1] range.
    input_image = StochasticRound()(input_image.clamp(0., 1.), True)
    return input_image


def quantize_residual_image(input_image, target_image):
    """
    Quantize the residual image, which is the difference between the input and target image.

    Parameters:
        input_image (Tensor): The input image tensor.
        target_image (Tensor): The target image tensor to compare against.

    Returns:
        Tensor: The quantized residual image after applying stochastic rounding.
    """
    # Calculate the residual (difference) between the input image and the target image.
    # Then normalize the residual and apply stochastic rounding.
    shift = (input_image - target_image) * 10.
    input_image = StochasticRound()(normalize(shift), True)
    return input_image


def find_latest_model(model_dir):
    """
    Find the .pth file with the largest epoch number in the given directory.

    Parameters:
        model_dir (str): Path to the directory containing model files.

    Returns:
        str: Path to the latest model file, or None if no .pth files are found.
    """
    # Initialize variables to track the maximum epoch and corresponding model file.
    max_epoch = -1
    latest_model_path = None
    # Regular expression pattern to match the file format: "model_{epoch}.pth"
    pattern = re.compile(r"model_(\d+)\.pth")
    # Iterate through files in the given directory.
    for file_name in os.listdir(model_dir):
        # Check if the file name matches the pattern.
        match = pattern.match(file_name)
        if match:
            # Extract the epoch number from the file name.
            epoch = int(match.group(1))
            # Update the maximum epoch and the path to the latest model if necessary.
            if epoch > max_epoch:
                max_epoch = epoch
                latest_model_path = os.path.join(model_dir, file_name)
    return latest_model_path


def sha256_of_image_array(img_array):
    """
    Compute the SHA-256 hash of a NumPy image array.

    Parameters:
    img_array (np.ndarray): The input image array.

    Returns:
    str: A 64-character hexadecimal SHA-256 hash string.
    """
    # Convert the image array to a raw byte sequence
    img_bytes = img_array.tobytes()
    # Compute the SHA-256 hash of the byte sequence and return the hexadecimal representation
    sha256_hash = hashlib.sha256(img_bytes).hexdigest()
    return sha256_hash


def sha256_to_bitstream(sha256_str):
    """
    Convert a 64-character hexadecimal SHA-256 string to a 256-bit binary stream.

    Each hexadecimal character represents 4 bits (since 16 = 2^4), so 64 hex characters equal 256 bits.

    Parameters:
    sha256_str (str): A 64-character hexadecimal SHA-256 hash string.

    Returns:
    np.ndarray: A NumPy array of 256 binary values (dtype=np.uint8), where each element is 0 or 1.
    """
    # Initialize an array of 256 zeros to store the binary bits
    bits = np.zeros(256, dtype=np.uint8)
    # Iterate through each character in the SHA-256 hex string
    for i, hex_char in enumerate(sha256_str):
        # Convert the hex character to its integer value (0 to 15)
        val = int(hex_char, 16)
        # Convert this 4-bit value to binary and store it in the bitstream
        # The bits are stored in big-endian order (most significant bit first)
        for j in range(4):
            bits[i * 4 + (3 - j)] = (val >> j) & 1
    return bits.tolist()


class BCH:
    def __init__(self, BCH_POLYNOMIAL_=501, BCH_BITS_=12):
        self.bch = bchlib.BCH(BCH_BITS_, BCH_POLYNOMIAL_)

    def Encode(self, data_: str):
        data_ = bytearray(data_, 'utf-8')
        ecc = self.bch.encode(data_)
        packet = data_ + ecc
        packet_binary = ''.join(format(x, '08b') for x in packet)
        secret_ = [int(x) for x in packet_binary]
        return secret_

    def Decode(self, secret_: list):
        packet_binary = "".join([str(int(bit)) for bit in secret_])
        packet = bytes(int(packet_binary[i: i + 8], 2) for i in range(0, len(packet_binary), 8))
        packet = bytearray(packet)

        # Ensure correct splitting of data and ECC bytes
        data_, ecc = packet[:-self.bch.ecc_bytes], packet[-self.bch.ecc_bytes:]
        bit_flips = self.bch.decode(data_, ecc)
        if bit_flips != -1:
            self.bch.correct(data_, ecc)
            return True, data_.decode('utf-8')
        else:
            return False, data_.decode('utf-8')


if __name__ == "__main__":
    import random


    def flip_n_bits(bits: list, n: int) -> list:
        flipped = bits.copy()
        bit_indices = random.sample(range(len(bits)), n)
        for idx in bit_indices:
            flipped[idx] ^= 1
        return flipped


    def generate_random_string(n: int) -> str:
        characters = string.ascii_letters + string.digits
        return ''.join(random.choices(characters, k=n))


    bch_codec = BCH()
    original_data = generate_random_string(20)
    print("Original data:", original_data)
    secret = bch_codec.Encode(original_data)
    print(len(secret))
    n = 0
    noisy_secret = flip_n_bits(secret, n)
    print(noisy_secret)
    print("Encoded bits:", secret[:64], "...")
    isdecode, decoded_data = bch_codec.Decode(noisy_secret)
    if isdecode:
        print("Decoded data:", decoded_data)
        print("Success!" if decoded_data == original_data else "Mismatch!")
    else:
        print("Decoding failed.")
