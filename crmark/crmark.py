import os
import torch
import random
import platform
import requests
import numpy as np
from PIL import Image
from tqdm import tqdm
from crmark.nets import Model
from torchvision import transforms
from typing import Tuple, Union, Any
from crmark.compressor.rdh import CustomRDH
from crmark.compressor.utils import sha256_of_image_array
from crmark.compressor.utils_compressors import TensorCoder
from crmark.compressor.utils import sha256_to_bitstream, BCH
import warnings

# Suppress warnings to keep console output clean
warnings.filterwarnings("ignore")

# Define module exports
__all__ = ["CRMark"]

# URLs for pre-trained model weights
_gray_512_256_MODEL_URL = "https://github.com/chenoly/CRMark/releases/download/v1.0/crmark_gray_size_512_bit_256.pth"
_color_256_64_MODEL_URL = "https://github.com/chenoly/CRMark/releases/download/v1.0/crmark_color_size_256_bit_64.pth"
_color_256_100_MODEL_URL = "https://github.com/chenoly/CRMark/releases/download/v1.0/crmark_color_size_256_bit_100.pth"


def _download_from_github_release(url: str, save_path: str) -> bool:
    """
    Download a file from a GitHub release URL, saving only if complete.

    Parameters:
        url (str): URL of the file to download (e.g., model weights file).
        save_path (str): Local path to save the downloaded file.

    Returns:
        bool: True if download and save are successful, False otherwise.
    """
    temp_path = None
    file_handle = None
    try:
        print("download:", url)
        # Use GitHub token for authentication
        headers = {"Authorization": f"token {os.getenv('GITHUB_TOKEN', '')}"}
        response = requests.get(url, stream=True, allow_redirects=True, headers=headers, timeout=30)

        # Check response status
        if response.status_code != 200:
            print(f"Failed to download from {url}. Status code: {response.status_code}")
            print(f"\nPlease manually download the file from:\n{url} \nand place it in: {os.path.dirname(save_path)}")
            return False

        # Get expected file size
        total_size = int(response.headers.get('content-length', 0))
        if total_size < 1024:
            print(f"Expected file size too small: {total_size} bytes")
            return False

        # Create temporary file path
        temp_path = save_path + '.tmp'
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        # Download to temporary file with progress bar
        with open(temp_path, 'wb') as file_handle:
            with tqdm(
                    desc=save_path,
                    total=total_size,
                    unit='B',
                    unit_scale=True,
                    unit_divisor=1024,
            ) as bar:
                for data in response.iter_content(chunk_size=1024):
                    file_handle.write(data)
                    bar.update(len(data))

        # Verify downloaded size
        downloaded_size = os.path.getsize(temp_path)
        if downloaded_size != total_size:
            print(f"Incomplete download. Downloaded: {downloaded_size} bytes, expected: {total_size} bytes")
            return False

        # Rename temp file to final path
        os.rename(temp_path, save_path)
        print(f"Successfully downloaded and saved to {save_path}")
        return True

    except (requests.RequestException, KeyboardInterrupt) as e:
        print(f"Download failed or interrupted: {e}")
        # Print manual download instructions
        print(f"\nPlease manually download the file from:\n{url}\nand place it in:\n{os.path.dirname(save_path)}")
        return False

    finally:
        # Ensure file handle is closed
        if file_handle is not None:
            try:
                file_handle.close()
            except Exception as e:
                print(f"Failed to close file handle: {e}")

        # Clean up temporary file if it exists
        if temp_path and os.path.exists(temp_path):
            try:
                os.unlink(temp_path)
                print(f"Cleaned up temporary file: {temp_path}")
            except OSError as e:
                print(f"Failed to delete temporary file {temp_path}: {e}")


def _download_models(model_mode):
    """
    Download pre-trained model weights for the specified color mode if not already cached.

    Parameters:
        model_mode (str): The image mode ("gray" for grayscale, "color" for RGB).

    Returns:
        None: The function downloads and caches the model weights if necessary.
    """
    # Determine the cache directory based on the operating system
    if platform.system() == "Windows":
        base_cache_dir = os.path.join(os.environ["USERPROFILE"], ".cache", "crmark")
    else:
        base_cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "crmark")
    os.makedirs(base_cache_dir, exist_ok=True)

    # Download grayscale model if needed
    if model_mode == "gray_512_256":
        gray_model_path = os.path.join(base_cache_dir, "crmark_gray_size_512_bit_256.pth")
        if not os.path.exists(gray_model_path):
            _download_from_github_release(_gray_512_256_MODEL_URL, gray_model_path)

    # Download color model if needed
    if model_mode == "color_256_64":
        model_model_path = os.path.join(base_cache_dir, "crmark_color_size_256_bit_64.pth")
        if not os.path.exists(model_model_path):
            _download_from_github_release(_color_256_64_MODEL_URL, model_model_path)

    # Download color model if needed
    if model_mode == "color_256_100":
        model_model_path = os.path.join(base_cache_dir, "crmark_color_size_256_bit_100.pth")
        if not os.path.exists(model_model_path):
            _download_from_github_release(_color_256_100_MODEL_URL, model_model_path)


class CRMark(object):
    """
    A class for reversible image watermarking using deep learning and reversible data hiding (RDH).

    This class supports encoding a message or binary watermark into an image and recovering both the
    original image and the watermark. It supports grayscale and color images with pre-trained models.
    """

    def __init__(self, model_mode="color_256_64", level_bits_len=10, freq_bits_len=10, device="cpu", float64=False):
        """
        Initialize the CRMark instance with the specified configuration.

        Parameters:
            model_mode (str): The model variant to use. Must be one of:
                - "color_256_64": For RGB images, embeds 64 bits total, allowing 3 characters (24 bits) after BCH coding.
                - "color_256_100": For RGB images, embeds 100 bits total, allowing 7 characters (56 bits) after BCH coding.
                - "gray_512_256": For grayscale images, embeds 256 bits total, allowing 20 characters (160 bits) after BCH coding.
            level_bits_len (int): Number of bits used for level encoding in tensor quantization. Default is 10.
            freq_bits_len (int): Number of bits used for frequency encoding in tensor quantization. Default is 10.
            device (str): The device used for computation, e.g., "cpu" or "cuda". Default is "cpu".
            float64 (bool): If True, uses float64 precision; otherwise, uses float32. Default is False.

        Returns:
            None: This constructor loads the appropriate model weights and prepares the instance for encoding/decoding.

        Raises:
            AssertionError: If the given model_mode is not one of the supported options.
        """
        # Validate color mode
        assert model_mode in ["color_256_64", "color_256_100",
                              "gray_512_256"], "model_mode must be 'color_256_64', 'color_256_100' or 'gray_512_256'"

        # Download required model weights
        _download_models(model_mode)

        # Initialize instance variables
        self.k = None  # Model parameter for kernel size
        self.fc = None  # Fully connected layer configuration
        self.rdh = None  # Reversible data hiding module
        self.iIWN = None  # Neural network model for watermarking
        self.device = device  # Computation device
        self.float64 = float64  # Precision flag
        self.img_size = None  # Expected image size
        self.bit_length = None  # Length of watermark in bits

        # Set channel dimension and BCH parameters based on color mode
        if model_mode == "color_256_64":
            self.channel_dim = 3  # RGB channels
            BCH_POLYNOMIAL_ = 137  # BCH polynomial for error correction
            BCH_BITS_ = 3  # BCH error correction bits
        elif model_mode == "color_256_100":
            self.channel_dim = 3  # RGB channels
            BCH_POLYNOMIAL_ = 137  # BCH polynomial for error correction
            BCH_BITS_ = 5  # BCH error correction bits
        else:
            self.channel_dim = 1  # Grayscale channel
            BCH_POLYNOMIAL_ = 501  # BCH polynomial for error correction
            BCH_BITS_ = 12  # BCH error correction bits

        # Initialize other components
        self.tensorcoder = None  # Tensor compression module
        self.model_mode = model_mode  # Store color mode
        self.net_min_size = None  # Minimum size for neural network input
        self.freq_bits_len = freq_bits_len  # Frequency bits for compression
        self.level_bits_len = level_bits_len  # Level bits for compression
        self.bch = BCH(BCH_POLYNOMIAL_, BCH_BITS_)  # BCH error correction instance
        self.transform = transforms.Compose([transforms.ToTensor()])  # Image to tensor transform

        # Set cache directory for model weights
        if platform.system() == "Windows":
            base_cache_dir = os.path.join(os.environ["USERPROFILE"], ".cache", "crmark")
        else:
            base_cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "crmark")
        os.makedirs(base_cache_dir, exist_ok=True)

        # Define model paths
        gray_model_path = os.path.join(base_cache_dir, "crmark_gray_size_512_bit_256.pth")
        color_256_64_model_path = os.path.join(base_cache_dir, "crmark_color_size_256_bit_64.pth")
        color_256_100_model_path = os.path.join(base_cache_dir, "crmark_color_size_256_bit_100.pth")

        # Load the appropriate model based on color mode
        if model_mode == "color_256_64":
            self.load_model(color_256_64_model_path)
        elif model_mode == "color_256_100":
            self.load_model(color_256_100_model_path)
        else:
            self.load_model(gray_model_path)

    def load_model(self, model_path: str):
        """
        Load pre-trained model weights and configure the watermarking model.

        Parameters:
            model_path (str): Path to the pre-trained model weights file.

        Returns:
            None: Configures the model and related components.
        """
        # Load model dictionary from file
        load_dict = torch.load(model_path, map_location="cpu", weights_only=False)

        # Extract parameters
        self.k = load_dict['param_dict']['k']  # Kernel size
        self.net_min_size = load_dict['param_dict']['min_size']  # Minimum input size
        self.fc = load_dict['param_dict']['fc']  # Fully connected layer config
        self.bit_length = load_dict['param_dict']['bit_length']  # Watermark bit length
        self.img_size = load_dict['param_dict']['img_size']  # Expected image size
        self.channel_dim = load_dict['param_dict']['channel_dim']  # Number of channels

        # Set precision to float64 if specified
        if self.float64:
            torch.set_default_dtype(torch.float64)

        # Initialize reversible data hiding module
        self.rdh = CustomRDH((self.img_size, self.img_size, self.channel_dim), self.device)

        # Initialize tensor coder for compression
        self.tensorcoder = TensorCoder(
            (self.img_size, self.img_size, self.channel_dim),
            (1, self.bit_length),
            self.level_bits_len,
            self.freq_bits_len
        )

        # Initialize neural network model
        self.iIWN = Model(self.img_size, self.channel_dim, self.bit_length, self.k, self.net_min_size, self.fc)
        self.iIWN.load_state_dict(load_dict['model_state_dict'])
        self.iIWN.to(self.device)
        self.iIWN.eval()  # Set model to evaluation mode

    def encode(self, img_path: str, message: str) -> Tuple[bool, Any]:
        """
        Embed a text message into an image as a watermark.

        Parameters:
            img_path (str): The file path to the cover image in which the message will be embedded.
            message (str): The text message to embed.
                           The required message length depends on the model's color mode:
                               - "color_256_64": length must be 3
                               - "color_256_100": length must be 7
                               - "gray_512_256": length must be 20

        Returns:
            Tuple[bool, PIL.Image]:
                - bool: True if embedding was successful, False otherwise.
                - PIL.Image: The resulting stego image containing the embedded watermark.
        """
        # Load and validate the cover image
        cover_img = Image.open(img_path)
        cover_img_ndarray = np.uint8(cover_img)

        # Validate message length and image dimensions
        if self.model_mode == "color_256_64":
            assert len(message) == 5 and cover_img_ndarray.ndim == 3, \
                "For color_256_64 model, the message length should be 5 and image must be RGB"
        if self.model_mode == "color_256_100":
            assert len(message) == 7 and cover_img_ndarray.ndim == 3, \
                "For color_256_100 model, the message length should be 7 and image must be RGB"
        if self.model_mode == "gray_512_256":
            assert len(message) == 20 and cover_img_ndarray.ndim == 2, \
                "For gray_512_256 model, the message length should be 20 and image must be grayscale"

        # Compute image hash for integrity verification
        cover_img_hash = sha256_of_image_array(cover_img_ndarray)
        cover_img_hash_bitstream = sha256_to_bitstream(cover_img_hash)

        # Transform image to tensor
        cover_img_tensor = self.transform(cover_img).unsqueeze(0).to(self.device)

        # Encode message with BCH and pad to bit length
        watermark = self.bch.Encode(message)
        watermark += [random.randint(0, 1) for _ in range(self.bit_length - len(watermark))]
        secret_tensor = torch.as_tensor(watermark, dtype=torch.float32).unsqueeze(0).to(self.device)

        # Perform watermark embedding
        with torch.no_grad():
            if self.float64:
                cover_img_tensor = cover_img_tensor.to(torch.float64)
                secret_tensor = secret_tensor.to(torch.float64)
            overflow_stego, z = self.iIWN.forward(cover_img_tensor, secret_tensor, True, False)

        # Process stego image
        stego_255 = torch.round(overflow_stego * 255.)
        z_round = torch.round(z)
        clip_stego_img, auxbits = self.tensorcoder.compress(stego_255, z_round)

        # Embed auxiliary bits and hash using RDH
        issuccessful, rdh_stego_img = self.rdh.embed(clip_stego_img, auxbits + cover_img_hash_bitstream)
        if not issuccessful:
            print("The reversible embedding is failed")

        return issuccessful, Image.fromarray(np.uint8(rdh_stego_img))

    def recover(self, stego_img_path: str) -> Union[Tuple[bool, Any, Any], Tuple[bool, None, None]]:
        """
        Recover the original cover image and embedded message from a stego image.

        Parameters:
            stego_img_path (str): Path to the stego image containing the watermark.

        Returns:
            Union[Tuple[bool, PIL.Image, str], Tuple[bool, None, None]]:
                - bool: True if the image was attacked (hash mismatch), False otherwise.
                - PIL.Image or None: Recovered cover image, or None if recovery fails.
                - str or None: Recovered message, or None if recovery fails.
        """
        # Load stego image
        stego_img = Image.open(stego_img_path)
        stego_img_ndarray = np.uint8(stego_img)

        # Extract data using RDH
        issuccessful, clipped_stego_img, ext_auxbits = self.rdh.extract(stego_img_ndarray)
        if issuccessful:
            # Extract image hash and decompress stego image
            cover_img_hash_bitstream = ext_auxbits[-256:]
            rec_overflow_stego, rec_z = self.tensorcoder.decompress(clipped_stego_img, ext_auxbits[:-256])

            # Recover cover image and watermark
            with torch.no_grad():
                if self.float64:
                    rec_z = rec_z.to(torch.float64)
                    rec_overflow_stego = rec_overflow_stego.to(torch.float64)
                rec_cover_tensor, rec_watermark = self.iIWN.forward(rec_overflow_stego / 255., rec_z, True, True)

            # Convert recovered cover tensor to image
            if rec_cover_tensor.shape[1] == 1:
                rec_cover = torch.round(rec_cover_tensor * 255.)[0][0].detach().cpu().numpy()
            else:
                rec_cover = torch.round(rec_cover_tensor * 255.)[0].permute(1, 2, 0).detach().cpu().numpy()
            rec_cover = np.uint8(rec_cover)

            # Verify image hash
            rec_cover_img_hash = sha256_of_image_array(rec_cover)
            rec_cover_img_hash_bitstream = sha256_to_bitstream(rec_cover_img_hash)

            # Process recovered watermark
            rec_watermark = torch.round(torch.clip(rec_watermark, 0., 1.))
            rec_watermark = rec_watermark[0].detach().cpu().numpy().astype(int).tolist()
            valid_part = rec_watermark[:(len(rec_watermark) // 8) * 8]
            isdecode, decoded_data = self.bch.Decode(valid_part)

            # Check if image was attacked
            if np.array_equal(rec_cover_img_hash_bitstream, cover_img_hash_bitstream):
                return False, Image.fromarray(rec_cover), decoded_data
            else:
                return True, Image.fromarray(rec_cover), decoded_data
        else:
            return True, None, None

    def decode(self, stego_img_path: str) -> Tuple[bool, str]:
        """
        Extract and decode the embedded message from a stego image.

        Parameters:
            stego_img_path (str): Path to the stego image containing the watermark.

        Returns:
            Tuple[bool, str]:
                - bool: True if decoding was successful, False otherwise.
                - str: The decoded message extracted from the watermark.
        """
        # Load and transform stego image
        stego_img = Image.open(stego_img_path)
        stego_img_ndarray = np.uint8(stego_img)
        stego_img_tensor = self.transform(stego_img_ndarray).unsqueeze(0).to(self.device)

        # Generate random noise for decoding
        sampled_z = torch.randn(size=(1, self.bit_length)).to(self.device)

        # Extract watermark
        with torch.no_grad():
            _, _ext_watermark = self.iIWN.forward(stego_img_tensor, sampled_z, True, True)

        # Process extracted watermark
        _ext_watermark = torch.round(torch.clip(_ext_watermark, 0., 1.))
        ext_watermark = _ext_watermark[0].detach().cpu().numpy().astype(int).tolist()
        valid_part = ext_watermark[:(len(ext_watermark) // 8) * 8]
        isdecode, decoded_data = self.bch.Decode(valid_part)
        return isdecode, decoded_data

    def encode_bits(self, img_path: str, watermark: list) -> Tuple[bool, Any]:
        """
        Embed a binary watermark into an image.

        Parameters:
            img_path (str): Path to the cover image to embed the watermark.
            watermark (list): List of binary values (0 or 1) to embed as the watermark.

        Returns:
            Tuple[bool, PIL.Image]:
                - bool: True if embedding was successful, False otherwise.
                - PIL.Image: The stego image with the embedded watermark.
        """
        # Load cover image
        cover_img = Image.open(img_path)
        cover_img_ndarray = np.uint8(cover_img)

        # Compute image hash
        cover_img_hash = sha256_of_image_array(cover_img_ndarray)
        cover_img_hash_bitstream = sha256_to_bitstream(cover_img_hash)

        # Transform image to tensor
        cover_img_tensor = self.transform(cover_img).unsqueeze(0).to(self.device)
        secret_tensor = torch.as_tensor(watermark, dtype=torch.float32).unsqueeze(0).to(self.device)

        # Perform watermark embedding
        with torch.no_grad():
            if self.float64:
                cover_img_tensor = cover_img_tensor.to(torch.float64)
                secret_tensor = secret_tensor.to(torch.float64)
            overflow_stego, z = self.iIWN.forward(cover_img_tensor, secret_tensor, True, False)

        # Process stego image
        stego_255 = torch.round(overflow_stego * 255.)
        z_round = torch.round(z)
        clip_stego_img, auxbits = self.tensorcoder.compress(stego_255, z_round)

        # Embed auxiliary bits and hash using RDH
        issuccessful, rdh_stego_img = self.rdh.embed(clip_stego_img, auxbits + cover_img_hash_bitstream)
        if not issuccessful:
            print("The reversible embedding is failed")
        return issuccessful, Image.fromarray(np.uint8(rdh_stego_img))

    def recover_bits(self, stego_img_path: str) -> Union[Tuple[bool, Any, Any], Tuple[bool, None, None]]:
        """
        Recover the original cover image and binary watermark from a stego image.

        Parameters:
            stego_img_path (str): Path to the stego image containing the watermark.

        Returns:
            Union[Tuple[bool, PIL.Image, list], Tuple[bool, None, None]]:
                - bool: True if the image was attacked (hash mismatch), False otherwise.
                - PIL.Image or None: Recovered cover image, or None if recovery fails.
                - list or None: Recovered watermark as a list of bits, or None if recovery fails.
        """
        # Load stego image
        stego_img = Image.open(stego_img_path)
        stego_img_ndarray = np.uint8(stego_img)

        # Extract data using RDH
        issuccessful, clipped_stego_img, ext_auxbits = self.rdh.extract(stego_img_ndarray)
        if issuccessful:
            # Extract image hash and decompress stego image
            cover_img_hash_bitstream = ext_auxbits[-256:]
            rec_overflow_stego, rec_z = self.tensorcoder.decompress(clipped_stego_img, ext_auxbits[:-256])

            # Recover cover image and watermark
            with torch.no_grad():
                if self.float64:
                    rec_z = rec_z.to(torch.float64)
                    rec_overflow_stego = rec_overflow_stego.to(torch.float64)
                rec_cover_tensor, rec_watermark = self.iIWN.forward(rec_overflow_stego / 255., rec_z, True, True)

            # Convert recovered cover tensor to image
            if rec_cover_tensor.shape[1] == 1:
                rec_cover = torch.round(rec_cover_tensor * 255.)[0][0].detach().cpu().numpy()
            else:
                rec_cover = torch.round(rec_cover_tensor * 255.)[0].permute(1, 2, 0).detach().cpu().numpy()
            rec_cover = np.uint8(rec_cover)

            # Verify image hash
            rec_cover_img_hash = sha256_of_image_array(rec_cover)
            rec_cover_img_hash_bitstream = sha256_to_bitstream(rec_cover_img_hash)

            # Process recovered watermark
            rec_watermark = torch.round(torch.clip(rec_watermark, 0., 1.))
            rec_watermark = rec_watermark.squeeze(0).detach().cpu().numpy().astype(int).tolist()

            # Check if image was attacked
            if np.array_equal(rec_cover_img_hash_bitstream, cover_img_hash_bitstream):
                return False, Image.fromarray(rec_cover), rec_watermark
            else:
                return True, Image.fromarray(rec_cover), rec_watermark
        else:
            return True, None, None

    def decode_bits(self, stego_img_path: str) -> list:
        """
        Extract the binary watermark from a stego image.

        Parameters:
            stego_img_path (str): Path to the stego image containing the watermark.

        Returns:
            list: The extracted watermark as a list of binary values (0 or 1).
        """
        # Load and transform stego image
        stego_img = Image.open(stego_img_path)
        stego_img_ndarray = np.uint8(stego_img)
        stego_img_tensor = self.transform(stego_img_ndarray).unsqueeze(0).to(self.device)

        # Generate random noise for decoding
        sampled_z = torch.randn(size=(1, self.bit_length)).to(self.device)

        # Extract watermark
        with torch.no_grad():
            _, _ext_watermark = self.iIWN.forward(stego_img_tensor, sampled_z, True, True)

        # Process extracted watermark
        _ext_watermark = torch.round(torch.clip(_ext_watermark, 0., 1.))
        ext_watermark = _ext_watermark[0].detach().cpu().numpy().astype(int).tolist()
        return ext_watermark
