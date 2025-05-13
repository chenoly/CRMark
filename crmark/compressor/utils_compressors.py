import torch
import numpy as np
from torch import Tensor
from numpy import ndarray
from .arithmeticcoder import ArithmeticEncoder


class CustomArithmeticEncoder:
    def __init__(self, level_bits_len: int = 10, freq_bits_len: int = 20):
        """
        Initialize the encoder with the specified bit lengths for levels and frequencies.

        :param level_bits_len: Bit length used for encoding integer values.
        :param freq_bits_len: Bit length used for encoding frequency information.
        """
        self.buffer_bits = 40
        self.freq_bits_len = freq_bits_len
        self.level_bits_len = level_bits_len

    def ndarray2strlist(self, data: ndarray) -> list:
        """
        Convert a NumPy array into a list of strings.

        :param data: Input ndarray to convert.
        :return: List of strings where each element is a string representation of a value in the ndarray.
        """
        data_list = data.flatten().tolist()
        data_str_list = [str(data_item) for data_item in data_list]
        return data_str_list

    def strlist2ndarray(self, data_str_list: list) -> list:
        """
        Convert a list of strings back into a list of integers.

        :param data_str_list: List of strings to convert.
        :return: List of integers corresponding to the original data values.
        """
        data_list = [int(data_str_item) for data_str_item in data_str_list]
        return data_list

    def datastr2bits(self, data_str_list):
        """
        Encode each string in the input list into a fixed-length binary bit stream.

        :param data_str_list: List of strings to encode.
        :return: A flat list of integers (0s and 1s) representing the encoded bits.
        """
        encoded_bits = []
        for item in data_str_list:
            num = int(item)
            if num < 0:
                num = (1 << self.freq_bits_len) + num  # Handle negative integers
            binary_representation = bin(num)[2:]  # Convert to binary string
            padded_binary = binary_representation.zfill(self.freq_bits_len)  # Zero-pad the string
            if len(padded_binary) > self.freq_bits_len:
                raise ValueError(f"Value {num} cannot be represented in {self.freq_bits_len} bits.")
            encoded_bits.extend(int(bit) for bit in padded_binary)  # Add each bit as an integer to the list
        return encoded_bits

    def bits2datastr(self, encoded_bits):
        """
        Decode a list of binary bits back into the original string values.

        :param encoded_bits: List of integers (0s and 1s) representing the encoded bits.
        :return: List of strings corresponding to the decoded values.
        """
        decoded_data = []
        for i in range(0, len(encoded_bits), self.freq_bits_len):
            bit_str = ''.join(map(str, encoded_bits[i:i + self.freq_bits_len]))  # Group bits into chunks
            num = int(bit_str, 2)  # Convert binary string to an integer
            if num >= (1 << (self.freq_bits_len - 1)):
                num -= (1 << self.freq_bits_len)  # Convert back to signed integer if necessary
            decoded_data.append(str(num))  # Convert back to string and add to list
        return decoded_data

    def integer2bits(self, integer: int):
        """
        Convert an integer to a binary bit list of fixed length.

        :param integer: The integer to convert.
        :return: A list of bits representing the integer.
        """
        if integer < 0:
            integer = (1 << self.level_bits_len) + integer  # Handle negative integers
        if integer >= (1 << self.level_bits_len):
            raise ValueError(f"Value {integer} cannot be represented in {self.level_bits_len} bits.")
        binary_representation = bin(integer)[2:]  # Convert to binary string
        padded_binary = binary_representation.zfill(self.level_bits_len)  # Zero-pad the string
        return [int(bit) for bit in padded_binary]  # Convert binary string to list of bits

    def bits2integer(self, bits: list):
        """
        Convert a list of bits back into an integer.

        :param bits: List of bits (0s and 1s) representing the integer.
        :return: The decoded integer value.
        """
        if len(bits) != self.level_bits_len:
            raise ValueError(f"Bits list must have length {self.level_bits_len}.")
        bit_str = ''.join(str(bit) for bit in bits)  # Convert list of bits to binary string
        num = int(bit_str, 2)  # Convert binary string to integer
        if num >= (1 << (self.level_bits_len - 1)):
            num -= (1 << self.level_bits_len)  # Handle signed integers
        return num

    def compress(self, data: ndarray, frequencies=None) -> list:
        """
        Compress an ndarray by encoding its elements as binary bit streams.

        :param frequencies:
        :param data: NumPy ndarray to compress.
        :return: List of encoded bits representing the compressed data.
        """
        data_str_list = self.ndarray2strlist(data)
        if frequencies is None:
            _frequencies = list(set(data_str_list))  # Get unique elements (frequencies)
            freqs_bits = self.datastr2bits(_frequencies)  # Encode frequencies into bits
            auxbits = freqs_bits + self.integer2bits(len(_frequencies))  # Auxiliary bits: frequency bits + length
            frequencies_input = _frequencies + ["<EOM>"]
        else:
            frequencies_input = frequencies + ["<EOM>"]
            auxbits = []
        coder = ArithmeticEncoder(frequencies=frequencies_input, bits=self.buffer_bits)  # Initialize arithmetic encoder
        data_bits = list(coder.encode(data_str_list + ["<EOM>"]))  # Encode data string list
        if frequencies is None:
            return data_bits + auxbits
        else:
            return data_bits

    def decompress(self, data_freqs_bits: list, frequencies=None) -> ndarray:
        """
        Decompress the encoded bit stream back into an ndarray.

        :param frequencies:
        :param data_freqs_bits: List of encoded bits, including frequency information.
        :return: NumPy ndarray with the decompressed data.
        """
        if frequencies is None:
            len_bits_freqs = data_freqs_bits[-self.level_bits_len:]  # Extract the length of frequencies from the end
            retain_bits = data_freqs_bits[:-self.level_bits_len]  # Remove length bits from the data stream
            len_freqs = self.bits2integer(len_bits_freqs)  # Convert length bits to integer
            freqs_bits = retain_bits[-len_freqs * self.freq_bits_len:]  # Extract frequency bits
            retain_bits = retain_bits[:-len_freqs * self.freq_bits_len]  # Remove frequency bits from data stream
            frequencies = self.bits2datastr(freqs_bits)  # Decode frequencies from bits
            data_bits = retain_bits  # Remaining bits are the actual data bits
            frequencies += ["<EOM>"]
        else:
            data_bits = data_freqs_bits
            frequencies += ["<EOM>"]
        coder = ArithmeticEncoder(frequencies=frequencies, bits=self.buffer_bits)  # Initialize arithmetic decoder
        data_str_list = list(coder.decode(data_bits))  # Decode data bits
        data = [int(data_item) for data_item in data_str_list[:-1]]  # Convert decoded strings back to integers
        return np.asarray(data)  # Return as a NumPy array


class ACCompress:
    def __init__(self, im_size, z_size, level_bits_len, freq_bits_len, device: str = "cpu"):
        """
        Initialize the ACCompress class with image and latent dimensions, bit lengths, and device settings.

        Parameters:
        im_size (tuple): The shape of the image as (Height, Width, Channels).
        z_size (tuple): The shape of the latent tensor z as (Height, Width).
        level_bits_len (int): Bit length used by the arithmetic encoder for quantization levels.
        freq_bits_len (int): Bit length used to encode frequency-related information.
        device (str): The device for tensor computation, e.g., 'cpu' or 'cuda'.
        """
        self.mark_len = 20  # Number of bits used to encode the length of drop_z bits
        self.im_size = im_size  # (H, W, C)
        self.z_size = z_size  # (H, W)
        self.device = device
        self.im_len = im_size[0] * im_size[1] * im_size[2]  # Total number of image pixels
        self.z_len = z_size[0] * z_size[1]  # Total number of latent variables
        self.coder = CustomArithmeticEncoder(level_bits_len=level_bits_len, freq_bits_len=freq_bits_len)

    def combine_bits(self, z_bits: list, stego_bits: list):
        """
        Combine the latent vector bits and stego overflow bits into a single bitstream.

        The bitstream starts with a 20-bit binary length field encoding the length of z_bits.

        Parameters:
        z_bits (list): Encoded bits of the latent variable z.
        stego_bits (list): Encoded bits representing overflow information of the stego image.

        Returns:
        list: A combined bitstream as a list of 0s and 1s.
        """
        length_bits = format(len(z_bits), f'0{self.mark_len}b')  # Convert length to binary string
        length_bits = [int(b) for b in length_bits]  # Convert string to list of ints
        combined_bits = length_bits + z_bits + stego_bits  # Concatenate all parts
        return combined_bits

    def split_bits(self, combined_bits: list):
        """
        Split the combined bitstream into separate latent vector bits and stego image bits.

        Parameters:
        combined_bits (list): A list containing a binary stream.

        Returns:
        tuple: (z_bits, stego_bits) separated from the bitstream.
        """
        length_bits = combined_bits[:self.mark_len]  # Extract first 20 bits
        z_length = int(''.join(map(str, length_bits)), 2)  # Convert to integer length
        z_bits = combined_bits[self.mark_len:self.mark_len + z_length]  # Extract z bits
        stego_bits = combined_bits[self.mark_len + z_length:]  # Extract remaining as stego bits
        return z_bits, stego_bits

    def encode(self, stego_img, drop_z):
        """
        Encode the stego image and latent vector z into a binary bitstream.

        Parameters:
        stego_img (Tensor): Tensor of shape (1, C, H, W) representing the stego image.
        drop_z (Tensor): Tensor of shape (1, H, W) representing the latent representation.

        Returns:
        tuple: (clipped_stego_img, (combined_bits, z_bits, stego_bits))
        """
        overflow_bits_list = []
        clip_stego_img = None

        if stego_img is not None:
            # Clip pixel values to [0, 255] and detect overflow
            clip_stego_img = torch.clip(stego_img, 0, 255)
            steg_img_numpy = stego_img.squeeze(0).detach().permute(1, 2, 0).cpu().numpy()
            overflow = np.zeros_like(steg_img_numpy)

            # Calculate overflow for pixels out of range
            overflow[steg_img_numpy > 255] = steg_img_numpy[steg_img_numpy > 255] - 255
            overflow[steg_img_numpy < 0] = 0 - steg_img_numpy[steg_img_numpy < 0]
            overflow = overflow.astype(int)

            # Compress overflow data using arithmetic coding
            overflow_bits_list = self.coder.compress(overflow.flatten())

        drop_z_bits_list = []
        if drop_z is not None:
            # Clip z values to allowed range and compress
            drop_z_numpy = drop_z.squeeze(0).detach().cpu().numpy()
            drop_z_numpy = np.clip(drop_z_numpy, -2 ** self.coder.freq_bits_len + 1, 2 ** self.coder.freq_bits_len - 1)
            drop_z_numpy_clip = drop_z_numpy.astype(np.int64)
            drop_z_bits_list = self.coder.compress(drop_z_numpy_clip.flatten())

        # Combine bitstreams depending on input availability
        if stego_img is None:
            data_list = drop_z_bits_list
        elif drop_z is None:
            data_list = overflow_bits_list
        else:
            data_list = self.combine_bits(drop_z_bits_list, overflow_bits_list)
            # Optional: recover individual bitstreams to verify combination
            drop_z_bits_list, overflow_bits_list = self.split_bits(data_list)

        return clip_stego_img, (data_list, drop_z_bits_list, overflow_bits_list)

    def decode(self, clip_stego_img: Tensor, data_bits: list):
        """
        Decode the bitstream back into the stego image and latent variable z.

        Parameters:
        clip_stego_img (Tensor): Tensor of shape (1, C, H, W), with values clipped to [0, 255].
        data_bits (list): A list of binary bits encoding the latent and overflow information.

        Returns:
        tuple: Reconstructed stego image (Tensor) and latent vector z (Tensor).
        """
        # Create binary masks for 0 and 255 pixel locations (for overflow correction)
        mask_0 = (clip_stego_img == 0).int()
        mask_255 = (clip_stego_img == 255).int()

        # Split the bitstream into z and overflow components
        drop_z_bits_list, overflow_bits_list = self.split_bits(data_bits)

        # Decompress both parts
        drop_z_list = self.coder.decompress(drop_z_bits_list)
        overflow_list = self.coder.decompress(overflow_bits_list)

        # Reconstruct the overflow map and apply correction
        overflow = torch.as_tensor(overflow_list).reshape(self.im_size).to(self.device).permute(2, 0, 1).unsqueeze(0)
        stego_img = (clip_stego_img - mask_0 * overflow + mask_255 * overflow).float()

        # Reconstruct the latent z
        rec_z = torch.as_tensor(drop_z_list).reshape(self.z_size).to(self.device).float()

        return stego_img, rec_z


class SparseTensorCompressor:
    def __init__(self, image_shape: tuple, z_shape: tuple, level_bits_len: int, freq_bits_len: int, val_bits: int = 10):
        """
        Initialize the SparseTensorCompressor.

        Args:
            image_shape (tuple): Shape of the input image as (height, width, channels).
            z_shape (tuple): Shape of the drop_z latent tensor.
            level_bits_len (int): Number of bits used for level encoding in arithmetic coding.
            freq_bits_len (int): Number of bits used for frequency encoding in arithmetic coding.
            val_bits (int, optional): Number of bits used to encode overflow values. Default is 10.
        """
        self.height, self.width, self.channels = image_shape
        self.mark_len = 20
        self.z_shape = z_shape

        self.row_bits = int(np.ceil(np.log2(self.height)))  # Bits needed to index row
        self.col_bits = int(np.ceil(np.log2(self.width)))  # Bits needed to index column
        self.channel_bits = 0 if self.channels == 1 else int(np.ceil(np.log2(self.channels)))  # Channel index bits
        self.value_bits = int(val_bits)  # Bits for signed overflow values

        self.coder = CustomArithmeticEncoder(level_bits_len=level_bits_len, freq_bits_len=freq_bits_len)

    def combine_bits(self, z_bits: list, stego_bits: list) -> list:
        """
        Combine z_bits and stego_bits into a single bitstream with a length marker.

        Args:
            z_bits (list): Bitstream representing the drop_z tensor.
            stego_bits (list): Bitstream representing overflow data from the image.

        Returns:
            list: Combined bitstream with a prepended length marker for z_bits.
        """
        length_bits = format(len(z_bits), f'0{self.mark_len}b')
        length_bits = [int(b) for b in length_bits]
        return length_bits + z_bits + stego_bits

    def split_bits(self, combined_bits: list) -> tuple:
        """
        Split a combined bitstream into z_bits and stego_bits.

        Args:
            combined_bits (list): Bitstream containing length marker + z_bits + stego_bits.

        Returns:
            tuple: (z_bits, stego_bits), both as lists of bits.
        """
        length_bits = combined_bits[:self.mark_len]
        z_length = int(''.join(map(str, length_bits)), 2)
        z_bits = combined_bits[self.mark_len:self.mark_len + z_length]
        stego_bits = combined_bits[self.mark_len + z_length:]
        return z_bits, stego_bits

    def compress(self, stego_img: torch.Tensor, drop_z: torch.Tensor) -> tuple:
        """
        Compress the stego image and drop_z tensor into a compact bitstream.

        Args:
            stego_img (torch.Tensor): The image tensor (shape: [1, C, H, W]), values in arbitrary range.
            drop_z (torch.Tensor): The drop_z latent tensor.

        Returns:
            tuple:
                - clip_stego_img (torch.Tensor): Clipped image in range [0, 255].
                - data_tuple (tuple): (combined_bits, drop_z_bits_list, overflow_bits_list)
        """
        overflow_bits_list = []
        clip_stego_img = None

        if stego_img is not None:
            clip_stego_img = torch.clip(stego_img, 0, 255)
            steg_img_numpy = stego_img.squeeze(0).detach().permute(1, 2, 0).cpu().numpy()

            overflow = np.zeros_like(steg_img_numpy)
            overflow[steg_img_numpy > 255] = steg_img_numpy[steg_img_numpy > 255] - 255
            overflow[steg_img_numpy < 0] = 0 - steg_img_numpy[steg_img_numpy < 0]
            overflow = overflow.astype(int)

            pos_1, pos_2, pos_3 = (overflow != 0).nonzero()
            non_zero_values = overflow[pos_1, pos_2, pos_3]

            compressed_bitstream = ''
            for row, col, channel, val in zip(pos_1.tolist(), pos_2.tolist(), pos_3.tolist(), non_zero_values.tolist()):
                row_bin = f'{row:0{self.row_bits}b}'
                col_bin = f'{col:0{self.col_bits}b}'
                channel_bin = f'{channel:0{self.channel_bits}b}' if self.channels > 1 else ''
                value_bin = f'{val:0{self.value_bits}b}'
                compressed_bitstream += row_bin + col_bin + channel_bin + value_bin

            overflow_bits_list = [int(bit) for bit in compressed_bitstream]

        drop_z_bits_list = []
        if drop_z is not None:
            drop_z_numpy = drop_z.squeeze(0).detach().cpu().numpy()
            drop_z_numpy = np.clip(drop_z_numpy, -2 ** self.coder.freq_bits_len + 1, 2 ** self.coder.freq_bits_len - 1)
            drop_z_numpy = drop_z_numpy.astype(np.int64)
            drop_z_bits_list = self.coder.compress(drop_z_numpy.flatten())

        if stego_img is None:
            data_list = drop_z_bits_list
        elif drop_z is None:
            data_list = overflow_bits_list
        else:
            data_list = self.combine_bits(drop_z_bits_list, overflow_bits_list)

        return clip_stego_img, (data_list, drop_z_bits_list, overflow_bits_list)

    def decompress(self, clip_stego_img: torch.Tensor, data_bits: list) -> tuple:
        """
        Decompress a bitstream and reconstruct both the stego image and the drop_z tensor.

        Args:
            clip_stego_img (torch.Tensor): Clipped stego image tensor ([1, C, H, W]) with values in [0, 255].
            data_bits (list): Bitstream that contains compressed drop_z and overflow data.

        Returns:
            tuple:
                - stego_img (torch.Tensor): Reconstructed image tensor ([1, C, H, W]).
                - rec_z (torch.Tensor): Reconstructed drop_z tensor.
        """
        drop_z_bits_list, overflow_bits_list = self.split_bits(data_bits)
        drop_z_list = self.coder.decompress(drop_z_bits_list)
        rec_z = torch.as_tensor(drop_z_list, dtype=torch.float).reshape(self.z_shape)

        compressed_bitstream = ''.join(str(bit) for bit in overflow_bits_list)
        overflow = np.zeros((self.height, self.width, self.channels))
        index = 0

        while index < len(compressed_bitstream):
            row = int(compressed_bitstream[index:index + self.row_bits], 2)
            col = int(compressed_bitstream[index + self.row_bits:index + self.row_bits + self.col_bits], 2)
            channel_start = index + self.row_bits + self.col_bits

            if self.channels > 1:
                channel = int(compressed_bitstream[channel_start:channel_start + self.channel_bits], 2)
                index += self.row_bits + self.col_bits + self.channel_bits
            else:
                channel = 0
                index += self.row_bits + self.col_bits

            value = int(compressed_bitstream[index:index + self.value_bits], 2)
            overflow[row, col, channel] = value
            index += self.value_bits

        mask_0 = (clip_stego_img == 0.).float()
        mask_255 = (clip_stego_img == 255.).float()
        overflow_tensor = torch.as_tensor(overflow, dtype=clip_stego_img.dtype).permute(2, 0, 1).unsqueeze(0)

        stego_img = clip_stego_img - mask_0 * overflow_tensor + mask_255 * overflow_tensor
        return stego_img, rec_z


class TensorCoder:
    def __init__(self, image_shape, drop_z_shape, level_bits_len, freq_bits_len):
        """
        Initialize the TensorCoder with two compression strategies.

        :param image_shape: Tuple representing the shape of the image tensor (H, W, C).
        :param drop_z_shape: Tuple representing the shape of the latent vector z (H, W).
        :param level_bits_len: Bit length used for encoding level values in arithmetic coder.
        :param freq_bits_len: Bit length used for encoding frequency values in arithmetic coder.
        """
        self.sparsetensorcompressor = SparseTensorCompressor(image_shape, drop_z_shape, level_bits_len, freq_bits_len)
        self.accompress = ACCompress(image_shape, drop_z_shape, level_bits_len, freq_bits_len)

    def compress(self, input_tensor, drop_z):
        """
        Compress the input tensor and drop_z using two strategies and choose the shorter one.

        :param input_tensor: Tensor of the image to be compressed.
        :param drop_z: Tensor of the latent variable z to be compressed.
        :return: A tuple of (clipped image tensor, encoded bitstream + metadata).
        """
        # Encode using arithmetic coding
        clip_stego_img_a, data_list_a = self.accompress.encode(input_tensor, drop_z)
        # Encode using sparse tensor compression
        clip_stego_img_s, data_list_s = self.sparsetensorcompressor.compress(input_tensor, drop_z)

        # Compare bitstream lengths and select the shorter one
        if len(data_list_a[0]) > len(data_list_s[0]):
            # Use sparse tensor compression; prepend flag 0
            data_list = [0] + data_list_s[0]
        else:
            # Use arithmetic compression; prepend flag 1
            data_list = [1] + data_list_a[0]

        # Return the clipped image (only one version kept) and chosen compression result
        clip_stego_img = clip_stego_img_a
        return clip_stego_img, data_list

    def decompress(self, clip_stego_img, data_list):
        """
        Decompress the bitstream back into image tensor and latent vector z.

        :param clip_stego_img: The clipped stego image tensor used for reconstructing overflow.
        :param data_list: Encoded bitstream with prepended flag indicating the compression strategy.
        :param mode: Decompression mode (default is 's', reserved for extensibility).
        :return: Reconstructed image tensor and latent variable z tensor.
        """
        # Extract flag bit
        data_list_ = data_list[1:]
        if data_list[0] == 1:
            # Use arithmetic decoding
            rec_img, rec_drop_z = self.accompress.decode(clip_stego_img, data_list_)
        else:
            # Use sparse tensor decoding
            rec_img, rec_drop_z = self.sparsetensorcompressor.decompress(clip_stego_img, data_list_)
        return rec_img, rec_drop_z


if __name__ == "__main__":
    import torch
    import numpy as np

    # Define the image and latent vector sizes
    im_size = (400, 400, 3)  # Example image size (H, W, C)
    z_size = (1, 100)  # Example latent vector size (H, W)

    # Define the device ('cpu' or 'cuda')
    device = 'cpu'  # or torch.device('cuda') if using GPU

    # Create random stego image and latent vector (drop_z) tensors
    stego_img = torch.randint(-10, 280, (1, *(im_size[2], im_size[0], im_size[1])), dtype=torch.float32,
                              device=device)  # Stego image of shape (1, H, W, C)
    drop_z = torch.randint(44631230, 44631236, (1, *z_size), dtype=torch.float32,
                           device=device)  # Latent vector (z) of shape (1, H, W)

    print(2 ** 40 > 44631236)
    # Initialize the TensorCoder
    tensor_coder = TensorCoder(im_size, z_size, 10, 40)

    # Encode the stego image and latent vector
    clip_encoded_img, data_bits = tensor_coder.compress(stego_img, drop_z)

    # Decode the binary data back into the stego image and latent vector
    decoded_img, decoded_z = tensor_coder.decompress(clip_encoded_img, data_bits[0])

    # Print only the first 10 elements of the decoded image and latent vector
    print("original stego image [first 10 values]:", stego_img.flatten()[:10].tolist())
    print("Decoded stego image [first 10 values]:", decoded_img.flatten()[:10].tolist())

    print("original latent vector [first 10 values]:", drop_z.flatten()[:10].tolist())
    print("Decoded latent vector (drop_z) [first 10 values]:", decoded_z.flatten()[:10].tolist())

    # Check if the decoded tensors match the original tensors
    assert torch.allclose(stego_img, decoded_img, atol=1e-8), "Stego image decoding failed!"
    assert torch.allclose(drop_z, decoded_z, atol=1e-8), "Latent vector decoding failed!"
    print("Encoding and decoding successful!")
