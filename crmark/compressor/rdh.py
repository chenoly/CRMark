import math
import torch
import random
import numpy as np
from PIL import Image
from torch import Tensor
from typing import Tuple, List
from numpy import ndarray
from .arithmeticcoder import CustomArithmeticEncoder


class RDH:
    def __init__(self, img_size: Tuple[int, int, int], bit_plane: int = 3, level_bits_len: int = 10, freq_bits_len: int = 10):
        """
        Initialize the RDH (Reversible Data Hiding) class.

        :param img_size: Tuple representing the dimensions of the image (height, width, channels).
        :param height_end: Height at which the image is split for embedding.
        """
        self.grayscale_bit = 8
        self.img_size = img_size
        self.bit_plane = bit_plane
        self.storage_len = self.img_size[1] * self.img_size[2] * self.bit_plane
        self.c_len = math.ceil(np.log2(img_size[2])) + 1  # Bits required for channel index
        self.w_len = math.ceil(np.log2(img_size[1])) + 1  # Bits required for width index
        self.h_len = math.ceil(np.log2(img_size[0])) + 1  # Bits required for height index
        self.freq_len = self.c_len + self.w_len + self.h_len
        self.customAC = CustomArithmeticEncoder(level_bits_len, freq_bits_len)

    def set_mask(self, split_height: int):
        """

        Args:
            split_height:

        Returns:

        """
        base_mask = np.fromfunction(lambda h, w: (h + w) % 2, (self.img_size[0] - split_height, self.img_size[1]),
                                    dtype=int)
        mask = np.repeat(base_mask[:, :, np.newaxis], self.img_size[2], axis=2)
        self.mask_o = mask
        self.mask_x = 1 - mask

    def prediect(self, cover_img: ndarray, h: int, w: int, c: int):
        """


        :param cover_img: The cover image as a NumPy array.
        :param h: Height index of the pixel.
        :param w: Width index of the pixel.
        :param c: Channel index of the pixel.
        :return: Predicted value based on the MED algorithm.
        """
        mask = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=np.float32)
        block = cover_img[h - 1: h + 2, w - 1: w + 2, c]
        predict_value = np.round(np.sum(mask * block) / 4.)
        return predict_value

    def predicting_error(self, cover_img: ndarray):
        """
        Calculate the prediction error of the cover image.

        :param cover_img: The cover image as a NumPy array.
        :return: A tuple containing the prediction error and predicted values.
        """
        H, W, C = cover_img.shape
        pv = np.copy(cover_img)
        for h in range(1, H - 1):
            for w in range(1, W - 1):
                for c in range(C):
                    pv[h, w, c] = self.prediect(cover_img, h, w, c)
        pv_o = self.mask_o * pv
        pv_x = self.mask_x * pv
        pe_o = cover_img * self.mask_o - pv_o
        pe_x = cover_img * self.mask_x - pv_x
        return pe_o, pe_x, pv_o, pv_x

    def split_img(self, cover_img: ndarray, height_end: int):
        """
        Split the cover image into two parts: the part for embedding and the location map.

        :param cover_img: The cover image as a NumPy array.
        :return: A tuple containing the image for embedding and the location map.

        Args:
            height_end:
        """
        img4locmap = cover_img[:height_end, :, :]  # Image section for embedding
        img4embed = cover_img[height_end:, :, :]  # Image section for location mapping
        return np.float32(img4embed), np.float32(img4locmap)

    def merge_img(self, watermarked_img4embed: ndarray, marked_img4locmap: ndarray, split_height: int):
        """
        Merge the embedded image and the location map into a single watermarked image.

        :param watermarked_img4embed: The watermarked image section for embedding.
        :param marked_img4locmap: The modified location map.
        :return: The final merged watermarked image.

        Args:
            split_height:
        """
        watermarked_img = np.zeros(shape=self.img_size)
        watermarked_img[:split_height, :, :] = marked_img4locmap
        watermarked_img[split_height:, :, :] = watermarked_img4embed
        return watermarked_img

    def get_top_two_frequent_values(self, pe: ndarray, set_mask: ndarray):
        """
        Finds the two most frequently occurring values in the `pe` array,
        considering only the elements where `set_mask` is 1.

        Args:
            pe (ndarray): A 3D array representing prediction errors or feature values.
            set_mask (ndarray): A binary mask of the same shape as `pe`, indicating valid regions (1) and ignored regions (0).

        Returns:
            tuple: (first_most, first_count, second_most, second_count)
                - first_most (int or None): The most frequently occurring value.
                - first_count (int or None): The count of the most frequently occurring value.
                - second_most (int or None): The second most frequently occurring value.
                - second_count (int or None): The count of the second most frequently occurring value.
        """
        # Extract the central region, excluding the boundary pixels
        pe_central = pe[1:-1, 1:-1, :]
        mask_central = set_mask[1:-1, 1:-1, :]
        # Flatten the array, selecting only elements where mask == 1
        pe_flat = pe_central[mask_central == 1]
        # Get unique values and their respective counts
        unique, counts = np.unique(pe_flat, return_counts=True)
        # Sort values by count in descending order
        sorted_indices = np.argsort(counts)[::-1]
        # Retrieve the top two most frequent values
        top_two_indices = sorted_indices[:2]
        first_most = unique[top_two_indices[0]] if len(top_two_indices) > 0 else None
        first_count = counts[top_two_indices[0]] if len(top_two_indices) > 0 else None
        second_most = unique[top_two_indices[1]] if len(top_two_indices) > 1 else None
        second_count = counts[top_two_indices[1]] if len(top_two_indices) > 1 else None
        return first_most, first_count, second_most, second_count

    def shift_and_embed(self, pe: ndarray, set_mask: ndarray, wm_list: list):
        """
        Shift the prediction error values and embed the watermark bits.

        Args:
            pe (ndarray): The prediction error array.
            set_mask (ndarray): A binary mask indicating valid embedding regions.
            wm_list (list): The list of watermark bits to embed.

        Returns:
            tuple: (embedded_pe, min_value, max_value, capacity, stopcoordinate, remaining_wm_list)
                - embedded_pe (ndarray): The modified prediction error array with embedded watermark bits.
                - min_value (int): The lower threshold for embedding.
                - max_value (int): The upper threshold for embedding.
                - capacity (int): The number of bits that can be embedded.
                - stopcoordinate (tuple): The coordinate where embedding stopped.
                - remaining_wm_list (list): Remaining watermark bits if embedding was not fully completed.
        """
        # Step 1: Compute the two most frequent values
        first_most, first_count, second_most, second_count = self.get_top_two_frequent_values(pe, set_mask)
        capacity = first_count + second_count  # Available embedding capacity

        # Step 2: Define embedding thresholds
        min_value = int(min(first_most, second_most))
        max_value = int(max(first_most, second_most))

        # Step 3: Initialize the modified prediction error array
        shifted_pe = pe.copy()
        H, W, C = pe.shape
        wm_index = 0  # Initialize watermark bit index
        stopcoordinate = (0, 0, 0)  # Default stop coordinate

        # Step 4: Shift and embed watermark simultaneously
        for h in range(1, H - 1):
            for w in range(1, W - 1):
                for c in range(C):
                    if set_mask[h, w, c] == 1 and wm_index < len(wm_list):
                        stopcoordinate = (h, w, c)
                        if pe[h, w, c] < min_value:
                            shifted_pe[h, w, c] = pe[h, w, c] - 1.  # Shift down
                        elif pe[h, w, c] > max_value:
                            shifted_pe[h, w, c] = pe[h, w, c] + 1.  # Shift up
                        elif pe[h, w, c] == max_value:
                            shifted_pe[h, w, c] += wm_list[wm_index]
                            wm_index += 1
                        elif pe[h, w, c] == min_value:
                            shifted_pe[h, w, c] -= wm_list[wm_index]
                            wm_index += 1
                        if wm_index == len(wm_list):  # If all watermark bits are embedded, return
                            return shifted_pe, min_value, max_value, capacity, stopcoordinate, []
        # Step 5: Return results with remaining watermark bits (if any)
        return shifted_pe, min_value, max_value, capacity, stopcoordinate, wm_list[wm_index:]

    def extract_and_shift(self, embedded_pe: ndarray, min_value: int, max_value: int, set_mask: ndarray,
                          stopcoordinate: tuple):
        """
        Extract the embedded watermark and restore the original prediction error.

        Args:
            embedded_pe (ndarray): The watermarked prediction error array.
            min_value (int): The lower threshold used during embedding.
            max_value (int): The upper threshold used during embedding.
            set_mask (ndarray): A binary mask indicating valid embedding regions.
            stopcoordinate (tuple): The coordinate where embedding stopped.

        Returns:
            tuple: (restored_pe, extracted_wm_list)
                - restored_pe (ndarray): The restored prediction error array.
                - extracted_wm_list (list): The extracted watermark bits.
        """
        H, W, C = embedded_pe.shape
        restored_pe = embedded_pe.copy()  # Copy embedded PE to restore
        extracted_wm_list = []  # List to store extracted watermark bits

        # Determine the stopping point
        stop_reached = False  # Flag to stop extraction early

        # Iterate over the image to extract watermark and restore prediction error
        for h in range(1, H - 1):
            if stop_reached: break
            for w in range(1, W - 1):
                if stop_reached: break
                for c in range(C):
                    if stop_reached: break
                    if set_mask[h, w, c] == 1:
                        if embedded_pe[h, w, c] == max_value + 1:
                            extracted_wm_list.append(1)
                            restored_pe[h, w, c] = max_value
                        elif embedded_pe[h, w, c] == max_value:
                            extracted_wm_list.append(0)
                        elif embedded_pe[h, w, c] == min_value - 1:
                            extracted_wm_list.append(1)
                            restored_pe[h, w, c] = min_value
                        elif embedded_pe[h, w, c] == min_value:
                            extracted_wm_list.append(0)
                        if (h, w, c) == stopcoordinate:
                            stop_reached = True  # Stop at last embedded bit
                            break

        # Restore the original prediction error by shifting back
        for h in range(1, H - 1):
            for w in range(1, W - 1):
                for c in range(C):
                    if set_mask[h, w, c] == 1:
                        if restored_pe[h, w, c] < min_value:
                            restored_pe[h, w, c] += 1  # Shift back down values
                        elif restored_pe[h, w, c] > max_value:
                            restored_pe[h, w, c] -= 1  # Shift back up values
                    if (h, w, c) == stopcoordinate:
                        return restored_pe, extracted_wm_list
        return restored_pe, extracted_wm_list

    def compute_stego_img(self, pe_o: ndarray, pe_x: ndarray, pv_o: ndarray, pv_x: ndarray):
        stego_img = (pe_o + pv_o) * self.mask_o + (pe_x + pv_x) * self.mask_x
        return stego_img

    def compute_overflow_map(self, stego_img: ndarray, mask: ndarray, stopcoordinate: Tuple) -> list:
        """
        Compute the location map indicating overflow or underflow pixels after embedding.
        The outermost border pixels are excluded from the computation.

        Args:
            stego_img: The stego image after embedding, with shape (H, W, C) and dtype=np.uint8 or np.float32.
            mask: A mask array indicating valid embedding regions, having the same shape as stego_img.
            stopcoordinate: A coordinate (h, w, c) that specifies an early stopping point in the form (height, width, channel).

        Returns:
            list: A list of 0s and 1s, where 1 indicates an overflow or underflow pixel, and 0 indicates a normal pixel.
        """
        H, W, C = stego_img.shape
        if stego_img.dtype is not np.float32:
            stego_img = stego_img.astype(np.float32)  # Convert to float32 for overflow/underflow checking
        location_map = []
        for h in range(1, H - 1):
            for w in range(1, W - 1):
                for c in range(C):
                    if mask[h, w, c] == 1:  # Process only pixels within the valid embedding region
                        pixel_value = stego_img[h, w, c]
                        if pixel_value > 255 or pixel_value < 0:
                            location_map.append(1)  # Overflow or underflow occurred
                        else:
                            location_map.append(0)  # No overflow or underflow
                    if (h, w, c) == stopcoordinate:
                        return location_map  # Stop early if the specified coordinate is reached
        return location_map

    def recovery_overflow_stego_image(self, clipped_stego_img: ndarray, mask: ndarray, overflow_map: list) -> ndarray:
        """
        Recover the original stego image before clipping by restoring overflow and underflow pixels based on the location map.

        Args:
            clipped_stego_img: The stego image after clipping, with shape (H, W, C) and dtype=np.uint8.
            mask: A mask array indicating valid embedding regions, having the same shape as clipped_stego_img.
            overflow_map: A list of 0s and 1s, where 1 indicates an overflow or underflow pixel, and 0 indicates a normal pixel.

        Returns:
            ndarray: The recovered stego image with dtype=np.float32.
        """
        H, W, C = clipped_stego_img.shape
        recovered_img = clipped_stego_img.astype(np.float32)
        location_idx = 0
        for h in range(1, H - 1):
            for w in range(1, W - 1):
                for c in range(C):
                    if mask[h, w, c] == 1 and location_idx < len(overflow_map):
                        if overflow_map[location_idx] == 1:
                            # Overflow or underflow occurred; recover the pixel
                            if clipped_stego_img[h, w, c] == 0:
                                recovered_img[h, w, c] = -1.0  # Underflow pixel
                            elif clipped_stego_img[h, w, c] == 255:
                                recovered_img[h, w, c] = 256.0  # Overflow pixel
                        location_idx += 1
                        if location_idx == len(overflow_map):
                            return recovered_img
        return recovered_img

    def encodeIntegerbyGivenLength(self, n: int, length: int) -> list:
        """
        Encode an integer as a binary list of a given length, supporting both positive and negative integers.

        :param n: The integer to encode.
        :param length: The desired length of the binary representation.
        :return: A list representing the binary encoding of the integer.
        """
        # Check if the number is negative
        if n < 0:
            n = (1 << length) + n  # Convert to 2's complement for negative numbers
        binary_representation = bin(n)[2:]  # Get binary representation without '0b' prefix
        if len(binary_representation) < length:
            binary_representation = binary_representation.zfill(length)  # Pad with zeros to the left
        elif len(binary_representation) > length:
            raise ValueError("The number cannot be represented in the given length")
        return [int(bit) for bit in binary_representation]  # Convert to a list of integers

    def decodeIntegerbyGivenBits(self, bits: list) -> int:
        """
        Decode a binary list back into an integer, considering the possibility of negative numbers.

        :param bits: The binary representation as a list of bits.
        :return: The decoded integer.
        """
        bit_string = ''.join(map(str, bits))  # Convert list of bits to a string
        value = int(bit_string, 2)  # Convert binary string to integer
        # Check if the number is negative (if the sign bit is set)
        if bits[0] == 1:
            value -= (1 << len(bits))  # Convert to negative value
        return value  # Return the decoded integer

    def encode_auxiliary_information(self, overflow_map_o, min_v_o, max_v_o, stop_coor_o,
                                     overflow_map_x, min_v_x, max_v_x, stop_coor_x):
        """
        Compute the auxiliary information required for the reversible data hiding process.

        Args:
            overflow_map_o (list): A bit list representing the overflow location map for channel O.
            min_v_o (int): The minimum pixel value in the overflow region for channel O.
            max_v_o (int): The maximum pixel value in the overflow region for channel O.
            stop_coor_o (tuple): The stopping coordinates (H, W, C) for processing channel O.
            overflow_map_x (list): A bit list representing the overflow location map for channel X.
            min_v_x (int): The minimum pixel value in the overflow region for channel X.
            max_v_x (int): The maximum pixel value in the overflow region for channel X.
            stop_coor_x (tuple): The stopping coordinates (H, W, C) for processing channel X.

        Returns:
            list: A list of bits representing the encoded auxiliary information.
        """

        # Compute the number of bits required to store the length of overflow map O
        l_mask_o = int(self.c_len + self.h_len + self.w_len)  # Length encoding bits
        l_mask_o_bits = [int(b) for b in format(len(overflow_map_o), f'0{l_mask_o}b')]

        # Concatenate the overflow maps from both channels into a single bitstream
        overflow_map_bitstream = l_mask_o_bits + overflow_map_o + overflow_map_x

        # Compress the combined overflow map bitstream using arithmetic coding
        compressed_overflow_map_bitstream = self.customAC.compress(np.asarray(overflow_map_bitstream), ["1", "0"])

        # Encode the min and max pixel values for channel O using fixed-length binary encoding
        min_v_o_bits = self.encodeIntegerbyGivenLength(min_v_o, self.grayscale_bit)
        max_v_o_bits = self.encodeIntegerbyGivenLength(max_v_o, self.grayscale_bit)

        # Encode the min and max pixel values for channel X using fixed-length binary encoding
        min_v_x_bits = self.encodeIntegerbyGivenLength(min_v_x, self.grayscale_bit)
        max_v_x_bits = self.encodeIntegerbyGivenLength(max_v_x, self.grayscale_bit)

        # Encode the stopping coordinates (H, W, C) for channel O
        h_bits = self.encodeIntegerbyGivenLength(stop_coor_o[0], self.h_len)
        w_bits = self.encodeIntegerbyGivenLength(stop_coor_o[1], self.w_len)
        c_bits = self.encodeIntegerbyGivenLength(stop_coor_o[2], self.c_len)
        coor_o_bits = h_bits + w_bits + c_bits

        # Encode the stopping coordinates (H, W, C) for channel X
        h_bits = self.encodeIntegerbyGivenLength(stop_coor_x[0], self.h_len)
        w_bits = self.encodeIntegerbyGivenLength(stop_coor_x[1], self.w_len)
        c_bits = self.encodeIntegerbyGivenLength(stop_coor_x[2], self.c_len)
        coor_x_bits = h_bits + w_bits + c_bits

        # Concatenate all encoded components to form the final auxiliary information bitstream
        all_bits = (compressed_overflow_map_bitstream + min_v_o_bits + max_v_o_bits +
                    min_v_x_bits + max_v_x_bits + coor_o_bits + coor_x_bits)

        return all_bits

    def decode_auxiliary_information(self, bits4auxinfo: list):
        """
        Decomposes the auxiliary information bitstream to retrieve the overflow maps, min/max values, and stop coordinates.

        Args:
            bits4auxinfo (list): A list of bits representing the encoded auxiliary information.

        Returns:
            tuple: A tuple containing:
                - overflow_map_mask_o (list): A bit list representing the overflow location map for the original image.
                - min_v_o (int): The minimum value for the original image.
                - max_v_o (int): The maximum value for the original image.
                - stop_coor_o (tuple): The stop coordinate (h, w, c) for the original image.
                - overflow_map_mask_x (list): A bit list representing the overflow location map for the attacked image.
                - min_v_x (int): The minimum value for the attacked image.
                - max_v_x (int): The maximum value for the attacked image.
                - stop_coor_x (tuple): The stop coordinate (h, w, c) for the attacked image.
        """
        # Extract the bits representing the coordinates and min/max values
        total_coor_bits = int(2 * (self.h_len + self.w_len + self.c_len))
        total_min_max_bits = int(4 * self.grayscale_bit)
        corr_bits = bits4auxinfo[-(total_coor_bits + total_min_max_bits):]
        compressed_bitstream = bits4auxinfo[:-(total_coor_bits + total_min_max_bits)]

        # Length of the bitstream encoding the map mask dimensions (h_len, w_len, c_len)
        l_mask_o = int(self.c_len + self.h_len + self.w_len)

        # Decompress the bitstream containing the overflow map information
        compressed_overflow_map_bitstream = list(self.customAC.decompress(compressed_bitstream, ["1", "0"]))

        # Extract the bits representing the length of the overflow map
        l_mask_o_bits = compressed_overflow_map_bitstream[:l_mask_o]
        len_mask_o = int(''.join(map(str, l_mask_o_bits)), 2)

        # Split the overflow map into mask_o and mask_x based on the extracted length
        overflow_map_mask_o = compressed_overflow_map_bitstream[l_mask_o:][:len_mask_o]
        overflow_map_mask_x = compressed_overflow_map_bitstream[l_mask_o:][len_mask_o:]

        # Decode the min and max values for the original and attacked images
        min_v_o_bits = corr_bits[:self.grayscale_bit]
        max_v_o_bits = corr_bits[self.grayscale_bit:2 * self.grayscale_bit]
        min_v_x_bits = corr_bits[2 * self.grayscale_bit:3 * self.grayscale_bit]
        max_v_x_bits = corr_bits[3 * self.grayscale_bit:4 * self.grayscale_bit]

        min_v_o = self.decodeIntegerbyGivenBits(min_v_o_bits)
        max_v_o = self.decodeIntegerbyGivenBits(max_v_o_bits)
        min_v_x = self.decodeIntegerbyGivenBits(min_v_x_bits)
        max_v_x = self.decodeIntegerbyGivenBits(max_v_x_bits)

        # Decode the stop coordinates
        coor_bits = corr_bits[4 * self.grayscale_bit:]
        coor_o_bits = coor_bits[:int(self.h_len + self.w_len + self.c_len)]
        coor_x_bits = coor_bits[int(self.h_len + self.w_len + self.c_len):]

        h_o = self.decodeIntegerbyGivenBits(coor_o_bits[:self.h_len])
        w_o = self.decodeIntegerbyGivenBits(coor_o_bits[self.h_len:self.h_len + self.w_len])
        c_o = self.decodeIntegerbyGivenBits(coor_o_bits[self.h_len + self.w_len:])
        stop_coor_o = (h_o, w_o, c_o)

        h_x = self.decodeIntegerbyGivenBits(coor_x_bits[:self.h_len])
        w_x = self.decodeIntegerbyGivenBits(coor_x_bits[self.h_len:self.h_len + self.w_len])
        c_x = self.decodeIntegerbyGivenBits(coor_x_bits[self.h_len + self.w_len:])
        stop_coor_x = (h_x, w_x, c_x)

        return (
            overflow_map_mask_o, min_v_o, max_v_o, stop_coor_o,
            overflow_map_mask_x, min_v_x, max_v_x, stop_coor_x
        )

    def extract_lsb(self, image: ndarray) -> list:
        """
        Extract the least significant bits (LSBs) from the image, based on the object's bit_plane attribute.
        """
        if image.dtype is not np.uint8:
            image = image.astype(np.uint8)
        if image.ndim != 3:
            raise ValueError("Input image must have 3 dimensions.")
        if self.bit_plane < 1 or self.bit_plane > 8:
            raise ValueError("Bit plane must be between 1 and 8.")
        mask = (1 << self.bit_plane) - 1
        lsb_bits = []
        for h in range(image.shape[0]):
            for w in range(image.shape[1]):
                for c in range(image.shape[2]):
                    pixel_value = image[h, w, c]
                    lsb_bits_for_pixel = pixel_value & mask
                    bits_str = format(lsb_bits_for_pixel, f'0{self.bit_plane}b')[::-1]
                    lsb_bits.extend([int(bit) for bit in bits_str])
        return lsb_bits

    def insert_lsb(self, image: ndarray, lsb_bits: list) -> ndarray:
        """
        Insert the least significant bits (LSBs) back into the image, based on the object's bit_plane attribute.
        """
        if image.dtype is not np.uint8:
            image = image.astype(np.uint8)
        if image.ndim != 3:
            raise ValueError("Input image must have 3 dimensions.")
        if self.bit_plane < 1 or self.bit_plane > 8:
            raise ValueError("Bit plane must be between 1 and 8.")
        mask = (1 << self.bit_plane) - 1
        num_bits = self.bit_plane * image.shape[0] * image.shape[1] * image.shape[2]
        if len(lsb_bits) != num_bits:
            raise ValueError(f"Expected {num_bits} LSB bits, but got {len(lsb_bits)}.")
        lsb_idx = 0
        for h in range(image.shape[0]):
            for w in range(image.shape[1]):
                for c in range(image.shape[2]):
                    pixel_value = image[h, w, c]
                    pixel_value &= ~mask
                    lsb_bits_for_pixel = 0
                    for b in range(self.bit_plane):
                        lsb_bits_for_pixel |= (lsb_bits[lsb_idx] << b)
                        lsb_idx += 1
                    pixel_value |= lsb_bits_for_pixel
                    image[h, w, c] = np.uint8(pixel_value)
        return image

    def encode_bitswithlength(self, bitstream: list, pos: str = "prefix"):
        """
        Encode a bitstream with its length information.

        Args:
            bitstream (list): A list of binary bits (e.g., [1, 0, 1, 1]).
            pos (str): Position of the length information, either "prefix" or "suffix".

        Returns:
            list: A new bitstream with length information added.
        """
        assert pos in ["prefix", "suffix"], ValueError("pos must be either 'prefix' or 'suffix'")
        # Calculate the number of bits needed to represent the length
        bits_len = int(self.c_len + self.h_len + self.w_len)
        # Convert the length to a fixed-length binary string
        length_bits = format(len(bitstream), f'0{bits_len}b')
        # Convert the binary string to a list of integers
        length_bits = list(map(int, length_bits))
        # Combine length bits and the original bitstream
        if pos == "prefix":
            return length_bits + bitstream
        else:
            return bitstream + length_bits

    def decode_bitswithlength(self, encoded_bits: list, pos: str = "prefix"):
        """
        Decode a bitstream by its length information.

        Args:
            encoded_bits (list): A list of binary bits containing multiple encoded bitstreams.
            pos (str): Position of the length information, either "prefix" or "suffix".

        Returns:
            list: The decoded bitstream (list of 0s and 1s).
        """
        assert pos in ["prefix", "suffix"], ValueError("pos must be either 'prefix' or 'suffix'")
        # Calculate the number of bits needed to represent the length
        bits_len = int(self.c_len + self.h_len + self.w_len)
        # Extract length bits and determine the length of the bitstream
        if pos == "prefix":
            length_bits = encoded_bits[:bits_len]  # Length bits at the beginning
            length = int(''.join(map(str, length_bits)), 2)
            decoded_bitstream = encoded_bits[bits_len:bits_len + length]
            other_bits = encoded_bits[bits_len + length:]
        else:
            length_bits = encoded_bits[-bits_len:]  # Length bits at the end
            length = int(''.join(map(str, length_bits)), 2)
            decoded_bitstream = encoded_bits[:-bits_len][-length:]
            other_bits = encoded_bits[:-(bits_len + length)]
        return decoded_bitstream, other_bits

    def encode_auxbitslist(self, auxbits: list):
        """
        Encode a list of auxiliary bitstreams into a single bitstream.

        This function concatenates the length of each bitstream (in binary) followed by the bitstream itself
        into a single bitstream. The length of each bitstream is encoded using a fixed-length binary representation
        defined by `self.c_len + self.h_len + self.w_len`.

        Args:
            auxbits (list): A list of lists, where each inner list represents a bitstream of 0s and 1s.

        Returns:
            list: A single list representing the encoded bitstream, where each bitstream is prefixed with its length
                  (encoded as binary).
        """
        encoded_bits = []
        for bitstream in auxbits:
            encoded_bits.extend(self.encode_bitswithlength(bitstream, pos="prefix"))
        return encoded_bits

    def decode_auxbitslist(self, encoded_bits):
        """
        Decode a single encoded bitstream back into a list of auxiliary bitstreams.

        This function reads the length of each bitstream (in binary) from the encoded bitstream, extracts the corresponding
        bitstream, and returns the original list of bitstreams.

        Args:
            encoded_bits (list): A single list representing the encoded bitstream, where each bitstream is prefixed with its
                                  length (encoded as binary).

        Returns:
            list: A list of lists, where each inner list represents a decoded bitstream of 0s and 1s.
        """
        decoded_bitstreams = []
        idx = 0
        # Define the number of bits used to represent the length of each bitstream
        bits_len = int(self.c_len + self.h_len + self.w_len)
        # Process the encoded bitstream by extracting bitstreams one by one
        while idx < len(encoded_bits):
            # Extract the binary length of the current bitstream
            length_bits = encoded_bits[idx: idx + bits_len]
            # Convert the binary length bits to an integer
            length = int(''.join(map(str, length_bits)), 2)
            # Move the index forward by the length of the binary length bits
            idx += bits_len
            # Extract the actual bitstream using the length value
            bitstream = encoded_bits[idx: idx + length]
            # Add the decoded bitstream to the list
            decoded_bitstreams.append(bitstream)
            # Move the index forward by the length of the extracted bitstream
            idx += length
        return decoded_bitstreams

    def embed_watermark(self, cover_img: ndarray, watermark_list: list):
        """
        Embed a watermark into the cover image using reversible data hiding techniques.

        :param cover_img: The cover image as a NumPy array.
        :param watermark_list: The list of watermark bits to embed.
        :return: A tuple indicating success and the resulting watermarked image.
        """
        now_watermark_list = watermark_list.copy()
        pe_o, pe_x, pv_o, pv_x = self.predicting_error(cover_img)
        result_o = self.shift_and_embed(pe_o, self.mask_o, now_watermark_list)
        shifted_stego_pe_o, min_v_o, max_v_o, capacity_o, stop_coor_o, rest_wm_list = result_o
        stego_img4embed_o = self.compute_stego_img(shifted_stego_pe_o, pe_x, pv_o, pv_x)
        overflow_map_mask_o = self.compute_overflow_map(stego_img4embed_o, self.mask_o, stop_coor_o)
        auxbit = self.encode_auxiliary_information(overflow_map_mask_o, min_v_o, max_v_o, stop_coor_o, [], 0, 0,
                                                   (0, 0, 0))
        stego_img = np.clip(stego_img4embed_o, 0, 255)
        if len(rest_wm_list) > 0:
            pe_o, pe_x, pv_o, pv_x = self.predicting_error(stego_img4embed_o)
            result_x = self.shift_and_embed(pe_x, self.mask_x, rest_wm_list)
            shifted_stego_pe_x, min_v_x, max_v_x, capacity_x, stop_coor_x, rest_wm_list = result_x
            stego_img4embed_x = self.compute_stego_img(pe_o, shifted_stego_pe_x, pv_o, pv_x)
            overflow_map_mask_x = self.compute_overflow_map(stego_img4embed_x, self.mask_x, stop_coor_x)
            auxbit = self.encode_auxiliary_information(overflow_map_mask_o, min_v_o, max_v_o, stop_coor_o,
                                                       overflow_map_mask_x, min_v_x, max_v_x, stop_coor_x)
            stego_img = np.clip(stego_img4embed_x, 0, 255)
        return stego_img, auxbit, rest_wm_list

    def embed_once(self, cover_img: ndarray, watermark_list: list, time_index: int):
        """
        Embed a watermark into the cover image through iterative embedding.

        :param cover_img: The original cover image to embed the watermark into (H, W, C).
        :param watermark_list: A list of bits representing the watermark to embed.
        :param time_index: The time of embedding.
        :return:
            True: Indicating the embedding process has completed.
            marked_img: The image with the embedded watermark (H, W, C).
            time_index: The time of embedding
        """
        try_index = 0
        lsb_used_length = 500
        while True:
            split_height = int(np.ceil(lsb_used_length / self.storage_len))
            self.set_mask(split_height)
            img4embed, img4locmap = self.split_img(cover_img, split_height)
            lsb_bits = self.extract_lsb(img4locmap)
            used_lsb_bits = self.encode_bitswithlength(lsb_bits[:lsb_used_length], pos="prefix")
            watermark_list_now = used_lsb_bits + watermark_list
            img4embed, auxbits, rest_wm_list = self.embed_watermark(img4embed, watermark_list_now)
            flatten_auxbits = self.encode_bitswithlength(auxbits, pos="prefix")
            split_height_bits = format(split_height, f'0{self.h_len}b')
            split_height_bits = list(map(int, split_height_bits))
            if time_index == 0:
                flatten_auxbits_with_stop_flag = [0] + split_height_bits + flatten_auxbits
            else:
                flatten_auxbits_with_stop_flag = [1] + split_height_bits + flatten_auxbits
            if len(flatten_auxbits_with_stop_flag) <= lsb_used_length:
                lsb_bits[:len(flatten_auxbits_with_stop_flag)] = flatten_auxbits_with_stop_flag
                marked_img4locmap = self.insert_lsb(img4locmap, lsb_bits)
                marked_img = self.merge_img(img4embed, marked_img4locmap, split_height)
                return np.uint8(np.clip(marked_img, 0, 255)), rest_wm_list
            else:
                lsb_used_length = int(np.ceil(len(flatten_auxbits_with_stop_flag) / 500.) * 500)
                try_index += 1

    def embed(self, cover_img: ndarray, watermark_list: list):
        """
        Embeds a given watermark into the cover image iteratively until all watermark bits are embedded.

        Args:
            cover_img (ndarray): The original cover image where the watermark will be embedded.
                                 It can be a grayscale or color image with shape (H, W) or (H, W, C).
            watermark_list (list): A list of binary watermark bits to be embedded into the image.

        Returns:
            ndarray: The resulting stego image with the embedded watermark.
        """
        issuccessful = True
        stego_img = cover_img.copy()
        # Ensure the image has three dimensions (H, W, C) for consistency
        if stego_img.ndim == 2:
            stego_img = stego_img.reshape((*stego_img.shape, 1))
        # Iteratively embed watermark bits until the list is empty
        time_index = 0
        while len(watermark_list) > 0 and issuccessful:
            stego_img, next_watermark_list = self.embed_once(stego_img, watermark_list, time_index)
            issuccessful = len(next_watermark_list) < len(watermark_list)
            watermark_list = next_watermark_list
            time_index += 1
        # Convert back to 2D if the original image was grayscale
        if stego_img.shape[2] == 1:
            stego_img = stego_img[:, :, 0]
        return issuccessful, stego_img

    def extract(self, stego_img: ndarray):
        """
        Extracts the embedded watermark bits from the given stego image.

        Args:
            stego_img (ndarray): The stego image containing the embedded watermark.
                                 It can be a grayscale or color image with shape (H, W) or (H, W, C).

        Returns:
            list: A list of extracted binary watermark bits, in the order they were embedded.
        """
        ext_all_bits = []  # List to store all extracted watermark bits
        iscontinue = True  # Flag to indicate whether more bits are available for extraction
        issuccessful = True
        # Ensure the image has three dimensions (H, W, C) for consistency
        if stego_img.ndim == 2:
            stego_img = stego_img.reshape((*stego_img.shape, 1))
        # Iteratively extract watermark bits until no more bits are found
        while iscontinue:
            try:
                iscontinue, stego_img, ext_bits = self.extract_once(stego_img)
                ext_all_bits = ext_bits + ext_all_bits
            except Exception as e:
                iscontinue = False
                issuccessful = False
            # Prepend the newly extracted bits to maintain the correct order
        if stego_img.shape[2] == 1:
            stego_img = stego_img[:, :, 0]
        return issuccessful, np.uint8(np.clip(stego_img, 0, 255)), ext_all_bits

    def extract_once(self, stego_img: ndarray):
        """
        Extract the embedded watermark bits from the stego image through iterative extraction.

        :param stego_img: The stego image from which the watermark is to be extracted (H, W, C).
        :return:
            stego_img: The stego image after processing (H, W, C).
            wm_list: The list of extracted watermark bits (in reverse order of extraction).
        """

        stego4ext, img4locmap = self.split_img(stego_img, 1)
        lsb_bits = self.extract_lsb(img4locmap)
        iscontinue = bool(lsb_bits[0])
        split_height_bits = lsb_bits[1:1 + self.h_len]
        split_height = int(''.join(map(str, split_height_bits)), 2)
        self.set_mask(split_height)
        stego4ext, img4locmap = self.split_img(stego_img, split_height)
        lsb_bits = self.extract_lsb(img4locmap)
        flatten_auxbits, _ = self.decode_bitswithlength(lsb_bits[1 + self.h_len:], pos="prefix")
        rec_img4emb, wm_list = self.extract_watermark(stego4ext, flatten_auxbits)
        used_lsb_bits, ext_bits = self.decode_bitswithlength(wm_list, pos="prefix")
        lsb_bits[:len(used_lsb_bits)] = used_lsb_bits
        rec_img4locmap = self.insert_lsb(img4locmap, lsb_bits)
        rec_cover = self.merge_img(rec_img4emb, rec_img4locmap, split_height)
        return iscontinue, rec_cover, ext_bits

    def extract_watermark(self, stego4ext: ndarray, auxbits: List[List]):
        """
        Extract the watermark from the stego image by recovering auxiliary information
        and using reversible error prediction techniques.

        Args:
            stego4ext: The stego image from which the watermark is extracted.
            auxbits: A list of auxiliary bits used for the extraction process.

        Returns:
            stego4ext: The final stego image after watermark extraction.
            wm_list: The extracted watermark bits.
        """
        if stego4ext.dtype == np.float32:
            stego4ext = stego4ext.astype(np.float32)
        wm_list = []  # Initialize an empty list to store the extracted watermark
        # Decode the auxiliary information for the current bit
        decode_info = self.decode_auxiliary_information(auxbits)
        overflow_map_mask_o, min_v_o, max_v_o, stop_coor_o, overflow_map_mask_x, min_v_x, max_v_x, stop_coor_x = decode_info
        # Recover the overflow maps from the stego image
        stego4ext = self.recovery_overflow_stego_image(stego4ext, self.mask_o, overflow_map_mask_o)
        stego4ext = self.recovery_overflow_stego_image(stego4ext, self.mask_x, overflow_map_mask_x)
        if stop_coor_x != (0, 0, 0):
            # Predict the errors (pe) and predictions (pv) for the recovered stego image
            pe_o, pe_x, pv_o, pv_x = self.predicting_error(stego4ext)
            # Extract the watermark bits from the error predictions (pe) for both overflow regions
            recovered_pe_x, wm_list_x = self.extract_and_shift(pe_x, min_v_x, max_v_x, self.mask_x, stop_coor_x)
            # Recompute the stego image using the recovered predicting errors (pe) and predictions (pv)
            stego4ext = self.compute_stego_img(pe_o, recovered_pe_x, pv_o, pv_x)
            wm_list = wm_list_x + wm_list
        # Predict the errors and predictions for the new stego image after watermark recovery
        pe_o, pe_x, pv_o, pv_x = self.predicting_error(stego4ext)
        # Extract the watermark bits again for the other overflow region
        recovered_pe_o, wm_list_o = self.extract_and_shift(pe_o, min_v_o, max_v_o, self.mask_o, stop_coor_o)
        # Recompute the stego image using the recovered predicting errors (pe) for both overflow regions
        recovered_img4embed = self.compute_stego_img(recovered_pe_o, pe_x, pv_o, pv_x)
        # Combine the watermark bits extracted from both overflow regions
        wm_list = wm_list_o + wm_list
        return recovered_img4embed, wm_list


class CustomRDH:
    def __init__(self, img_size: Tuple[int, int, int], device: str = "cpu"):
        """
        Initializes the CustomRDH class for reversible data hiding (RDH).

        :param img_size: The dimensions of the image as (Height, Width, Channels), e.g., (256, 256, 3).
        :param height_end: The height limit for embedding the watermark, default is 5. This controls
                           how many rows from the top will be used for embedding.
        :param device: Specifies the computation device ("cuda" for GPU or "cpu").
        """
        self.device = device
        self.rdh = RDH(img_size)

    def embed(self, cover_img: Tensor, watermark_list: list):
        """
        Embeds a watermark into the cover image using reversible data hiding.

        :param cover_img: The cover image as a tensor of shape (N, C, H, W), where N=1 (one image),
                          C=channels, H=height, and W=width.
        :param watermark_list: List of bits or data to embed within the cover image.
        :return: A tuple containing:
                 - capacity: The capacity used in the image for embedding (measured in bits or similar units).
                 - rdh_stego_img: The output stego image containing the embedded watermark.
        """
        N, C, H, W = cover_img.shape
        assert N == 1  # Ensure a single image in the batch
        # Convert the tensor to a NumPy array for RDH embedding
        cover_img_numpy = cover_img.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
        # Embed watermark using RDH method
        issuccessful, rdh_stego_img = self.rdh.embed(cover_img_numpy, watermark_list)
        return issuccessful, np.uint8(rdh_stego_img)

    def extract(self, rdh_stego_img: ndarray):
        """
        Extracts the embedded watermark and reconstructs the cover image.

        :param rdh_stego_img: The stego image (NumPy ndarray) with the embedded watermark.
        :return: A tuple containing:
                 - rec_stego_img_tensor: The reconstructed stego image as a tensor on the specified device.
                 - wm_list: The extracted watermark list.
        """
        # Use RDH method to extract watermark and reconstruct the stego image
        issuccessful, rec_stego_img, wm_list = self.rdh.extract(rdh_stego_img)
        # Convert the reconstructed image to a tensor and move it to the specified device
        if rdh_stego_img.ndim == 3:
            rec_stego_img_tensor = torch.as_tensor(rec_stego_img, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
        else:
            rec_stego_img_tensor = torch.as_tensor(rec_stego_img, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        return issuccessful, rec_stego_img_tensor.to(self.device), wm_list


def calculate_psnr(original: np.ndarray, compressed: np.ndarray) -> float:
    """
    Calculate the PSNR (Peak Signal-to-Noise Ratio) between two images.

    :param original: The original image as a numpy ndarray.
    :param compressed: The compressed/reconstructed image as a numpy ndarray.
    :return: The PSNR value in decibels (dB).
    """
    # Ensure the two images have the same shape
    assert original.shape == compressed.shape, "Input images must have the same dimensions"

    # Calculate MSE (Mean Squared Error)
    mse = np.mean((original - compressed) ** 2)

    # If MSE is zero, return infinity (no error)
    if mse == 0:
        return float('inf')

    # MAX value for 8-bit images
    max_value = 255.0

    # Calculate PSNR
    psnr = 10 * np.log10(max_value ** 2 / mse)
    return psnr


def test4grayimage(img_path):
    # Open the cover image in grayscale mode
    cover_img = Image.open(img_path).convert("L")  # Convert to grayscale
    cover_img = np.float32(cover_img)
    cover_img_np = cover_img.reshape(cover_img.shape[0], cover_img.shape[1], 1)  # Reshape for single-channel

    # Initialize RDH with the image size
    rdh = RDH(img_size=cover_img_np.shape, height_end=10)

    # Generate a random watermark
    watermark_length = 50000
    watermark_list = [random.randint(0, 1) for _ in range(watermark_length)]

    # Embed the watermark into the cover image
    issuccessful, stego_img = rdh.embed(cover_img_np, watermark_list)
    # stego_img[0:20, :] = cover_img[0:20, :]
    # Save the watermarked image
    Image.fromarray(np.uint8(stego_img)).save('images/stego_img_gray.png')
    reload_stego_img = np.float32(Image.open('images/stego_img_gray.png'))

    # Extract the watermark from the stego image
    recovered_cover_img, wm_list = rdh.extract(reload_stego_img)

    # Check if recovered image and watermark match the originals
    print(np.array_equal(recovered_cover_img, cover_img_np), np.array_equal(watermark_list, wm_list))

    # Calculate and print PSNR
    print(f"PSNR: {calculate_psnr(cover_img, stego_img)}")
    print("Watermark embedding successful.")


def test4rgbimage(img_path):
    # Open the cover image in grayscale mode
    cover_img = Image.open(img_path)  # Convert to grayscale
    cover_img_np = np.float32(cover_img)

    # Initialize RDH with the image size
    rdh = RDH(img_size=cover_img_np.shape)

    # Generate a random watermark
    watermark_length = 10000
    watermark_list = [random.randint(0, 1) for _ in range(watermark_length)]

    # Embed the watermark into the cover image
    issuccessful, stego_img = rdh.embed(cover_img_np, watermark_list)

    # Save the watermarked image
    Image.fromarray(np.uint8(stego_img)).save('stego_img_rgb.tif')
    reload_stego_img = np.float32(Image.open('stego_img_rgb.tif'))

    # Extract the watermark from the stego image
    reload_stego_img_ = reload_stego_img.copy()
    reload_stego_img_[100, 12, 0] = reload_stego_img_[0, 12, 0] + 1
    issuccessful, recovered_cover_img_, wm_list_ = rdh.extract(reload_stego_img_)
    issuccessful, recovered_cover_img, wm_list = rdh.extract(reload_stego_img)

    print(np.array_equal(recovered_cover_img_, recovered_cover_img), np.array_equal(wm_list_, wm_list))

    print(np.where(recovered_cover_img != cover_img_np))

    print(len(wm_list), wm_list[:20], wm_list[-20:])
    print(len(watermark_list), watermark_list[:20], watermark_list[-20:])
    print(np.where(np.array(watermark_list) != np.array(wm_list)))
    print(np.sum(np.abs(cover_img_np - recovered_cover_img)))
    # Check if recovered image and watermark match the originals
    print(np.array_equal(recovered_cover_img, cover_img_np), np.array_equal(watermark_list, wm_list))
    # Calculate and print PSNR
    print(f"PSNR: {calculate_psnr(cover_img_np, stego_img)}")
    print("Watermark embedding successful.")


if __name__ == "__main__":
    test4rgbimage('images/BaboonRGB.tif')
    # test4grayimage('compressor/images/BaboonRGB.tif')
