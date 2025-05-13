#!/usr/bin/env python3
"""
Cumulative sum objects and Fenwick trees for fast operations
============================================================

Fenwick tree and CumulativeSum classes designed to work with adaptive FED_MODEL.

In an adaptive model we do not use frequencies like

>>> frequencies = {'a': 4, 'b': 1, 'c': 3}

in the arithmetic coder. Instead we provide a list of symbols like

>>> symbols = ['a', 'b', 'c']

and set each count to one.

>>> frequencies = {symbol:1 for symbol in frequencies}

A cumulative sum object is updated as the model see more of each symbol.

>>> cumsum = CumulativeSum(frequencies)
>>> cumsum.get_low_high('a')
(0, 1)
>>> cumsum.add_count('a', 1)
>>> cumsum.get_low_high('a')
(0, 2)
>>> cumsum.get_low_high('b')
(2, 3)

By using a Fenwick tree we can get O(log n) time operations for getting and
setting counts as symbols are seen. Since the access pattern of the encoder
is to alternate between getting symbol counts and updating them, this gives
O(log n) performance instead of O(n). In practice n is the unique number of
symbols, which is not a large value, so this does not matter that much.
Still nice to use a data structure with good asymptotic performance though.


"""

import copy
import random
import itertools

import numpy as np
from numpy import ndarray


class FenwickTree:
    """A data structure for maintaining cumulative (prefix) sums.
    All operations are O(log n).

    This implementation is based on: https://github.com/dstein64/fenwick

    Examples
    --------
    >>> frequencies = [1, 0, 2, 1, 1, 3, 0, 4]
    >>> ft = FenwickTree(frequencies)
    """

    def __init__(self, frequencies):
        """Initializes n frequencies to zero."""
        self._v = list(frequencies)

        # Initialize in O(n) with specified frequencies.
        for idx in range(1, len(self) + 1):
            parent_idx = idx + (idx & -idx)  # parent in update tree
            if parent_idx <= len(self):
                self._v[parent_idx - 1] += self._v[idx - 1]

    def __len__(self):
        return len(self._v)

    def prefix_sum(self, stop):
        """Returns sum of first elements (sum up to *stop*, exclusive).

        Examples
        --------
        >>> ft = FenwickTree([1, 0, 2, 1, 1, 3, 0, 4])
        >>> ft.prefix_sum(1) == 1
        True
        >>> ft.prefix_sum(2) == 1 + 0
        True
        >>> ft.prefix_sum(3) == 1 + 0 + 2
        True
        >>> ft.prefix_sum(4) == 1 + 0 + 2 + 1
        True
        """
        if stop <= 0 or stop > len(self):
            raise IndexError("index out of range")
        _sum = 0
        while stop > 0:
            _sum += self._v[stop - 1]
            stop &= stop - 1
        return _sum

    def range_sum(self, start, stop):
        """Returns sum from start (inclusive) to stop (exclusive).

        Examples
        --------
        >>> ft = FenwickTree([1, 0, 2, 1, 1, 3])
        >>> ft.range_sum(0, 3) == 1 + 0 + 2
        True
        >>> ft.range_sum(0, 5) == 1 + 0 + 2 + 1 + 1
        True

        """
        if start < 0 or start >= len(self):
            raise IndexError("index out of range")
        if stop <= start or stop > len(self):
            raise IndexError("index out of range")
        result = self.prefix_sum(stop)
        if start > 0:
            result -= self.prefix_sum(start)
        return result

    def __getitem__(self, idx):
        """Get item value (not cumsum) at index.

        Examples
        --------
        >>> ft = FenwickTree([1, 0, 2, 1, 1, 3, 0, 4])
        >>> ft[0], ft[1], ft[2], ft[3]
        (1, 0, 2, 1)
        >>> ft[-1]
        4
        """
        if isinstance(idx, int):
            idx = idx % len(self)
            return self.range_sum(idx, idx + 1)
        else:
            raise IndexError(f"Indexing only works with integers, got {idx}")

    def frequencies(self):
        """Retrieves all frequencies in O(n).

        Examples
        --------
        >>> ft = FenwickTree([1, 0, 2, 1, 1])
        >>> ft.frequencies()
        [1, 0, 2, 1, 1]
        """
        _frequencies = [0] * len(self)
        for idx in range(1, len(self) + 1):
            _frequencies[idx - 1] += self._v[idx - 1]
            parent_idx = idx + (idx & -idx)
            if parent_idx <= len(self):
                _frequencies[parent_idx - 1] -= self._v[idx - 1]
        return _frequencies

    def add(self, idx, k):
        """Adds k to idx'th element (0-based indexing).

        Examples
        --------
        >>> ft = FenwickTree([1, 0, 2, 1, 1])
        >>> ft.add(0, 2)
        >>> ft.add(3, -4)
        >>> ft.frequencies()
        [3, 0, 2, -3, 1]
        >>> ft.range_sum(0, 4) == 3 + 0 + 2 - 3
        True
        """
        if idx < 0 or idx >= len(self):
            raise IndexError("index out of range")
        idx += 1
        while idx <= len(self):
            self._v[idx - 1] += k
            idx += idx & -idx

    def __setitem__(self, idx, value):
        # It's more efficient to use add directly, as opposed to
        # __setitem__, since the latter calls __getitem__.
        self.add(idx, value - self[idx])

    def bisect_left(self, value):
        """
        Returns the smallest index i such that the cumulative sum up to i is >= value.
        If no such index exists, returns len(self).
        This operation is O(log n).

        Examples
        --------
        >>> ft = FenwickTree([1, 3, 5, 10])
        >>> ft.prefix_sum(4)
        19
        >>> ft.bisect_left(2)
        1
        >>> ft.bisect_left(9)
        3
        >>> ft.bisect_left(1)
        1
        >>> ft.bisect_left(0.5)
        0
        >>> ft.bisect_left(99)
        4
        """
        # https://stackoverflow.com/questions/34699616/fenwick-trees-to-determine-which-interval-a-point-falls-in
        j = 2 ** len(self)

        i = -1
        while j > 0:
            if i + j < len(self) and value >= self._v[i + j]:
                value -= self._v[i + j]
                i += j
            j >>= 1
        return i + 1

    def __eq__(self, other):
        return isinstance(other, FenwickTree) and self._v == other._v


class NaiveCumulativeSum:
    """Cumulative sum with slow asymptotic performance."""

    def __init__(self, frequencies, update=True):
        """Create cumulative sum in O(n) time."""
        self.frequencies = dict(frequencies)
        self.ranges = dict(self.ranges_from_frequencies(self.frequencies))
        self.update = update

    def get_low_high(self, symbol):
        """Get (low, high) for symbol in O(1) time."""
        return self.ranges[symbol]

    def add_count(self, symbol, value):
        """Update count in O(n) time."""
        if self.update:
            self.frequencies[symbol] += value
            self.ranges = dict(self.ranges_from_frequencies(self.frequencies))

    def total_count(self):
        """Get sum of all frequencies in O(n) time."""
        return sum(self.frequencies.values())

    def reset(self):
        """Set all frequency counts to one."""
        self.frequencies = {frequency: 1 for frequency in self.frequencies}
        self.ranges = dict(self.ranges_from_frequencies(self.frequencies))

    @staticmethod
    def ranges_from_frequencies(frequencies):
        """Build a dictionary of ranges from a dictionary of frequencies.

        Examples
        --------
        >>> freq = {'a': 5, 'b': 3, 'c': 2}
        >>> dict(NaiveCumulativeSum.ranges_from_frequencies(freq))
        {'a': (0, 5), 'b': (5, 8), 'c': (8, 10)}
        """
        cumsum = 0
        for symbol, frequency in sorted(frequencies.items()):
            yield (symbol, (cumsum, cumsum + frequency))
            cumsum += frequency

    def search_ranges(self, value):
        """Find symbol such that low <= value < high in O(n) time.

        Examples
        --------
        >>> cumsum = NaiveCumulativeSum({'a': 5, 'b': 3, 'c': 2})
        >>> cumsum.search_ranges(2)
        'a'
        >>> cumsum.search_ranges(5)
        'b'
        """
        for symbol, (low, high) in self.ranges.items():
            if low <= value < high:
                return symbol
        raise ValueError("Could not locate value in ranges.")


class CumulativeSum:
    """Cumulative sum with fast asymptotic performance."""

    def __init__(self, frequencies, update=True):
        """Create cumulative sum in O(n) time."""
        symbols = sorted(frequencies.keys())
        self.idx_to_symbol = dict(enumerate(symbols))
        self.symbol_to_idx = {s: i for (i, s) in self.idx_to_symbol.items()}
        self.fenwick_tree = FenwickTree([frequencies[s] for s in symbols])
        self.update = update

    def get_low_high(self, symbol):
        """Get (low, high) for symbol in O(log n) time.

        Examples
        --------
        >>> cumsum = CumulativeSum({'a': 2, 'b': 3, 'c': 4})
        >>> cumsum.get_low_high('a')
        (0, 2)
        >>> cumsum.get_low_high('b')
        (2, 5)
        >>> cumsum.get_low_high('c')
        (5, 9)
        """
        idx = self.symbol_to_idx[symbol]
        if idx == 0:
            return (0, self.fenwick_tree[idx])

        sum_upto = self.fenwick_tree.prefix_sum(idx)
        return (sum_upto, sum_upto + self.fenwick_tree[idx])

    def add_count(self, symbol, value):
        """Update count in O(log n) time.

        Examples
        --------
        >>> cumsum = CumulativeSum({'a': 2, 'b': 3, 'c': 4})
        >>> cumsum.add_count('b', 2)
        >>> cumsum.get_low_high('a')
        (0, 2)
        >>> cumsum.get_low_high('b')
        (2, 7)
        >>> cumsum.get_low_high('c')
        (7, 11)
        """
        if self.update:
            idx = self.symbol_to_idx[symbol]
            self.fenwick_tree.add(idx, value)

    def total_count(self):
        """Get sum of all frequencies in O(log n) time.

        Examples
        --------
        >>> cumsum = CumulativeSum({'a': 2, 'b': 3, 'c': 4})
        >>> cumsum.total_count()
        9
        >>> cumsum.add_count('c', 2)
        >>> cumsum.total_count()
        11
        """
        return self.fenwick_tree.prefix_sum(len(self.fenwick_tree))

    def reset(self):
        """Set all frequency counts to one."""
        self.fenwick_tree = FenwickTree([1] * len(self.fenwick_tree))

    def search_ranges(self, value):
        """Find symbol such that low <= value < high in O(n) time.

        Examples
        --------
        >>> cumsum = CumulativeSum({'a': 5, 'b': 3, 'c': 2})
        >>> cumsum.search_ranges(2)
        'a'
        >>> cumsum.search_ranges(5)
        'b'
        """
        idx = self.fenwick_tree.bisect_left(value)
        return self.idx_to_symbol[idx]


"""
This module implements a ArithmeticEncoder class for encoding and decoding.


Minimal example
===============

Create a message, which is an iterable consisting of hashable symbols.

>>> message = ['A', 'B', 'B', 'B', '<EOM>']

Create frequency counts - the model needs to know how common each symbol is.
The essential compression idea is that high-frequency symbols get fewers bits.

>>> frequencies = {'A': 1, 'B': 3, '<EOM>': 1}

Now create the encoder and encoe the message.

>>> encoder = ArithmeticEncoder(frequencies=frequencies, bits=6)
>>> bits = list(encoder.encode(message))
>>> bits
[0, 1, 0, 1, 1, 0, 0, 1]

Verify that decoding brings back the original message.

>>> list(encoder.decode(bits))
['A', 'B', 'B', 'B', '<EOM>']


Compression of infrequent symbols
=================================

Here is an example with many common letters. In 'Crime and Punishment' by
Fyodor Dostoyevsky the symbol 'e' is around 136 times more frequent than 'q'.

>>> import random
>>> rng = random.Random(42)
>>> message = rng.choices(['e', 'q'], weights=[136, 1], k=10_000) + ["<EOM>"]
>>> frequencies = {'e': 13600, 'q': 100, '<EOM>': 1}

The 10_000 symbols are compressed to a small number of bits

>>> encoder = ArithmeticEncoder(frequencies=frequencies, bits=16)
>>> bits = list(encoder.encode(message))
>>> len(bits)
676

"""


class BitQueue:
    """A queue to keep track of bits to follow.

    Examples
    --------
    >>> bitqueue = BitQueue()
    >>> bitqueue += 3
    >>> list(bitqueue.bit_plus_follow(0))
    [0, 1, 1, 1]
    >>> bitqueue += 2
    >>> list(bitqueue.bit_plus_follow(1))
    [1, 0, 0]
    """

    bits_to_follow = 0  # Initialize the counter

    def __add__(self, bits):
        self.bits_to_follow += bits  # Add to the counter
        return self

    def bit_plus_follow(self, bit):
        yield bit  # Yield the bit, then `bits_to_follow` of the opposite bit
        yield from itertools.repeat(int(not bit), times=self.bits_to_follow)
        self.bits_to_follow = 0  # Reset the counter


class ArithmeticEncoder:
    """An implementation of arithmetic coding based on:

    - Ian H. Witten, Radford M. Neal, and John G. Cleary. 1987.
      Arithmetic coding for data compression.
      Commun. ACM 30, 6 (June 1987), 520–540.
      https://doi.org/10.1145/214762.214771
    - Data Compression With Arithmetic Coding
      https://marknelson.us/posts/2014/10/19/data-compression-with-arithmetic-coding.html

    This implementation pedagogical, not production ready code.
    You should probably not implement this in Python for real-world use
    cases, since the language is too slow and too high-level.
    """

    def __init__(self, frequencies: list, *, bits=16, verbose=0, EOM="<EOM>"):
        """Initialize an arithmetic encoder/decoder.

        Parameters
        ----------
        frequencies : dict
            A dictionary mapping symbols to frequencies, e.g. {'A':3, 'B':2}.
        bits : int, optional
            The number of bits to use in the buffer. The default is 6.
        verbose : int, optional
            How much information to print. The default is 0.
        EOM : str, optional
            An End Of Message (OEM) symbol. The default is '<EOM>'.

        Examples
        --------
        >>> message = ['A', 'B', 'B', 'B', '<EOM>']
        >>> frequencies = {'A': 1, 'B': 3, '<EOM>': 1}
        >>> encoder = ArithmeticEncoder(frequencies=frequencies, bits=6)
        >>> bits = list(encoder.encode(message))
        >>> bits
        [0, 1, 0, 1, 1, 0, 0, 1]
        >>> list(encoder.decode(bits))
        ['A', 'B', 'B', 'B', '<EOM>']

        Instead of using fixed frequencies, it's possible to use a simple
        dynamic probability model by passing a list of symbols as `frequencies`.
        The initial frequency of every symbol will then be 1, and as the model
        sees each symbol in the message it updates the frequencies. The decoder
        reverses this process.

        >>> message = ['R', 'N', '<EOM>']
        >>> frequencies = list(set(message))
        >>> encoder = ArithmeticEncoder(frequencies=frequencies)
        >>> bits = list(encoder.encode(message))
        >>> list(encoder.decode(bits)) == message
        True

        """
        self.EOM = EOM
        self.frequencies = frequencies.copy()
        self.bits = bits
        self.verbose = verbose

        # Build cumulative sum from frequencies
        if isinstance(frequencies, dict):
            assert all(isinstance(freq, int) for freq in self.frequencies.values())
            assert self.EOM in self.frequencies.keys()
            self.cumsum = CumulativeSum(dict(frequencies), update=False)
        elif isinstance(frequencies, (list, set)):
            assert self.EOM in self.frequencies
            frequencies = {symbol: 1 for symbol in self.frequencies}
            self.cumsum = CumulativeSum(frequencies, update=True)
        # The total range. Examples in comments are with 4 bits
        self.TOP_VALUE = (1 << self.bits) - 1  # 0b1111 = 15
        self.FIRST_QUARTER = (self.TOP_VALUE >> 2) + 1  # 0b0100 = 4
        self.HALF = self.FIRST_QUARTER * 2  # 0b1000 = 8
        self.THIRD_QUARTER = self.FIRST_QUARTER * 3  # 0b1100 = 12

        # Equation on page 533 - check if there is enough precision
        if self.cumsum.total_count() > int((self.TOP_VALUE + 1) / 4) + 1:
            msg = "Insufficient precision to encode low-probability symbols."
            msg += "\nIncrease the value of `bits` in the encoder."
            raise Exception(msg)

        if self.verbose > 0:
            print("Initialized with:")
            print(f" bits          = {self.bits}")
            print(
                f" TOP_VALUE     = 0b{self.TOP_VALUE:0{self.bits}b} ({self.TOP_VALUE})"
            )
            print(
                f" THIRD_QUARTER = 0b{self.THIRD_QUARTER:0{self.bits}b} ({self.THIRD_QUARTER})"
            )
            print(f" HALF          = 0b{self.HALF:0{self.bits}b} ({self.HALF})")
            print(
                f" FIRST_QUARTER = 0b{self.FIRST_QUARTER:0{self.bits}b} ({self.FIRST_QUARTER})"
            )

    def _print_state(self, low, high, value=None, *, prefix=" ", end="\n"):
        range_ = high - low + 1
        print(prefix + f"High value: 0b{high:0{self.bits}b} ({high})")
        if value is not None:
            print(prefix + f"Value:      0b{value:0{self.bits}b} ({value})")
        print(prefix + f"Low value:  0b{low:0{self.bits}b} ({low})")
        print(prefix + f"Range: [{low}, {high + 1}) Width: {range_}", end=end)

    def encode(self, iterable):
        """Encode an iterable of symbols, yielding bits (0/1).

        Examples
        --------
        >>> message = iter(['A', 'B', '<EOM>'])
        >>> frequencies = {'A': 5, 'B': 2, '<EOM>': 1}
        >>> encoder = ArithmeticEncoder(frequencies=frequencies)
        >>> list(encoder.encode(message))
        [1, 0, 0, 1, 1, 0, 1]
        """
        if self.verbose:
            print("------------------------ ENCODING ------------------------")

        # Work with a copy of the cumulative sum, in case encoding/decoding
        # works in lockstep. If we use: encoder.decode(encoder.encode(msg))
        # and we do not take a copy, then encoder/decoder mutate the same obj.
        cumsum = copy.deepcopy(self.cumsum)

        bit_queue = BitQueue()  # Keep track of bits to follow

        # Initial low and high values for the range [low, high)
        low = 0
        high = self.TOP_VALUE

        # Loop over every symbol in the input stream `iterable`
        for i, symbol in enumerate(iterable, 1):
            if self.verbose > 0:
                print(f"\nProcessing symbol number {i}: {repr(symbol)}")
                print("-" * 32)

            # Current range
            range_ = high - low + 1

            # Algorithm invariants
            if range_ < cumsum.total_count():
                msg = "Insufficient precision to encode low-probability symbols."
                msg += "\nIncrease the value of `bits` in the encoder."
                raise Exception(msg)
            assert 0 <= low <= high <= self.TOP_VALUE
            assert low < self.HALF <= high
            assert high - low > self.FIRST_QUARTER

            # Print current state of the low and high values
            if self.verbose > 0:
                self._print_state(low, high, prefix="")

            # Get the symbol counts (non-normalized cumulative probabilities)
            symbol_low, symbol_high = cumsum.get_low_high(symbol)

            # Transform the range [low, high) based on probability of symbol.
            # Note: due to floating point issues, even the order of operations
            # must match EXACTLY between the encoder and decoder here.
            total_count = cumsum.total_count()
            high = low + int(range_ * symbol_high / total_count) - 1
            low = low + int(range_ * symbol_low / total_count)

            # Print state of low and high after transforming
            if self.verbose > 0:
                prob = (symbol_high - symbol_low) / total_count
                print(f"\nTransformed range (prob. of symbol '{symbol}': {prob:.4f}):")
                self._print_state(low, high, prefix="", end="\n\n")

            # This loop will run as long as one of the three cases below happen
            # (1) The first bit in `low` and `high` are both 0 (high < HALF)
            # (2) The first bit in  `low` and `high` are both 1 (low >= HALF)
            # (3) The first two bits in `low` and `high` are opposites
            while True:
                # Case (1): The first bits are both 0
                if high < self.HALF:
                    if self.verbose > 0:
                        print(" Range in lower half - both start with 0")
                        self._print_state(low, high, prefix="   ")
                    # Since HALF > `high` > `low`, both `high` and `low` have
                    # 0 in the first bit. We output this 0 bit.
                    yield from bit_queue.bit_plus_follow(bit=0)

                # Case (2): The first bits are both 0
                elif low >= self.HALF:
                    if self.verbose > 0:
                        print(" Range in upper half  - both start with 1")
                        self._print_state(low, high, prefix="   ")

                    # Since `high` > `low` >= HALF, both `high` and `low` have
                    # 1 as the first bit. We output this 1 bit.
                    yield from bit_queue.bit_plus_follow(bit=1)

                    # HALF is 0b1000..., and we remove the first bit from
                    # both `low` and `high`. An example:
                    # low : 0b10110 => 0b00110
                    # high: 0b11100 => 0b01100
                    low -= self.HALF
                    high -= self.HALF

                # Case (3): The first two bits are opposite
                elif low >= self.FIRST_QUARTER and high < self.THIRD_QUARTER:
                    if self.verbose > 0:
                        print(" Range in middle half - first 2 bits are opposite")
                        self._print_state(low, high, prefix="   ")

                    # At this point we know that `low` is in the second quarter
                    # and `high` is in the third quarter (since the other IF-
                    # statements did not trigger). Therefore the first
                    # two bits in `low` must be 01 and the first two bits in
                    # high must be 10.

                    # FIRST_QUARTER is 0b01000..., so these lines set the first
                    # two bits to 00 in `low` and set 01 in `high`. Example:
                    # low : 0b01xxx => 0b00xxx
                    # high: 0b10xxx => 0b01xxx
                    low -= self.FIRST_QUARTER
                    high -= self.FIRST_QUARTER
                    # The scaling of the bits outside of the IF-statement will
                    # then transform these to
                    # low : 0b01xxx => 0b00xxx => 0b0xxx0
                    # high: 0b10xxx => 0b01xxx => 0b1xxx1
                    # The overall effect is to get rid of the second largest
                    # bit. We don't know the value of this removed bit is untill
                    # the first bit converges to a value. Once the first value
                    # converges and we yield it, we must follow with an opposite
                    # bit. The number of opposite bits are now incremented.
                    bit_queue += 1
                else:
                    break  # Skip the bit shifting below the IF-statement

                # In all three cases above, we scale up bits by shifting every
                # bit to the left, then adding a 0 to `low` and a 1 to `high`.
                # Here is an example:
                # low : 0b00110 => 0b01100
                # high: 0b01100 => 0b11001
                low = 2 * low
                high = 2 * high + 1
                if self.verbose > 0:
                    print("  New values for high and low")
                    self._print_state(low, high, prefix="   ")

            # Increase the frequency of `symbol` if dynamic probability model
            cumsum.add_count(symbol, 1)

        # Check that the last symbol was the End Of Message (EOM) symbol
        if symbol != self.EOM:
            raise ValueError("Last symbol must be {repr(self.EOM)}, got {repr(symbol)}")

        # Finish encoding. Since low < HALF, we resolve ambiguity by yielding
        # bits [0, 1] if low < FIRST_QUARTER, else [1, 0].
        assert low < self.HALF
        bit_queue += 1
        yield from bit_queue.bit_plus_follow(int(low >= self.FIRST_QUARTER))

    def decode(self, iterable):
        """Decode an iterable of bits (0/1), yielding symbols.

        Examples
        --------
        >>> bits = [1, 0, 0, 1, 1, 0, 1]
        >>> frequencies = {'A': 5, 'B': 2, '<EOM>': 1}
        >>> encoder = ArithmeticEncoder(frequencies=frequencies)
        >>> list(encoder.decode(bits))
        ['A', 'B', '<EOM>']
        """
        if self.verbose:
            print("------------------------ DECODING ------------------------")

        cumsum = copy.deepcopy(self.cumsum)  # Take copy

        # Set up low, current value and high
        low = 0
        value = 0
        high = self.TOP_VALUE

        # Consume the first `self.bits` into the `value` variable.
        # For instance, if iterable = [0, 1, 0, 1] and self.bits = 6,
        # then value = 0b010100 after this step
        iterable = enumerate(itertools.chain(iter(iterable), itertools.repeat(0)), 1)
        first_bits = itertools.islice(iterable, self.bits)
        for i, input_bit in first_bits:
            value = (value << 1) + input_bit

        if self.verbose:
            print(f"Consumed the initial {i} bits: 0b{value:0{self.bits}b} ")

        # General loop
        while True:
            if self.verbose:
                print("Current state:")
                self._print_state(low, high, value, prefix=" ", end="\n")

            # Current range and current scaled value
            range_ = high - low + 1
            total_count = cumsum.total_count()
            scaled_value = ((value - low + 1) * total_count - 1) / range_
            symbol = cumsum.search_ranges(scaled_value)
            yield symbol

            # Scale high and low. This mimicks (reverses) the encoder process
            symbol_low, symbol_high = cumsum.get_low_high(symbol)
            total_count = cumsum.total_count()
            high = low + int(range_ * symbol_high / total_count) - 1
            low = low + int(range_ * symbol_low / total_count)

            # Increase the frequency of `symbol` if dynamic probability model
            cumsum.add_count(symbol, 1)

            if self.verbose:
                print(f"After yielding symbol '{symbol}' and scaling:")
                self._print_state(low, high, value, prefix=" ", end="\n\n")

            # The symbol was the End Of Message (EOM) symbol and we are done.
            if symbol == self.EOM:
                break

            while True:
                if high < self.HALF:
                    # All of `high`, `low` and `value` have 0 as the first bit.
                    if self.verbose > 0:
                        print("  Range in lower half - both start with 0")
                    pass
                elif low >= self.HALF:
                    # All of `high`, `low` and `value` have 1 as the first bit.
                    if self.verbose > 0:
                        print("  Range in upper half - both start with 1")
                    value -= self.HALF
                    low -= self.HALF
                    high -= self.HALF
                elif low >= self.FIRST_QUARTER and high < self.THIRD_QUARTER:
                    # Low is in the `second` quarter and `high` is in the third.
                    if self.verbose > 0:
                        print("  Range in middle half - first 2 bits are opposite")

                    value -= self.FIRST_QUARTER
                    low -= self.FIRST_QUARTER
                    high -= self.FIRST_QUARTER
                else:
                    break

                if self.verbose > 0:
                    self._print_state(low, high, value, prefix="   ", end="\n")

                # Shift all bits one to the left, add 0 to low and 1 to high.
                # From the input bit stream (iterable) we read the next bit,
                # and default to 0 if the generator is exhausted.
                low = 2 * low
                high = 2 * high + 1
                i, input_bit = next(iterable, 1)
                value = 2 * value + input_bit
                assert low <= value <= high

                if self.verbose > 0:
                    print(f"  Consumed bit {i}: {input_bit}")
                    self._print_state(low, high, value, prefix="   ", end="\n\n")


class CustomArithmeticEncoder:
    def __init__(self, level_bits_len: int = 10, freq_bits_len: int = 10):
        """
        Initialize the encoder with the specified bit lengths for levels and frequencies.

        :param level_bits_len: Bit length used for encoding integer values.
        :param freq_bits_len: Bit length used for encoding frequency information.
        """
        self.buffer_bits = 30
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


if __name__ == "__main__":
    # An example
    message = np.round(np.random.randn(256, 256, 3)).astype(int)
    encoder = CustomArithmeticEncoder()
    bits = encoder.compress(message)
    print(len(bits))
    ext_data = encoder.decompress(bits).reshape(message.shape)
    print("rec bool:", np.array_equal(message, ext_data))

    # # An example
    # message = np.random.randint(0, 2, size=(256, 256, 3))
    # encoder = CustomArithmeticEncoder()
    # bits = encoder.compress(message, ["1", "0"])
    # ext_data = encoder.decompress(bits, ["1", "0"]).reshape(message.shape)
    # print("rec bool:", np.array_equal(message, ext_data))

# if __name__ == "__main__":
#     # An example
#     random.seed(99)
#     message = []
#     for i in range(256 * 256 * 3):
#         if random.uniform(0, 1) < 0.0014:
#             message.append(f"{random.randint(-3, 5)}")
#         else:
#             message.append("0")
#     message.append("<EOM>")
#     # message = ["B", "A", "A", "A", "<EOM>"]
#     frequencies = list(set(message))
#     print(frequencies)
#     frequencies = ["-3", "-2", "-1", "0", "1", "2", "3", "4", "5", "<EOM>"]
#     print(frequencies)
#     encoder = ArithmeticEncoder(frequencies=frequencies, bits=20)
#     bits = list(encoder.encode(message))
#     print(len(bits))
#     decoded = list(encoder.decode(bits))
#     # print(decoded)
#     assert decoded == message
#
#     # Another example
#     message = ["R", "N", "<EOM>"]
#     frequencies = list(set(message))
#     encoder = ArithmeticEncoder(frequencies=frequencies, bits=14)
#     bits = list(encoder.encode(message))
#     decoded = list(encoder.decode(bits))


# if __name__ == "__main__":
#     import doctest
#
#     doctest.testmod()
