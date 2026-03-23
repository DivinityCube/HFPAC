"""
huffman.py — Huffman Coding for HFPAC
======================================
Huffman coding is an entropy coder — it assigns shorter bit sequences to
values that appear frequently, and longer ones to rare values.

For HFPAC, the input is the LPC residuals from lpc.py. Because LPC does
such a good job of prediction, most residuals are small numbers near zero.
Huffman takes advantage of that distribution to compress them efficiently.

Key concepts:
  - Build a frequency table from the residuals
  - Build a binary tree (lowest-frequency nodes get deepest/longest codes)
  - Assign a bit string (code) to each unique value
  - Encode: replace each residual with its bit string
  - Decode: walk the tree bit-by-bit to recover the original values

This is lossless — no information is discarded.
"""

import heapq
import struct
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Step 1 — The Huffman Tree Node
# ---------------------------------------------------------------------------

@dataclass(order=True)
class HuffmanNode:
    """
    A single node in the Huffman binary tree.

    Nodes are ordered by frequency so Python's heapq (min-heap) can
    efficiently find the two lowest-frequency nodes to merge.

    Leaf nodes carry a `value` (a residual integer).
    Internal nodes have no value — they just connect children.
    """
    freq: int
    value: Optional[int] = field(default=None, compare=False)
    left:  Optional["HuffmanNode"] = field(default=None, compare=False)
    right: Optional["HuffmanNode"] = field(default=None, compare=False)

    @property
    def is_leaf(self) -> bool:
        return self.left is None and self.right is None


# ---------------------------------------------------------------------------
# Step 2 — Build the Huffman Tree
# ---------------------------------------------------------------------------

def build_tree(residuals: List[int]) -> HuffmanNode:
    """
    Build a Huffman tree from a list of integer residuals.

    Algorithm:
      1. Count how often each unique value appears (frequency table)
      2. Create a leaf node for each unique value
      3. Repeatedly merge the two lowest-frequency nodes into a new
         internal node until only one root node remains

    The result is a tree where frequent values sit near the root
    (short codes) and rare values sit at the leaves (long codes).

    Args:
        residuals — list or array of integer LPC residuals

    Returns:
        root HuffmanNode of the completed tree
    """
    freq_table = Counter(residuals)

    # Edge case: only one unique value (e.g. silence or a constant signal).
    # Build a proper two-leaf tree so serialise_tree never sees a None child:
    #   root (internal)
    #   ├── left  leaf (the value)
    #   └── right leaf (the value, duplicate — decoder emits correct symbol either way)
    if len(freq_table) == 1:
        value, freq = next(iter(freq_table.items()))
        left  = HuffmanNode(freq=freq, value=value)
        right = HuffmanNode(freq=freq, value=value)
        return HuffmanNode(freq=freq * 2, left=left, right=right)

    # Seed the heap with one leaf node per unique value
    heap: List[HuffmanNode] = [
        HuffmanNode(freq=freq, value=val)
        for val, freq in freq_table.items()
    ]
    heapq.heapify(heap)

    # Merge until one root remains
    while len(heap) > 1:
        left  = heapq.heappop(heap)   # lowest frequency
        right = heapq.heappop(heap)   # second lowest

        merged = HuffmanNode(
            freq  = left.freq + right.freq,
            left  = left,
            right = right,
        )
        heapq.heappush(heap, merged)

    return heap[0]  # the root


# ---------------------------------------------------------------------------
# Step 3 — Generate the Code Table
# ---------------------------------------------------------------------------

def build_code_table(root: HuffmanNode) -> Dict[int, str]:
    """
    Walk the Huffman tree and assign a binary string code to each leaf value.

    Convention:
        Going left  → append '0'
        Going right → append '1'

    Args:
        root — root HuffmanNode from build_tree()

    Returns:
        code_table — dict mapping residual value → bit string, e.g. {0: '0', 1: '110', -1: '111', ...}
    """
    code_table: Dict[int, str] = {}

    def _traverse(node: HuffmanNode, prefix: str) -> None:
        if node.is_leaf:
            # A tree with a single node gets code '0'
            code_table[node.value] = prefix if prefix else "0"
            return
        if node.left:
            _traverse(node.left,  prefix + "0")
        if node.right:
            _traverse(node.right, prefix + "1")

    _traverse(root, "")
    return code_table


# ---------------------------------------------------------------------------
# Step 4 — Encode residuals to a compact bitstream
# ---------------------------------------------------------------------------

def encode(residuals: List[int], code_table: Dict[int, str]) -> Tuple[bytes, int]:
    """
    Encode a list of integer residuals into a compact byte string.

    Uses numpy.packbits (C-implemented) instead of Python string operations,
    giving a significant speed improvement on large frames.

    Returns:
        (encoded_bytes, num_bits) tuple
    """
    # Join all code strings, then convert to a numpy bit array in one shot.
    # np.frombuffer on the ASCII bytes of '0'/'1' chars, then subtract ord('0'),
    # is much faster than a Python loop over individual characters.
    combined = "".join(code_table[r] for r in residuals)
    num_bits  = len(combined)

    if num_bits == 0:
        return b"", 0

    bits = np.frombuffer(combined.encode(), dtype=np.uint8) - ord("0")

    # Pad to a full byte boundary
    padding = (8 - num_bits % 8) % 8
    if padding:
        bits = np.pad(bits, (0, padding))

    # C-speed bit packing
    packed = np.packbits(bits)
    return bytes(packed), num_bits


# ---------------------------------------------------------------------------
# Step 5 — Decode bytes back to residuals
# ---------------------------------------------------------------------------

def decode(
    encoded_bytes: bytes,
    root: HuffmanNode,
    num_bits: int,
    num_residuals: int,
) -> List[int]:
    """
    Decode a Huffman-encoded byte string back into integer residuals.

    Uses numpy.unpackbits (C-implemented) for byte→bit conversion,
    then walks the Huffman tree over the resulting numpy array.
    Tree traversal remains sequential (inherent to Huffman), but the
    slow Python byte-to-string conversion step is eliminated.

    Args:
        encoded_bytes  — bytes from encode()
        root           — root HuffmanNode (same tree used to encode)
        num_bits       — total valid bits (from encode())
        num_residuals  — expected number of output residuals

    Returns:
        list of integer residuals
    """
    if not encoded_bytes or num_bits == 0:
        return []

    # C-speed byte unpacking — strips padding automatically via [:num_bits]
    bits = np.unpackbits(np.frombuffer(encoded_bytes, dtype=np.uint8))[:num_bits]

    residuals = []
    node = root

    for bit in bits:
        node = node.left if bit == 0 else node.right

        if node.is_leaf:
            residuals.append(node.value)
            node = root

            if len(residuals) == num_residuals:
                break

    return residuals


# ---------------------------------------------------------------------------
# Step 6 — Serialise the Huffman tree (needed to store in .hfpac file)
# ---------------------------------------------------------------------------

def serialise_tree(root: HuffmanNode) -> bytes:
    """
    Serialise the Huffman tree to bytes so it can be stored in the file header.

    Format (written recursively, pre-order traversal):
        Leaf node:     b'\\x01' + 4-byte signed int (the residual value)
        Internal node: b'\\x00' + serialise(left) + serialise(right)

    The decoder reads this back in the same order to reconstruct the tree.
    """
    buf = bytearray()

    def _write(node: HuffmanNode) -> None:
        if node.is_leaf:
            buf.append(0x01)
            buf.extend(struct.pack(">i", node.value))  # big-endian int32
        else:
            buf.append(0x00)
            _write(node.left)
            _write(node.right)

    _write(root)
    return bytes(buf)


def deserialise_tree(data: bytes, offset: int = 0) -> Tuple[HuffmanNode, int]:
    """
    Reconstruct a Huffman tree from serialised bytes.

    Args:
        data   — bytes from serialise_tree()
        offset — starting position in data (for recursive calls)

    Returns:
        (root HuffmanNode, new offset after consuming tree bytes)
    """
    flag = data[offset]
    offset += 1

    if flag == 0x01:
        # Leaf node
        value = struct.unpack(">i", data[offset:offset + 4])[0]
        offset += 4
        return HuffmanNode(freq=0, value=value), offset
    else:
        # Internal node — read left then right subtree
        left,  offset = deserialise_tree(data, offset)
        right, offset = deserialise_tree(data, offset)
        node = HuffmanNode(freq=0, left=left, right=right)
        return node, offset


# ---------------------------------------------------------------------------
# Quick smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    import numpy as np
    sys.path.insert(0, ".")
    from lpc import compute_lpc_coefficients, encode_frame, FRAME_SIZE

    print("=== HFPAC huffman.py smoke test ===\n")

    # Generate a test signal (440 Hz sine, same as lpc.py test)
    sr = 44100
    t  = np.linspace(0, FRAME_SIZE / sr, FRAME_SIZE, endpoint=False)
    signal = (np.sin(2 * np.pi * 440 * t) * 32767).astype(np.float64)

    # Get LPC residuals
    coeffs    = compute_lpc_coefficients(signal)
    residuals = encode_frame(signal, coeffs).tolist()

    # Build Huffman tree + code table
    tree       = build_tree(residuals)
    code_table = build_code_table(tree)

    # Encode
    encoded_bytes, num_bits = encode(residuals, code_table)

    # Decode
    decoded_residuals = decode(encoded_bytes, tree, num_bits, len(residuals))

    # Verify roundtrip
    assert decoded_residuals == residuals, "Huffman roundtrip FAILED!"

    # Serialise / deserialise tree
    tree_bytes           = serialise_tree(tree)
    restored_tree, _     = deserialise_tree(tree_bytes)
    restored_codes       = build_code_table(restored_tree)
    decoded_again        = decode(encoded_bytes, restored_tree, num_bits, len(residuals))
    assert decoded_again == residuals, "Tree serialisation roundtrip FAILED!"

    # Stats
    raw_bits      = len(residuals) * 16        # 16-bit PCM residuals uncompressed
    huffman_bits  = num_bits
    tree_bytes_sz = len(tree_bytes)

    print(f"Residuals:              {len(residuals)} values")
    print(f"Unique symbols:         {len(code_table)}")
    print(f"Shortest code:          {min(len(c) for c in code_table.values())} bits")
    print(f"Longest code:           {max(len(c) for c in code_table.values())} bits")
    print(f"Raw size:               {raw_bits} bits ({raw_bits // 8} bytes)")
    print(f"Huffman payload:        {huffman_bits} bits ({len(encoded_bytes)} bytes)")
    print(f"Tree overhead:          {tree_bytes_sz} bytes")
    print(f"Total compressed:       {len(encoded_bytes) + tree_bytes_sz} bytes")
    print(f"Compression ratio:      {raw_bits / 8 / (len(encoded_bytes) + tree_bytes_sz):.2f}x")
    print(f"\n✅ Huffman encode/decode/serialise roundtrip complete!")