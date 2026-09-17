"""Mamba-3 SISO (trap + RoPE) and MIMO (rank-R) cores and blocks."""

from torchmamba.mamba3.mimo_block import Mamba3MimoBlock
from torchmamba.mamba3.mimo_core import mimo_as_r2_sisos, mimo_chunkwise, mimo_loop
from torchmamba.mamba3.siso_block import Mamba3SisoBlock, Mamba3SisoCache
from torchmamba.mamba3.siso_core import trap_rope_loop, trap_rope_matrix

__all__ = [
    "Mamba3SisoBlock",
    "Mamba3SisoCache",
    "Mamba3MimoBlock",
    "trap_rope_loop",
    "trap_rope_matrix",
    "mimo_loop",
    "mimo_chunkwise",
    "mimo_as_r2_sisos",
]
