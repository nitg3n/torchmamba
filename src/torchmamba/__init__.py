"""TorchMamba: clean PyTorch reference kernels for Mamba-1/2/3."""

__version__ = "0.1.1"

from torchmamba.mamba1.block import Mamba1Block, Mamba1Cache
from torchmamba.mamba1.core import selective_scan_loop, selective_scan_matrix
from torchmamba.mamba2.block import Mamba2Block
from torchmamba.mamba2.core import (
    boundary_mask_to_logdecay,
    ssd_chunkwise,
    ssd_quadratic,
    ssd_recurrent,
    ssd_step,
)
from torchmamba.mamba3.mimo_block import Mamba3MimoBlock
from torchmamba.mamba3.mimo_core import mimo_as_r2_sisos, mimo_chunkwise, mimo_loop
from torchmamba.mamba3.siso_block import Mamba3SisoBlock, Mamba3SisoCache
from torchmamba.mamba3.siso_core import trap_rope_loop, trap_rope_matrix

__all__ = [
    "Mamba1Block",
    "Mamba1Cache",
    "Mamba2Block",
    "Mamba3SisoBlock",
    "Mamba3SisoCache",
    "Mamba3MimoBlock",
    "selective_scan_loop",
    "selective_scan_matrix",
    "ssd_recurrent",
    "ssd_quadratic",
    "ssd_chunkwise",
    "ssd_step",
    "boundary_mask_to_logdecay",
    "trap_rope_loop",
    "trap_rope_matrix",
    "mimo_loop",
    "mimo_chunkwise",
    "mimo_as_r2_sisos",
]
