"""Vendored copy of the uGLAD library (non-commercial license).

The original project is available at https://github.com/Harshs27/uGLAD.
"""

from .uglad import (
	uGLAD_GL,
	uGLAD_multitask,
	forward_uGLAD,
	init_uGLAD,
	loss_uGLAD,
	run_uGLAD_CV,
	run_uGLAD_direct,
	run_uGLAD_missing,
	run_uGLAD_multitask,
)

__all__ = ["uglad", "glad", "uglad_utils"]
