"""Vendored copy of the uGLAD library (non-commercial license).

Copied from the ``uGLAD/`` folder of https://github.com/Harshs27/tGLAD
(imports made relative); the standalone uGLAD project, whose code has since
diverged, is https://github.com/Harshs27/uGLAD.
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
