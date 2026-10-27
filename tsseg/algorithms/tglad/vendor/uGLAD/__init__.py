"""Vendored copy of the uGLAD library (non-commercial license).

Copied from the ``uGLAD/`` folder of https://github.com/Harshs27/tGLAD
(imports made relative); the standalone uGLAD project, whose code has since
diverged, is https://github.com/Harshs27/uGLAD.

Changes to ``uglad.py`` (marked ``# tsseg:``): ``uGLAD_multitask.fit`` prints
its three progress lines only when ``verbose`` is true, and
``run_uGLAD_multitask`` prints the loss every ``max(1, int(EPOCHS/10))``
epochs instead of every ``int(EPOCHS/10)``, a modulo by zero below 10 epochs;
the docstring of ``get_final_precision_from_batch`` is a raw string, as its
``\\Theta`` is an invalid escape sequence (a ``SyntaxWarning`` at import from
Python 3.12).
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
