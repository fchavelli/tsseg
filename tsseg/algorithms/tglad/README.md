# tGLAD

Wrapper around the [tGLAD](https://github.com/Harshs27/tGLAD) method.
tGLAD detects change points by comparing successive Graphical Lasso precision
matrices estimated on sliding windows. A significant change in the graph
structure indicates a regime transition.

Requires `torch` and `glad` (vendored in `tsseg.misc.tGLAD-main`).