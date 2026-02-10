# Automatic Installation of Compatibility Patches

This package includes an automatic patching system to resolve compatibility issues with NumPy 2.0+ for the dependencies `pyhsmm` and `pybasicbayes`.

## Automatic Activation

Patches are activated automatically in two ways:

### 1. Via the .pth file (recommended)

After installing tsseg, run once:

```bash
python install_autopatch.py
```

This installs a `.pth` file that automatically activates the patches every time Python starts.

### 2. By importing tsseg

Alternatively, simply import `tsseg` before any other imports:

```python
import tsseg  # Automatically activates patches
import pyhsmm  # Now works
import pybasicbayes  # Now works
```

## Applied Patches

The following patches are applied automatically:

1. **numpy.core.umath_tests.inner1d**: Replaced with a compatible implementation using `np.einsum`
2. **np.Inf**: Added for compatibility with NumPy 2.0+
3. **Warnings suppressed**: Warnings about slow implementations are hidden

## Verification

To check that the patches are working:

```python
import pyhsmm
import pybasicbayes
print("All imports work")
```

## Deactivation

To disable automatic patches, remove the file:
```bash
rm $(python -c "import site; print(site.getsitepackages()[0])")/tsseg_autopatch.pth
```
