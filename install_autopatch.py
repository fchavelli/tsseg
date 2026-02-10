#!/usr/bin/env python3
"""
Installation script for tsseg automatic compatibility patches.
Run this script to enable automatic numpy compatibility patches for pyhsmm/pybasicbayes.

Usage: python install_autopatch.py
"""

import os

def patch_pyhsmm():
    """Patch pyhsmm files for NumPy 2.0 compatibility."""
    try:
        # Try to find pyhsmm path without importing (to avoid import errors)
        import site
        import glob
        
        pyhsmm_path = None
        for sp in site.getsitepackages() + [site.getusersitepackages()]:
            if sp and os.path.exists(sp):
                potential_path = os.path.join(sp, 'pyhsmm')
                if os.path.exists(potential_path):
                    pyhsmm_path = potential_path
                    break
        
        if not pyhsmm_path:
            print("ℹ pyhsmm not found, skipping pyhsmm patches")
            return False
        
        # Patch pyhsmm/util/stats.py
        stats_file = os.path.join(pyhsmm_path, 'util', 'stats.py')
        if os.path.exists(stats_file):
            with open(stats_file, 'r') as f:
                content = f.read()
            
            original_content = content
            
            # Replace inner1d import
            if 'from numpy.core.umath_tests import inner1d' in content:
                content = content.replace(
                    'from numpy.core.umath_tests import inner1d',
                    "inner1d = lambda x, y: np.einsum('ij,ij->i', x, y)"
                )
            
            if content != original_content:
                with open(stats_file, 'w') as f:
                    f.write(content)
                print(f"✓ Patched {stats_file}")
            else:
                print(f"ℹ {stats_file} already patched or doesn't need patching")
        
        return True
        
    except Exception as e:
        print(f"✗ Error patching pyhsmm: {e}")
        return False

def patch_pybasicbayes():
    """Patch pybasicbayes files for NumPy 2.0 compatibility."""
    try:
        # Try to find pybasicbayes path without importing (to avoid import errors)
        import site
        
        pybasicbayes_path = None
        for sp in site.getsitepackages() + [site.getusersitepackages()]:
            if sp and os.path.exists(sp):
                potential_path = os.path.join(sp, 'pybasicbayes')
                if os.path.exists(potential_path):
                    pybasicbayes_path = potential_path
                    break
        
        if not pybasicbayes_path:
            print("ℹ pybasicbayes not found, skipping pybasicbayes patches")
            return False
        
        patched_files = 0
        
        # Patch pybasicbayes/util/stats.py
        stats_file = os.path.join(pybasicbayes_path, 'util', 'stats.py')
        if os.path.exists(stats_file):
            with open(stats_file, 'r') as f:
                content = f.read()
            
            original_content = content
            
            # Replace inner1d import
            if 'from numpy.core.umath_tests import inner1d' in content:
                content = content.replace(
                    'from numpy.core.umath_tests import inner1d',
                    "inner1d = lambda x, y: np.einsum('ij,ij->i', x, y)"
                )
            
            # Replace np.Inf with np.inf
            if 'np.Inf' in content:
                content = content.replace('np.Inf', 'np.inf')

            if content != original_content:
                with open(stats_file, 'w') as f:
                    f.write(content)
                print(f"✓ Patched {stats_file}")
                patched_files += 1
            else:
                print(f"ℹ {stats_file} already patched or doesn't need patching")
        
        # Patch pybasicbayes/distributions/gaussian.py
        gaussian_file = os.path.join(pybasicbayes_path, 'distributions', 'gaussian.py')
        if os.path.exists(gaussian_file):
            with open(gaussian_file, 'r') as f:
                content = f.read()
            
            original_content = content
            
            # Replace inner1d import
            if 'from numpy.core.umath_tests import inner1d' in content:
                content = content.replace(
                    'from numpy.core.umath_tests import inner1d',
                    "inner1d = lambda x, y: np.einsum('ij,ij->i', x, y)"
                )
            
            if content != original_content:
                with open(gaussian_file, 'w') as f:
                    f.write(content)
                print(f"✓ Patched {gaussian_file}")
                patched_files += 1
            else:
                print(f"ℹ {gaussian_file} already patched or doesn't need patching")
        
        return patched_files > 0
        
    except Exception as e:
        print(f"✗ Error patching pybasicbayes: {e}")
        return False

def install_autopatch():
    """Install automatic compatibility patches."""
    print("Installing NumPy 2.0 compatibility patches...")
    
    # Patch installed libraries directly
    pyhsmm_patched = patch_pyhsmm()
    pybasicbayes_patched = patch_pybasicbayes()
    
    if pyhsmm_patched or pybasicbayes_patched:
        print("\n✓ Patches installed successfully!")
        print("  You can now import pyhsmm and pybasicbayes without NumPy compatibility issues.")
        return True
    else:
        print("\n⚠ No libraries were patched. Make sure pyhsmm and/or pybasicbayes are installed.")
        return False

if __name__ == "__main__":
    install_autopatch()
