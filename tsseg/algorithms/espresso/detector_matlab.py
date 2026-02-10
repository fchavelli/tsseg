import numpy as np

import matlab.engine


def espresso(ts, window_size, n_cps, chain_length=3, eng=None):
    close_eng = False

    if eng is None:
        eng = matlab.engine.start_matlab()
        s = eng.genpath("../matlab")
        eng.addpath(s, nargout=0)
        close_eng = True

    try:
        cps = eng.ESPRESSO(matlab.double(ts.tolist()), matlab.double([window_size]), matlab.int64([n_cps]), matlab.double([chain_length]))
    except:
        cps = [[]]

    if isinstance(cps, float):
        cps = [int(cps)]
    elif len(cps) > 0:
        cps = [int(_) for _ in cps[0]]
    else:
        cps = []

    if close_eng is True: eng.quit()
    return np.asarray(cps, dtype=np.int64)