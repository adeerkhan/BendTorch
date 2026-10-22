"""torch/main.bend's erf, operation for operation in numpy float32 (IEEE single additions and multiplications, no fused
multiply-add), for the exact tier of test_ops.py: the Bend program must give these bits. The coefficients are copied
from main.bend on purpose, so that a change to either side shows. Over all 2^32 inputs this algorithm is at most 0.93
ulp from float64's erf (measured with a C copy of the same operations against glibc's double erf).
"""

import numpy as np

f = np.float32
# |x| < 0.84375: erf(x) = x + x P(x^2), P's coefficients from the constant term up
P_SMALL = [0.12837916612625122, -0.37612634897232056, 0.11283732205629349, -0.026861993595957756,
           0.005208558402955532, -0.000823960464913398, 8.829222497297451e-05]
# [lo, next lo): erf(a) = K + S(a - c), S's coefficients from the constant term up
PIECES = [
    (0.84375, 1.0, 0.8427007794380188,
     [1.3508899776581984e-08, 0.41510748863220215, -0.41510751843452454, 0.13837093114852905, 0.06918490678071976,
      -0.06929104775190353, 0.004778923466801643, 0.017144912853837013, -0.010648881085216999]),
    (1.25, 1.5, 0.9661051630973816,
     [-1.6622220044837377e-08, 0.11893028765916824, -0.17839543521404266, 0.1387522667646408, -0.04459875077009201,
      -0.014879402704536915, 0.019323812797665596, -0.004488371312618256, -0.0023522074334323406]),
    (1.75, 2.0, 0.9953222870826721,
     [-2.2064206461891445e-08, 0.02066698484122753, -0.04133396968245506, 0.04822301119565964, -0.03444507718086243,
      0.013088003732264042, 0.00046277992078103125, -0.0033736424520611763, 0.0015402256976813078]),
    (2.25, 2.5, 0.9995930194854736,
     [2.846470792405853e-08, 0.002178284339606762, -0.005445694085210562, 0.008350063115358353, -0.008623690344393253,
      0.00611893879249692, -0.002765072975307703, 0.0005077228415757418]),
    (2.75, 3.0, 0.9999778866767883,
     [2.2842012725732275e-08, 0.00013925303937867284, -0.0004177672672085464, 0.0007891026325523853,
      -0.0010437482269480824, 0.001016448251903057, -0.0007547215791419148, 0.0003925197524949908]),
    (3.25, 3.5, 0.9999992847442627,
     [-2.7855115547481546e-08, 5.400151167123113e-06, -1.8894650565925986e-05, 4.221926792524755e-05,
      -6.778862734790891e-05, 8.422814426012337e-05, -7.95367595856078e-05, 4.151461325818673e-05]),
]
ONE_FROM = 3.92


def horner(coefs, t):
    s = np.full_like(t, f(coefs[-1]))
    for c in reversed(coefs[:-1]):
        s = f(c) + t * s
    return s


def erf(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    with np.errstate(all="ignore"):
        a = np.abs(x)
        z = x * x
        out = x + x * horner(P_SMALL, z)          # the small piece, also NaN's and both zeros'
        for lo, c, k, coefs in PIECES:            # later pieces overwrite earlier ones, as the tests run top down
            e = f(k) + horner(coefs, a - f(c))
            out = np.where(a >= f(lo), np.where(x < 0, -e, e), out)
        out = np.where(a >= f(ONE_FROM), np.where(x < 0, f(-1.0), f(1.0)), out)
    return out.astype(np.float32)
