import numpy as np
import pandas as pd
from synth_data import SynthCovariateData


synth = SynthCovariateData(1000, 5, 3, False)
X, Y, Y_true, C_true = synth.make_linear(tau=0, bias_Y=0.5, bias_C=1, sigma_Y=1, sigma_C=0.8, rho=0)

import matplotlib.pyplot as plt

plt.scatter(Y_true, C_true)
plt.show()