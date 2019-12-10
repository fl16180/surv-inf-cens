import numpy as np
import pandas as pd
from synth_data import SynthCovariateData

from scipy.stats import pearsonr



synth = SynthCovariateData(1000, 2, 2, False)
X, Y, Y_true, C_true = synth.make_linear(tau=0, bias_Y=0.5, bias_C=0.5, sigma_Y=0.01, sigma_C=0.01, rho=0.6)

# import matplotlib.pyplot as plt

# plt.scatter(Y_true, C_true)
# plt.show()
