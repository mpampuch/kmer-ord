# The standard log-difference formulation eliminates computing exponentiated geometric means explicitly by operating directly in log-space:

import numpy as np
import pandas as pd


def clr_log_diff(X: pd.DataFrame, eps: float = 1e-9) -> pd.DataFrame:
  """Centered Log-Ratio transformation using standard log-differences."""
  log_X = np.log(X + eps)
  return log_X.sub(log_X.mean(axis=1), axis=0)