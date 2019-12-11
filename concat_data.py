import pandas as pd
import os
from pathlib import Path

exp = 'lognormal_unobs'
path = Path('./results')

csvs = [x for x in os.listdir(path / exp) if '.csv' in x]

res = pd.DataFrame()
for c in csvs:
    tmp = pd.read_csv(path / exp / c)
    # print(tmp.head())
    res = pd.concat([res, tmp], axis=0)

res.to_csv(path / f'{exp}_all.csv', index=False)