import pandas as pd
from ast import literal_eval
# from plotnine import *
import matplotlib.pyplot as plt


logfile = './log.csv'

dat = pd.read_csv(logfile)
dat['cov'] = dat['inf_par'].map(lambda x: literal_eval(x)[2])

# dat2 = pd.melt(dat, id_vars=['cov'], value_vars=['joint', 'lognorm'], var_name='estimator')
# ggplot(dat2) + aes(x='cov', y='value') + geom_point()


tg = dat.groupby('cov', as_index=False).mean()
plt.plot(tg['cov'], tg['joint'], c='C1', label='joint')
plt.plot(tg['cov'], tg['lognorm'], c='C2', label='lognormal')
plt.scatter(dat['cov'], dat['joint'], c='C1')
plt.scatter(dat['cov'], dat['lognorm'], c='C2')

plt.legend()

plt.ylabel('est. E')
plt.xlabel('cov')
plt.show()
# plt.plot(dat['cov'])