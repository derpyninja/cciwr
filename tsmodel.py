import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.stattools import acf, pacf
from dateutil.relativedelta import relativedelta


class TimeSeriesModel(object):
    def __init__(self, ts):
        assert type(ts) is pd.Series, "Class expects a pd.Series as input."
        self.ts = ts
        self.freq = self.ts.index.freq
        self.y = ts.values.flatten()
        self.idx = ts.index.values # store datetime index
        self.slen = ts.shape[0] # length of SI data
        self.x = np.arange(1, self.slen + 1, 1) # discrete vector

        # set seed
        np.random.seed(42)

    def fit(self, ar_order=1, ar_trend='n', pdeg=1, nlags=30):
        """fit the model parameters"""
        # TODO: is this appropriate, or should I only 'self parameters'
        #  inside __init__?
        self.ar_order = ar_order
        self.ar_trend = ar_trend

        # np.polyfit returns polynomial coefficients with the highest power first
        self.model_params = np.polyfit(x=self.x, y=self.y, deg=pdeg).flatten()

        # calculate trend: linear for pdeg=1, quadratic for pdeg=2, etc...
        self.trend = np.polyval(p=self.model_params, x=self.x)
        self.residuals = self.y - self.trend  # residuals

        # AR
        self.pacf = pacf(self.residuals, nlags=nlags)
        self.alpha = self.pacf[self.ar_order]

        # TODO: calculate sd based on the formula above to circumvent using the AutoReg class at all (!)
        AR = AutoReg(endog=self.residuals,
                     lags=self.ar_order, trend=self.ar_trend) # fit AR process
        ARfit = AR.fit(cov_type="HC0") # robust SE
        html = ARfit.summary().as_html() # save model results
        self.sd = float(pd.read_html(html)[0].iloc[2,3]) # get sd
        #self.alpha = float(pd.read_html(html)[1].iloc[1,1]) # get autocorrelation

        self.trend = np.reshape(self.trend, (-1, 1))  # reshape to column vector of shape (n, 1)
        return self

    def monte_carlo(self, n=1):
        """Do n simulations."""
        self.out_shape = (self.slen, n)

        # trend matrix: trend can follow a polynomial of any degree pdeg
        trend_mat = np.tile(self.trend, (1, n))

        # white noise matrix
        white_noise = np.random.normal(0, self.sd, size=self.out_shape)

        # red noise matrix: fill iteratively
        red_noise = np.empty_like(white_noise)
        for pos in np.arange(1, self.slen):
            red_noise[pos,:] = self.alpha * red_noise[pos-1,:] + white_noise[pos,:]

        # compute sum
        xt = trend_mat + red_noise

        # to dataframe
        self.simulation = pd.DataFrame(data=xt, index=self.idx,
                                       columns=np.arange(1, xt.shape[1] + 1))
        return self

    def extrapolate(self, until, n=1):
        """Extrapolate time series based on the fitted AR model."""

        # construct index
        last_obs = self.ts.index[-1]
        first_extrp = last_obs + relativedelta(months=1)
        new_idx = pd.date_range(first_extrp, until, freq='M')
        combined_idx = pd.date_range(self.ts.index[0], until, freq='M')

        # extrapolate trend
        new_len = len(new_idx) + self.slen
        new_x = np.arange(1, new_len + 1)[self.slen:]
        trd_extrp = np.polyval(p=self.model_params, x=new_x)
        trd_extrp.shape = (len(trd_extrp), 1)
        self.slen_extrp = len(new_idx)
        out_shape_extrp = (self.slen_extrp, n)

        # simulate AR process
        trend_mat = np.tile(trd_extrp, (1, n))

        # extrapolate stochastic components
        white_noise = np.random.normal(0, self.sd, size=out_shape_extrp)

        # create (1, n) array of the last observations residual
        resid_last_obs = self.residuals[0]
        last_obs_array = np.repeat(resid_last_obs, n)

        # create empty red noise matrix
        red_noise = np.empty_like(white_noise)

        # initialise AR process with the last observation
        red_noise = np.vstack((last_obs_array, red_noise))

        # red noise matrix: fill iteratively
        for pos in np.arange(1, self.slen_extrp):
            red_noise[pos,:] = (self.alpha
                                * red_noise[pos-1,:]
                                + white_noise[pos,:])

        # compute sum
        xt = trend_mat + red_noise[1:,]


        # combine with observations
        observations = np.tile(np.reshape(self.y, (-1, 1)), (1, n))
        combined_series = np.concatenate((observations, xt), axis=0)

        # to series
        self.extrapolation = pd.DataFrame(
            data=combined_series,
            index=combined_idx,
            columns=np.arange(1, combined_series.shape[1] + 1))
        return self

    def plot(self, what='obs'):
        """Plot obs, sim or extrp"""
        if what not in ['obs', 'sim', 'extrp']:
            raise IOError("Data does not exist.")

        if what == 'obs':
            data = self.ts
        elif what == 'sim':
            data = self.simulation
        else:
            data = self.extrapolation

        # compute statistics
        if data.shape[1] > 1:
            qs = [0, 0.025, 0.25, 0.5, 0.75, 0.975, 1]
            self.quantiles = data.quantile(q=qs, axis=1).transpose()

            # plot simulation results
            # ------------------------------
            f, ax = plt.subplots(figsize=(10,5))
            # median
            self.quantiles[0.5].plot(ax=ax, color='blue', alpha=0.3)

            # observations
            ts_obs = pd.Series(data=np.ravel(self.y), index=self.idx)
            ts_obs.name = 'obs'
            ts_obs.plot(ax=ax, color='red', alpha=0.5)

            # min - max
            ax.fill_between(self.quantiles.index.values,
                            self.quantiles[0],
                            self.quantiles[1],
                            color='black', alpha=0.1)

            # 2.5% - 97.5%
            ax.fill_between(self.quantiles.index.values,
                            self.quantiles[0.025],
                            self.quantiles[0.975],
                            color='black', alpha=0.2)

            # 25% - 75%
            ax.fill_between(self.quantiles.index.values,
                            self.quantiles[0.25],
                            self.quantiles[0.75],
                            color='black', alpha=0.3)

        else:
            f, ax = plt.subplots(figsize=(10,5))
            data.plot(ax=ax, color='blue', alpha=0.3)

        plt.legend()
        return ax
