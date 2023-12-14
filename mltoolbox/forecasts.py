import os
from functools import wraps

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from mltoolbox.misc.generics import SuppressStdoutStderr, timing
from mltoolbox.datasets import dset
from scipy import stats
from sklearn.metrics import mean_absolute_error as MAE
from sklearn.model_selection import train_test_split

try:
    import prophet as fbp
except:
    print(
        'fbprophet package is missing: ForecastProphet class will not be available. \n Use "conda install -c conda-forge fbprophet" to install it in the current env.  '
    )

pd.plotting.register_matplotlib_converters()

class ForecastProphet_old():
    """
    Forecast class, wrapper for fbprophet.
    Additional features:
        - correlation and residual plots
        - data normalization
        - model evaluation with various metrics
        - data transformation logging
    """

    def __init__(self, input_data, input_ds_column='ds', input_y_column='y', dataset='train_set'):
        """

        Parameters
        ----------
        input_data : Pandas Dataframe
            containing a datetime column and a footfall data column
        input_ds_column: String
            Name of Date time column
        input_y_column: String
            Name of footfall data column
        dataset : String
            Can be <train_set> or <test_set>. If Input data is of type <dset>, selected dataset.
        """
        self.dataset=dataset
        if isinstance(input_data,dset):
            if input_ds_column=='index':
                input_data = getattr(input_data, self.dataset)
                print(input_data.head())
                input_data['index'] = input_data.index.to_series()
        self.data = pd.DataFrame(
            input_data[[input_ds_column, input_y_column]].values, columns=['ds', 'y']
        )
        self.data = self.data.astype(dtype={'ds': 'datetime64[ns]', 'y': np.float64})
        self.prophet = fbp.Prophet()
        self.forecast = None
        self.combined = None
        self.history = []

    def log_history(f):
        """
        Decorator for saving a history of called methods during the class instance lifespan
        """

        @wraps(f)
        def wrapper(instance, *args, **kwargs):
            instance.history.append([f.__qualname__.split('.')[1], args, kwargs])
            return f(instance, *args, **kwargs)

        return wrapper

    def apply_history(self, new_instance):
        """
        Apply class history to a new class instance

        Parameters
        ----------
        new_instance: ForecastProphet class

        """
        for call in self.history:
            getattr(new_instance, call[0])(*call[1], **call[2])


    @timing
    def fit(self, verbose=False):
        """
        Fit model to data

        verbose: Bool
            If True will print fitting details to console.
        """
        # self.prophet.fit(self.data)
        if verbose:
            self.prophet.fit(self.data)
        else:
            with SuppressStdoutStderr():
                self.prophet.fit(self.data)

    @timing
    def predict(self, periods=24, freq='H', include_history=True):
        """
        Predicts n periods in the future and stores them in self.forecast.

        Parameters
        ----------
        periods: Integer
            Number of periods to predict
        freq: String
            Frequency definition. Any valid frequency for pd.date_range, such as 'D' or 'M'.
        include_history: Boolean to include the historical dates in the data
            frame for predictions.

        """
        new = self.prophet.make_future_dataframe(
            periods=periods, freq=freq, include_history=include_history
        )
        self.forecast = self.prophet.predict(new)
        if include_history:
            self.combined = pd.merge(self.forecast, self.data, on='ds')

    @log_history
    def add_seasonality(
        self,
        name='monthly',
        period=12 * 30.5,
        fourier_order=5,
        prior_scale=None,
        mode=None,
        condition_name=None,
    ):
        self.prophet.add_seasonality(
            name=name,
            period=period,
            fourier_order=fourier_order,
            prior_scale=prior_scale,
            mode=mode,
            condition_name=condition_name,
        )

    @log_history
    def add_country_holidays(self, country_name='US'):
        self.prophet.add_country_holidays(country_name=country_name)

    def plot_predictions(self):
        """
        Plots predictions and fitted historical data.
        """
        if self.forecast is not None:
            self.prophet.plot(self.forecast)
        else:
            print('Need to predict first.')

    def mean_absolute_percentage_error(self):
        """
        Computes MAPE in percents

        Returns
        -------
        mape: Float
            MAPE in percents.

        """
        mape = (
            np.mean(np.abs((self.combined.y - self.combined.yhat) / self.combined.y))
            * 100
        )
        return mape

    def zscore_filter(self, df=None, varname='y', std=2):
        """
        Z-score filter

        Parameters
        ----------
        df : String
            Name of pandas dataframe in self to filter. If None will use self.data.
        varname : string
            Column to filter in the pandas.DataFrame. Default to 'y'.
        std : integer
            Threshold for the number of std around the median to replace
            by `np.nan`. Default is 3 (greater / less or equal).

        Returns
        -------
        """
        if df is None:
            df = self.data
            attrib_name = 'data'
        else:
            attrib_name = df
            df = getattr(self, df)
        print(
            'Filtering %s.%s at %.2f standard deviations.' % (attrib_name, varname, std)
        )
        df[varname + '_zscore'] = (df[varname] - df[varname].mean()) / df[varname].std(
            ddof=0
        )
        df.loc[df[varname + '_zscore'].abs() > std, varname] = np.nan

    def plot_joint_plot(
        self,
        data='combined',
        x='y',
        y='yhat',
        title=None,
        fpath='../figures',
        fname=None,
    ):
        """

        Parameters
        ----------
        data: String
            Name of Dataframe to plot
        x : string
            The variable on the y-axis
            Defaults to `y`, i.e. the observed values
        y : string
            The variable on the x-axis
            Defaults to `yhat`, i.e. the forecast or estimated values.
        title : string
            The title of the figure, default `None`.

        fpath : string
            The path to save the figures, default to `../figures`
        fname : string
            The filename for the figure to be saved
            ommits the extension, the figure is saved in png, jpeg and pdf

        """
        if getattr(self, data) is None:
            print('Join plot only works when doing in-sample prediction.')
        else:
            g = sns.jointplot(
                x=x, y=y, data=getattr(self, data), kind='reg', color='0.4', dropna=True
            )
            g.fig.set_figwidth(8)
            g.fig.set_figheight(8)
            ax = g.fig.axes[1]
            if title is not None:
                ax.set_title(title, fontsize=16)
            ax = g.fig.axes[0]
            annot_func = lambda a, b: (stats.pearsonr(a, b)[0] ** 2, MAE(a, b))

            g.annotate(
                annot_func,
                template='{stat}= {val:.2f}\nMAE={p:.2f}',
                stat='$R^2$',
                loc='upper left',
                fontsize=12,
            )

            ax.set_xlabel('True values', fontsize=15)
            ax.set_ylabel('Forecasted values', fontsize=15)
            ax.grid(ls=':')
            [l.set_fontsize(13) for l in ax.xaxis.get_ticklabels()]
            [l.set_fontsize(13) for l in ax.yaxis.get_ticklabels()]
            ax.grid(ls=':')

            if fname is not None:
                for ext in ['png', 'jpeg', 'pdf']:
                    g.fig.savefig(
                        os.path.join(fpath, '{}.{}'.format(fname, ext)), dpi=200
                    )

    def plot_residuals(
        self, data='combined', x='y', y='yhat', fpath='../figures', fname=None
    ):
        """
        Plot fit residuals

        Parameters
        ----------
        data: String
            Name of Dataframe to plot
        x : string
            The variable on the y-axis
            Defaults to `y`, i.e. the observed values
        y : string
            The variable on the x-axis
            Defaults to `yhat`, i.e. the forecast or estimated values.
        fpath : string
            The path to save the figures, default to `../figures`
        fname : string
            The filename for the figure to be saved
            ommits the extension, the figure is saved in png, jpeg and pdf

        """
        if getattr(self, data) is None:
            print('Join plot only works when doing in-sample prediction.')
        else:
            tmp = getattr(self, data).copy().dropna()
            residuals = tmp[y] - tmp[x]
            f, ax = plt.subplots(figsize=(8, 8))
            g = sns.distplot(residuals, ax=ax, color='0.4')
            ax.grid(ls=':')
            ax.set_xlabel('Residuals', fontsize=15)
            ax.set_ylabel('Normalised frequency', fontsize=15)
            ax.grid(ls=':')

            [l.set_fontsize(13) for l in ax.xaxis.get_ticklabels()]
            [l.set_fontsize(13) for l in ax.yaxis.get_ticklabels()]

            ax.axvline(0, color='0.4')

            ax.set_title('Residuals distribution', fontsize=17)

            ax.text(
                0.05,
                0.85,
                'Skewness = {:4.3f}\nMedian = {:4.3f}\nMean = {:4.3f}'.format(
                    stats.skew(residuals), residuals.median(), residuals.mean()
                ),
                fontsize=12,
                transform=ax.transAxes,
            )

            if fname is not None:
                for ext in ['png', 'jpeg', 'pdf']:
                    g.fig.savefig(
                        os.path.join(fpath, '{}.{}'.format(fname, ext)), dpi=200
                    )

    def evaluate_train_test(self, test_size=24, verbose=False):
        """
        Evaluate model by splitting in train/test data. Fit train, forecast test, and compare.

        Parameters
        ----------
         test_size : float, int or None, optional (default=0.25)
            If float, should be between 0.0 and 1.0 and represent the proportion
            of the dataset to include in the test split. If int, represents the
            absolute number of test samples. If None, the value is set to the
            complement of the train size.
        verbose: Bool
            If True will print fitting details to console.

        """
        train, test = train_test_split(self.data, test_size=test_size, shuffle=False)
        # train.reset_index(inplace=True)
        # test.reset_index(inplace=True)

        ev = ForecastProphet(input_data=train, input_ds_column='ds', input_y_column='y')
        self.apply_history(ev)

        ev.fit(verbose)
        ev.predict(periods=test.shape[0], freq='H')
        data = pd.concat([train, test], axis=0)
        ev.forecast.loc[:, 'y'] = data.loc[:, 'y']

        # Plot predictions on training and test data
        fig = plt.figure(facecolor='w', figsize=(10, 14))
        gs = fig.add_gridspec(nrows=3, ncols=2)
        # predictions
        ax1 = fig.add_subplot(gs[0, :])
        fcst_t = ev.forecast['ds'].dt.to_pydatetime()
        train_ix = np.where(ev.forecast['ds'] < test.ds.iloc[0])[0]
        test_ix = np.where(ev.forecast['ds'] >= test.ds.iloc[0])[0]
        ax1.plot(fcst_t, ev.forecast['y'], 'k.')
        ax1.plot(fcst_t[train_ix], ev.forecast['yhat'][train_ix], ls='-', c='#0072B2')
        ax1.plot(fcst_t[test_ix], ev.forecast['yhat'][test_ix], ls='-', c='darkorange')
        ax1.fill_between(
            fcst_t[train_ix],
            ev.forecast['yhat_lower'][train_ix],
            ev.forecast['yhat_upper'][train_ix],
            color='#0072B2',
            alpha=0.2,
        )
        ax1.fill_between(
            fcst_t[test_ix],
            ev.forecast['yhat_lower'][test_ix],
            ev.forecast['yhat_upper'][test_ix],
            color='darkorange',
            alpha=0.2,
        )
        [l.set_fontsize(8) for l in ax1.xaxis.get_ticklabels()]
        [l.set_fontsize(8) for l in ax1.yaxis.get_ticklabels()]

        # plot correlation -- need to get rid off nans.
        ev.forecast = ev.forecast.dropna()
        train_ix = np.where(ev.forecast['ds'] < test.ds.iloc[0])[0]
        test_ix = np.where(ev.forecast['ds'] >= test.ds.iloc[0])[0]

        # train
        ax2 = fig.add_subplot(gs[1, 0])
        rsquare = (
            stats.pearsonr(
                ev.forecast.iloc[train_ix]['y'], ev.forecast.iloc[train_ix]['yhat']
            )[0]
            ** 2
        )
        mae = MAE(ev.forecast.iloc[train_ix]['y'], ev.forecast.iloc[train_ix]['yhat'])
        ev.forecast.iloc[train_ix].plot(
            x='y', y='yhat', kind='scatter', c='black', s=5, ax=ax2
        )
        ax2.set_title('Train: R^2=%.2f, MAE=%.2f' % (rsquare, mae), fontsize=11)
        ax2.set_xlabel('True values', fontsize=10)
        ax2.set_ylabel('Forecasted values', fontsize=10)
        ax2.grid(ls=':')
        [l.set_fontsize(10) for l in ax2.xaxis.get_ticklabels()]
        [l.set_fontsize(10) for l in ax2.yaxis.get_ticklabels()]
        ax2.grid(ls=':')

        # test
        ax3 = fig.add_subplot(gs[1, 1])
        rsquare = (
            stats.pearsonr(
                ev.forecast.iloc[test_ix]['y'], ev.forecast.iloc[test_ix]['yhat']
            )[0]
            ** 2
        )
        mae = MAE(ev.forecast.iloc[test_ix]['y'], ev.forecast.iloc[test_ix]['yhat'])
        ev.forecast.iloc[test_ix].plot(
            x='y', y='yhat', kind='scatter', c='black', s=5, ax=ax3
        )
        ax3.set_title('Test: R^2=%.2f, MAE=%.2f' % (rsquare, mae), fontsize=11)
        ax3.set_xlabel('True values', fontsize=10)
        ax3.set_ylabel('Forecasted values', fontsize=10)
        ax3.grid(ls=':')
        [l.set_fontsize(10) for l in ax3.xaxis.get_ticklabels()]
        [l.set_fontsize(10) for l in ax3.yaxis.get_ticklabels()]
        ax3.grid(ls=':')

        # training set residuals
        train_residuals = ev.forecast['yhat'][train_ix] - ev.forecast['y'][train_ix]
        train_residuals = train_residuals.dropna()
        ax4 = fig.add_subplot(gs[2, 0])
        g = sns.distplot(train_residuals, ax=ax4, color='0.4')
        ax4.grid(ls=':')
        ax4.set_xlabel('Residuals', fontsize=10)
        ax4.set_ylabel('Normalised frequency', fontsize=10)
        ax4.grid(ls=':')
        [l.set_fontsize(10) for l in ax4.xaxis.get_ticklabels()]
        [l.set_fontsize(10) for l in ax4.yaxis.get_ticklabels()]
        ax4.axvline(0, color='0.4')
        ax4.set_title('Residuals distribution (train)', fontsize=11)
        ax4.text(
            0.05,
            0.85,
            'Skewness = {:4.3f}\nMedian = {:4.3f}\nMean = {:4.3f}'.format(
                stats.skew(train_residuals),
                train_residuals.median(),
                train_residuals.mean(),
            ),
            fontsize=10,
            transform=ax4.transAxes,
        )
        # test set residuals
        test_residuals = ev.forecast['yhat'][test_ix] - ev.forecast['y'][test_ix]
        test_residuals = test_residuals.dropna()
        ax5 = fig.add_subplot(gs[2, 1])
        g = sns.distplot(test_residuals, ax=ax5, color='0.4')
        ax5.grid(ls=':')
        ax5.set_xlabel('Residuals', fontsize=10)
        ax5.set_ylabel('Normalised frequency', fontsize=10)
        ax5.grid(ls=':')
        [l.set_fontsize(10) for l in ax5.xaxis.get_ticklabels()]
        [l.set_fontsize(10) for l in ax5.yaxis.get_ticklabels()]
        ax5.axvline(0, color='0.4')
        ax5.set_title('Residuals distribution (test)', fontsize=11)
        ax5.text(
            0.05,
            0.85,
            'Skewness = {:4.3f}\nMedian = {:4.3f}\nMean = {:4.3f}'.format(
                stats.skew(test_residuals),
                test_residuals.median(),
                test_residuals.mean(),
            ),
            fontsize=10,
            transform=ax5.transAxes,
        )
        plt.tight_layout()
        plt.show()
        return ev, train, test


class ForecastProphet():
    """
    Forecast class, wrapper for fbprophet.
    Additional features:
        - correlation and residual plots
        - data normalization
        - model evaluation with various metrics
        - data transformation logging
    """

    def __init__(self, input_dset, input_ds_column='ds', input_y_column='y', additional_regressors=None):
        """

        Parameters
        ----------
        input_dset : mltoolbox.datasets.dset
            dset object
        input_ds_column: String
            Name of Date time column
        input_y_column: String
            Name of data column
        additional_regressors: list of strings
            List of columns to be used as additional regressors
        """
        self.input_ds_column = input_ds_column
        self.input_y_column = input_y_column
        self.additional_regressors = additional_regressors
        self.dset = input_dset
        if input_ds_column=='index':
            self.dset.train_set['index'] = self.dset.train_set.index.to_series()
            self.dset.test_set['index']  = self.dset.test_set.index.to_series()
        self.prophet = fbp.Prophet()
        if additional_regressors is not None:
            for ar in self.additional_regressors:
                self.prophet.add_regressor(ar)
        self.forecast = None
        self.combined = None
        self.history = []

    def log_history(f):
        """
        Decorator for saving a history of called methods during the class instance lifespan
        """

        @wraps(f)
        def wrapper(instance, *args, **kwargs):
            instance.history.append([f.__qualname__.split('.')[1], args, kwargs])
            return f(instance, *args, **kwargs)

        return wrapper

    def apply_history(self, new_instance):
        """
        Apply class history to a new class instance

        Parameters
        ----------
        new_instance: ForecastProphet class

        """
        for call in self.history:
            getattr(new_instance, call[0])(*call[1], **call[2])


    @timing
    def fit(self, verbose=False):
        """
        Fit model to train_set

        verbose: Bool
            If True will print fitting details to console.
        """
        # self.prophet.fit(self.data)
        if self.additional_regressors is None:
            self.additional_regressors = []
        data = pd.DataFrame(
            self.dset.train_set[[self.input_ds_column, self.input_y_column]+ self.additional_regressors].values, columns=['ds', 'y'] + self.additional_regressors
        )
        data = data.astype(dtype={'ds': 'datetime64[ns]', 'y': np.float64})

        if verbose:
            self.prophet.fit(data)
        else:
            with SuppressStdoutStderr():
                self.prophet.fit(data)

    @timing
    def predict(self, forecast_schedule=None, periods=24, freq='H', include_history=True, on_test_set=True):
        """
        Predicts n periods in the future and stores them in self.forecast.

        Parameters
        ----------
        forecast_schedule: pandas dataframe
            Dataframe with 'ds' column containing dates (and times, optionally) and any additional regressors to forecast. If not None, will
            ignore periods, freq, history and test sets.
        periods: Integer
            Number of periods to predict
        freq: String
            Frequency definition. Any valid frequency for pd.date_range, such as 'D' or 'M'.
        include_history: Boolean to include the historical dates in the data
            frame for predictions.
        on_test_set: Bool
            If True, prediction will be performed on the <test_set> timestamps. Periods will be ignored.

        """
        if forecast_schedule is not None:
            new = forecast_schedule
        elif on_test_set:
            first_date = self.dset.test_set[self.input_ds_column].min()
            last_date = self.dset.test_set[self.input_ds_column].max()

            dates = pd.date_range(
                start=first_date,
                end=last_date,
                freq=freq)
            if include_history:
                dates = pd.concat([pd.Series(self.dset.train_set[self.input_ds_column]), pd.Series(dates)])
                dates = dates.reset_index(drop=True)
            new = pd.DataFrame({'ds': dates})
            if self.additional_regressors is not None:
                new = new.set_index(new.ds).join(
                    pd.concat(
                        [self.dset.train_set[self.additional_regressors],
                         self.dset.test_set[self.additional_regressors]]),
                    how='left').reset_index(drop=True)
        else:
            new = self.prophet.make_future_dataframe(
                periods=periods, freq=freq, include_history=include_history
            )
            if self.additional_regressors is not None:
                new = new.set_index(new.ds).join(
                    pd.concat(
                        [self.dset.train_set[self.additional_regressors], self.dset.test_set[self.additional_regressors]]),
                    how='left').reset_index(drop=True)
        self.forecast = self.prophet.predict(new)
        self.forecast.index.name = 'indx'
        # if include_history:
        #     self.combined = pd.merge(self.forecast, self.data, on='ds')
        return new


    @log_history
    def add_seasonality(
        self,
        name='monthly',
        period=12 * 30.5,
        fourier_order=5,
        prior_scale=None,
        mode=None,
        condition_name=None,
    ):
        self.prophet.add_seasonality(
            name=name,
            period=period,
            fourier_order=fourier_order,
            prior_scale=prior_scale,
            mode=mode,
            condition_name=condition_name,
        )

    @log_history
    def add_country_holidays(self, country_name='US'):
        self.prophet.add_country_holidays(country_name=country_name)

    @timing
    def sample_uncertainty(self, forecast_dates = None, nsamples=1000):
        if self.forecast is None:
            print("Can't sample uncertainty without fitting and predicting first." )
            return
        old_nsamples = self.prophet.uncertainty_samples
        self.prophet.uncertainty_samples = nsamples
        if forecast_dates is None:
            formatted_df = self.prophet.setup_dataframe(self.forecast[['ds']])
        else:
            formatted_df = self.prophet.setup_dataframe(forecast_dates)
        sampled_data = self.prophet.sample_posterior_predictive(formatted_df)['yhat']
        self.prophet.uncertainty_samples = old_nsamples
        return sampled_data

    def plot_predictions(self):
        """
        Plots predictions and fitted historical data.
        """
        if self.forecast is not None:
            self.prophet.plot(self.forecast)
        else:
            print('Need to predict first.')


    def plot_joint_plot(
        self,
        data='forecast',
        x='y',
        y='yhat',
        title=None,
        fpath='../figures',
        fname=None,
    ):
        """

        Parameters
        ----------
        data: String
            Name of Dataframe to plot
        x : string
            The variable on the y-axis
            Defaults to `y`, i.e. the observed values
        y : string
            The variable on the x-axis
            Defaults to `yhat`, i.e. the forecast or estimated values.
        title : string
            The title of the figure, default `None`.

        fpath : string
            The path to save the figures, default to `../figures`
        fname : string
            The filename for the figure to be saved
            ommits the extension, the figure is saved in png, jpeg and pdf

        """
        if getattr(self, data) is None:
            print('Join plot only works when doing in-sample prediction.')
        else:
            g = sns.jointplot(
                x=x, y=y, data=getattr(self, data), kind='reg', color='0.4', dropna=True
            )
            g.fig.set_figwidth(8)
            g.fig.set_figheight(8)
            ax = g.fig.axes[1]
            if title is not None:
                ax.set_title(title, fontsize=16)
            ax = g.fig.axes[0]
            annot_func = lambda a, b: (stats.pearsonr(a, b)[0] ** 2, MAE(a, b))

            g.annotate(
                annot_func,
                template='{stat}= {val:.2f}\nMAE={p:.2f}',
                stat='$R^2$',
                loc='upper left',
                fontsize=12,
            )

            ax.set_xlabel('True values', fontsize=15)
            ax.set_ylabel('Forecasted values', fontsize=15)
            ax.grid(ls=':')
            [l.set_fontsize(13) for l in ax.xaxis.get_ticklabels()]
            [l.set_fontsize(13) for l in ax.yaxis.get_ticklabels()]
            ax.grid(ls=':')

            if fname is not None:
                for ext in ['png', 'jpeg', 'pdf']:
                    g.fig.savefig(
                        os.path.join(fpath, '{}.{}'.format(fname, ext)), dpi=200
                    )

    def plot_residuals(
        self, data='forecast', x='y', y='yhat', fpath='../figures', fname=None
    ):
        """
        Plot fit residuals

        Parameters
        ----------
        data: String
            Name of Dataframe to plot
        x : string
            The variable on the y-axis
            Defaults to `y`, i.e. the observed values
        y : string
            The variable on the x-axis
            Defaults to `yhat`, i.e. the forecast or estimated values.
        fpath : string
            The path to save the figures, default to `../figures`
        fname : string
            The filename for the figure to be saved
            ommits the extension, the figure is saved in png, jpeg and pdf

        """
        if getattr(self, data) is None:
            print('Join plot only works when doing in-sample prediction.')
        else:
            tmp = getattr(self, data).copy().dropna()
            residuals = tmp[y] - tmp[x]
            f, ax = plt.subplots(figsize=(8, 8))
            g = sns.distplot(residuals, ax=ax, color='0.4')
            ax.grid(ls=':')
            ax.set_xlabel('Residuals', fontsize=15)
            ax.set_ylabel('Normalised frequency', fontsize=15)
            ax.grid(ls=':')

            [l.set_fontsize(13) for l in ax.xaxis.get_ticklabels()]
            [l.set_fontsize(13) for l in ax.yaxis.get_ticklabels()]

            ax.axvline(0, color='0.4')

            ax.set_title('Residuals distribution', fontsize=17)

            ax.text(
                0.05,
                0.85,
                'Skewness = {:4.3f}\nMedian = {:4.3f}\nMean = {:4.3f}'.format(
                    stats.skew(residuals), residuals.median(), residuals.mean()
                ),
                fontsize=12,
                transform=ax.transAxes,
            )

            if fname is not None:
                for ext in ['png', 'jpeg', 'pdf']:
                    g.fig.savefig(
                        os.path.join(fpath, '{}.{}'.format(fname, ext)), dpi=200
                    )

    def evaluate_train_test(self):
        """
        Evaluate model by splitting in train/test data. Fit train, forecast test, and compare.

        Parameters
        ----------
         test_size : float, int or None, optional (default=0.25)
            If float, should be between 0.0 and 1.0 and represent the proportion
            of the dataset to include in the test split. If int, represents the
            absolute number of test samples. If None, the value is set to the
            complement of the train size.
        verbose: Bool
            If True will print fitting details to console.
        on_test_set: Bool
            If True, evaluation will be performed on the <test_set> privided in <dset> and test_size will be ignored

        """


        try:
            self.fit()
        except Exception as e:
            if 'Prophet object can only be fit once' not in e.args[0]:
                print(e)

        self.predict(on_test_set=True, include_history=True)

        # train.reset_index(inplace=True)
        # test.reset_index(inplace=True)
        self.forecast.index = self.forecast.ds
        train_test_concat = pd.concat(
            [pd.Series(self.dset.train_set[self.input_y_column]), pd.Series(self.dset.test_set[self.input_y_column])])
        train_test_concat.index = train_test_concat.index.astype('datetime64[ns]')
        self.forecast = self.forecast.join(train_test_concat)
        self.forecast = self.forecast.rename(columns={self.input_y_column : 'y'})

        # Plot predictions on training and test data
        fig = plt.figure(facecolor='w', figsize=(10, 14))
        gs = fig.add_gridspec(nrows=3, ncols=2)
        # predictions
        ax1 = fig.add_subplot(gs[0, :])
        fcst_t = self.forecast['ds'].dt.to_pydatetime()
        train_ix = np.where(self.forecast['ds'] < self.dset.test_set[self.input_ds_column].iloc[0])[0]
        test_ix = np.where(self.forecast['ds'] >= self.dset.test_set[self.input_ds_column].iloc[0])[0]
        ax1.plot(fcst_t, self.forecast['y'], 'k.')
        ax1.plot(fcst_t[train_ix], self.forecast['yhat'][train_ix], ls='-', c='#0072B2')
        ax1.plot(fcst_t[test_ix], self.forecast['yhat'][test_ix], ls='-', c='darkorange')
        ax1.fill_between(
            fcst_t[train_ix],
            self.forecast['yhat_lower'][train_ix],
            self.forecast['yhat_upper'][train_ix],
            color='#0072B2',
            alpha=0.2,
        )
        ax1.fill_between(
            fcst_t[test_ix],
            self.forecast['yhat_lower'][test_ix],
            self.forecast['yhat_upper'][test_ix],
            color='darkorange',
            alpha=0.2,
        )
        [l.set_fontsize(8) for l in ax1.xaxis.get_ticklabels()]
        [l.set_fontsize(8) for l in ax1.yaxis.get_ticklabels()]

        # plot correlation -- need to get rid off nans.
        self.forecast = self.forecast.dropna()
        train_ix = np.where(self.forecast['ds'] < self.dset.test_set[self.input_ds_column].iloc[0])[0]
        test_ix = np.where(self.forecast['ds'] >= self.dset.test_set[self.input_ds_column].iloc[0])[0]

        # train
        ax2 = fig.add_subplot(gs[1, 0])
        if len(train_ix)>0:
            rsquare = (
                stats.pearsonr(
                    self.forecast.iloc[train_ix]['y'], self.forecast.iloc[train_ix]['yhat']
                )[0]
                ** 2
            )


            mae = MAE(self.forecast.iloc[train_ix]['y'], self.forecast.iloc[train_ix]['yhat'])
            self.forecast.iloc[train_ix].plot(
                x='y', y='yhat', kind='scatter', c='black', s=5, ax=ax2
            )
            ax2.set_title('Train: R^2=%.2f, MAE=%.2f' % (rsquare, mae), fontsize=11)
            ax2.set_xlabel('True values', fontsize=10)
            ax2.set_ylabel('Forecasted values', fontsize=10)
            ax2.grid(ls=':')
            [l.set_fontsize(10) for l in ax2.xaxis.get_ticklabels()]
            [l.set_fontsize(10) for l in ax2.yaxis.get_ticklabels()]
            ax2.grid(ls=':')

        # test
        ax3 = fig.add_subplot(gs[1, 1])
        if len(test_ix)>0:
            rsquare = (
                stats.pearsonr(
                    self.forecast.iloc[test_ix]['y'], self.forecast.iloc[test_ix]['yhat']
                )[0]
                ** 2
            )
            mae = MAE(self.forecast.iloc[test_ix]['y'], self.forecast.iloc[test_ix]['yhat'])
            self.forecast.iloc[test_ix].plot(
                x='y', y='yhat', kind='scatter', c='black', s=5, ax=ax3
            )
            ax3.set_title('Test: R^2=%.2f, MAE=%.2f' % (rsquare, mae), fontsize=11)
            ax3.set_xlabel('True values', fontsize=10)
            ax3.set_ylabel('Forecasted values', fontsize=10)
            ax3.grid(ls=':')
            [l.set_fontsize(10) for l in ax3.xaxis.get_ticklabels()]
            [l.set_fontsize(10) for l in ax3.yaxis.get_ticklabels()]
            ax3.grid(ls=':')

        # training set residuals
        train_residuals = self.forecast['yhat'][train_ix] - self.forecast['y'][train_ix]
        train_residuals = train_residuals.dropna()
        ax4 = fig.add_subplot(gs[2, 0])
        g = sns.distplot(train_residuals, ax=ax4, color='0.4')
        ax4.grid(ls=':')
        ax4.set_xlabel('Residuals', fontsize=10)
        ax4.set_ylabel('Normalised frequency', fontsize=10)
        ax4.grid(ls=':')
        [l.set_fontsize(10) for l in ax4.xaxis.get_ticklabels()]
        [l.set_fontsize(10) for l in ax4.yaxis.get_ticklabels()]
        ax4.axvline(0, color='0.4')
        ax4.set_title('Residuals distribution (train)', fontsize=11)
        ax4.text(
            0.05,
            0.85,
            'Skewness = {:4.3f}\nMedian = {:4.3f}\nMean = {:4.3f}'.format(
                stats.skew(train_residuals),
                train_residuals.median(),
                train_residuals.mean(),
            ),
            fontsize=10,
            transform=ax4.transAxes,
        )
        # test set residuals
        test_residuals = self.forecast['yhat'][test_ix] - self.forecast['y'][test_ix]
        test_residuals = test_residuals.dropna()
        ax5 = fig.add_subplot(gs[2, 1])
        g = sns.distplot(test_residuals, ax=ax5, color='0.4')
        ax5.grid(ls=':')
        ax5.set_xlabel('Residuals', fontsize=10)
        ax5.set_ylabel('Normalised frequency', fontsize=10)
        ax5.grid(ls=':')
        [l.set_fontsize(10) for l in ax5.xaxis.get_ticklabels()]
        [l.set_fontsize(10) for l in ax5.yaxis.get_ticklabels()]
        ax5.axvline(0, color='0.4')
        ax5.set_title('Residuals distribution (test)', fontsize=11)
        ax5.text(
            0.05,
            0.85,
            'Skewness = {:4.3f}\nMedian = {:4.3f}\nMean = {:4.3f}'.format(
                stats.skew(test_residuals),
                test_residuals.median(),
                test_residuals.mean(),
            ),
            fontsize=10,
            transform=ax5.transAxes,
        )
        plt.tight_layout()
        plt.show()



if __name__ == '__main__':
    pass