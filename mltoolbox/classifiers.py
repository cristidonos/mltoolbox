import sys
from time import sleep, time

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd

import mltoolbox
import os
from mltoolbox.datasets import dset
from mltoolbox.misc.generics import timing
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.metrics import accuracy_score, mean_squared_error
from skopt import BayesSearchCV
from xgboost import XGBClassifier

# package_path = os.path.dirname(mltoolbox.__file__).replace(os.path.sep + 'lib' + os.path.sep,
#                                                            os.path.sep + 'Lib' + os.path.sep)
# sys.path.insert(0, package_path + os.path.join(os.path.sep, 'external', 'parametric_tsne',os.path.sep))
# import parametric_tSNE as ptSNE

from mltoolbox.external.parametric_tsne import parametric_tSNE as ptSNE
from mltoolbox.external.parametric_tsne.parametric_tSNE.utils import get_multiscale_perplexities

seed = 0

candidates = {
    'xgb': {
        'func': XGBClassifier(),
        'params':
            {
                'random_state': (seed,),
                'learning_rate': (1e-4, 1, 'log-uniform'),
                'n_estimators': (50, 7000),
                'max_depth': (2, 6),
                'reg_alpha': (1e-4, 0.5, 'log-uniform'),
                'reg_lambda': (1e-4, 0.5, 'log-uniform'),
                'booster': ['gbtree', 'gblinear']
            },

    },
    'gpc': {
        'func': GaussianProcessClassifier(),
        'params':
            {
                'random_state': (seed,),
                'max_iter_predict': (50, 1000),
                'multi_class': ['one_vs_rest', 'one_vs_one'],
            },

    },
}


class Hyperoptimizer():
    class BayesSearchCV(BayesSearchCV):
        def _run_search(self, x): raise BaseException('Use newer skopt')

    def optimize(self, model, params,
                 x_train=None, y_train=None, x_test=None, y_test=None, type=None,
                 n_jobs=-1, n_iter=15, verbose=0, seed=None):
        opt = BayesSearchCV(
            model,
            params,
            n_iter=n_iter,
            random_state=seed,
            n_jobs=n_jobs,
            verbose=verbose,
        )
        opt.fit(x_train, np.ravel(y_train))
        if type == 'classification':
            score = accuracy_score(y_test, opt.best_estimator_.predict(x_test))
        elif type == 'regression':
            score = np.sqrt(mean_squared_error(y_test, opt.best_estimator_.predict(x_test)))
        else:
            print('Hyperoptimizer support only classification and regression. Make sure "type" is properly specified.')
        return opt.best_estimator_, opt.best_params_, score

    def __init__(self, candidates=None, type=None, x_train=None, y_train=None, x_test=None, y_test=None, n_jobs=-1,
                 n_iter=15, verbose=0, seed=None):
        self.candidates = candidates
        self.type = type
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.n_jobs = n_jobs
        self.n_iter = n_iter
        self.verbose = verbose
        self.seed = seed
        self.results = {}
        self.results_table = None
        print('Starting hyperoptimization of %d algorithms.' % len(self.candidates.keys()))

    @timing
    def run(self):
        for c in self.candidates.keys():
            c_start = time()
            print('Model: Training ' + c + '...', end='')
            sleep(.1)  # make sure
            func = self.candidates[c]['func']
            params = self.candidates[c]['params']
            n_jobs = self.n_jobs
            if 'keras' in c:
                n_jobs = 1  # keras is un-serializable
            self.results[c] = {}
            self.results[c]['model'], self.results[c]['params'], self.results[c]['score'] = self.optimize(func, params,
                                                                                                          x_train=self.x_train,
                                                                                                          y_train=self.y_train,
                                                                                                          x_test=self.x_test,
                                                                                                          y_test=self.y_test,
                                                                                                          n_jobs=n_jobs,
                                                                                                          n_iter=self.n_iter,
                                                                                                          verbose=self.verbose,
                                                                                                          type=self.type,
                                                                                                          seed=self.seed)
            self.results[c]['runtime_seconds'] = time() - c_start
            print(' took %.3f seconds.' % (self.results[c]['runtime_seconds']))
            sleep(.1)  # make sure
            print('  * Test score: [%.4f]' % self.results[c]['score'])

        self.results_table = pd.DataFrame.from_dict(self.results).T
        self.results_table.sort_values(by=['score', 'runtime_seconds'], ascending=[False, True], inplace=True)

        return self

    def summary(self):
        if self.results_table.keys() is None:
            print('You need to call "run()" first.')
        else:
            print(self.results_table[['score', 'runtime_seconds']])
        return self

    def get_best_algorithm(self):
        return self.results_table.iloc[0].to_dict()


class Ptsne():
    def __init__(self, dset, n_components=2, perplexities=None, alpha=None, kwargs={}):
        self.data = dset
        self.n_components = n_components
        self.perplexities = perplexities
        self.alpha = alpha
        self.color_palette = sns.color_palette("hls", len(self.data.variables))
        self.fitted_train = None
        self.fitted_test = None
        self.labels_train = self.data.train_set[self.data.target]
        if self.labels_train.shape[1] > 1:
            # convert from one-hot vectors to labels
            self.labels_train = self.labels_train.dot(self.labels_train.columns)
        self.labels_test = self.data.test_set[self.data.target]
        if self.labels_test.shape[1] > 1:
            # convert from one-hot vectors to labels
            self.labels_test = self.labels_test.dot(self.labels_test.columns)

        if self.alpha is None:
            self.alpha = self.n_components - 1.0
        if perplexities is None:
            self.perplexities = get_multiscale_perplexities(2 * self.data.train_set.shape[0])
            print('Using multiple perplexities: %s' % (','.join(map(str, self.perplexities))))

        self.ptsne = ptSNE.Parametric_tSNE(num_inputs=len(self.data.variables), num_outputs=self.n_components,
                                           perplexities=self.perplexities, alpha=self.alpha, **kwargs)

    @timing
    def fit_ptsne(self, epochs=20, kwargs={}):
        print('Fitting Parametric t-SNE model.')
        columns = [c for c in self.data.data.columns if c not in self.data.target]
        self.ptsne.fit(training_data=self.data.train_set[columns].values, epochs=epochs, **kwargs)
        return self

    @timing
    def transform_ptsne(self, dataset=['train', 'test']):
        columns = [c for c in self.data.data.columns if c not in self.data.target]
        for d in dataset:
            print('Transforming %s.' % d)
            tmp = getattr(self.data, d + '_set')
            setattr(self, 'fitted_' + d, self.ptsne.transform(test_data=tmp[columns].values))
        return self

    def plot2d(self, alpha=0.5, train_symbol='.', test_symbol='s'):
        # train
        color_ix = 0
        for ci in np.unique(self.labels_train):
            cur_plot_rows = np.where(self.labels_train == ci)[0]
            cur_color = self.color_palette[color_ix]
            cur_cmap = sns.light_palette(cur_color, as_cmap=True)
            try:
                sns.kdeplot(self.fitted_train[cur_plot_rows, 0].ravel(), self.fitted_train[cur_plot_rows, 1].ravel(),
                            cmap=cur_cmap,
                            shade=True,
                            alpha=alpha, shade_lowest=False)
            except:
                print('PTSNE.PLOT2D: can not plot kde for %s.' % str(ci))
            centroid = np.median(self.fitted_train[cur_plot_rows], axis=0)
            plt.scatter(*centroid, c='black', marker='x')
            plt.annotate('%s' % ci, xy=centroid, xycoords='data', alpha=0.5,
                         horizontalalignment='right', verticalalignment='center')
            plt.plot(self.fitted_train[cur_plot_rows, 0], self.fitted_train[cur_plot_rows, 1], train_symbol,
                     color=cur_color, alpha=alpha)
            color_ix = color_ix + 1
        # test
        color_ix = 0
        for ci in set(self.labels_test):
            cur_plot_rows = np.where(self.labels_test == ci)[0]
            cur_color = self.color_palette[color_ix]
            plt.plot(self.fitted_test[cur_plot_rows, 0].ravel(), self.fitted_test[cur_plot_rows, 1].ravel(),
                     test_symbol,
                     color=cur_color, label=ci, alpha=alpha)
            color_ix = color_ix + 1


if __name__ == "__main__":
    # file = r'C:\Users\i503207\Documents\RompetrolCloudPoint\CloudPointTrain.xlsx'

    # ds = dset(file, target=["target"], id="id", header=0,
    #           variables=['feat1', 'feat13', 'feat15', 'feat17']
    #           )

    # data = sns.load_dataset('iris')
    # ds = dset(data, target=["species"], header=0, target_type=''
    #           # variables=['feat1', 'feat13', 'feat15', 'feat17']
    #           )
    #
    # (ds
    #  .remove_nan_rows(columns=[ds.target], any_nan=True)
    #  .remove_nan_columns(any_nan=True)
    #  .categorical2numeric(exclude_target_column=False)
    #  .split_train_test_sets(train_test_split=0.2, shuffle_split=True, seed=0)
    #  .normalize(type='unitnorm')
    #  .summary()
    #  )

    from sklearn.datasets import load_breast_cancer
    data = load_breast_cancer()
    data = pd.concat([pd.DataFrame(data.data, columns=data.feature_names), pd.DataFrame(data.target, columns=['cancer'])],axis=1)

    ds = dset(data, target=["cancer"], header=0, target_type=''
              # variables=['feat1', 'feat13', 'feat15', 'feat17']
              )

    (ds
     .remove_nan_rows(columns=[ds.target], any_nan=True)
     .remove_nan_columns(any_nan=True)
     .categorical2numeric(exclude_target_column=False)
     .split_train_test_sets(train_test_split=0.2, shuffle_split=True, seed=0)
     .normalize(type='gaussian')
     .summary()
     )

    ptsne = Ptsne(ds, perplexities=None, n_components=2, kwargs={'batch_size' :64})
    ptsne.fit_ptsne(epochs=25000)  # , kwargs={'verbose' : True})
    ptsne.transform_ptsne()
    # ptsne.plot2d()

    hopt = (Hyperoptimizer(candidates=candidates, type='classification',
                           x_train=ptsne.fitted_train,
                           y_train=ptsne.labels_train,
                           x_test=ptsne.fitted_test,
                           y_test=ptsne.labels_test,
                           n_iter=20)
            .run()
            .summary()
            )
    hopt.get_best_algorithm()

