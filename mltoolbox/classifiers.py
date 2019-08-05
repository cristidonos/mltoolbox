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
from mltoolbox.misc import plots
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.svm import SVC
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
import sklearn.manifold
from sklearn.metrics import accuracy_score, mean_squared_error
from skopt import BayesSearchCV
from xgboost import XGBClassifier
from sklearn import preprocessing

from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout
from keras import regularizers
from keras.callbacks import EarlyStopping, ModelCheckpoint
import keras.backend as K

from matplotlib.colors import ListedColormap

# package_path = os.path.dirname(mltoolbox.__file__).replace(os.path.sep + 'lib' + os.path.sep,
#                                                            os.path.sep + 'Lib' + os.path.sep)
# sys.path.insert(0, package_path + os.path.join(os.path.sep, 'external', 'parametric_tsne',os.path.sep))
# import parametric_tSNE as ptSNE

from mltoolbox.external.parametric_tsne import parametric_tSNE as ptSNE
from mltoolbox.external.parametric_tsne.parametric_tSNE.utils import get_multiscale_perplexities

seed = 0
from numpy.random import seed as npseed

npseed(seed)
from tensorflow import set_random_seed

set_random_seed(seed)

candidates = {
    'gpc': {
        'func': GaussianProcessClassifier(),
        'params':
            {
                'random_state': (seed,),
                'max_iter_predict': (50, 1000),
                'multi_class': ['one_vs_rest', 'one_vs_one'],
            },

    },
    'knn': {
        'func': KNeighborsClassifier(),
        'params':
            {
                'n_neighbors': (3, 25),
            },
    },
    'rf': {
        'func': RandomForestClassifier(),
        'params':
            {
                'random_state': (seed,),
                'n_estimators': (10, 2000),
                'max_depth': (2, 6),
                'min_samples_leaf': (1, 10),
            },

    },
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
    'mlp': {
        'func': MLPClassifier(),
        'params':
            {
                'random_state': (seed,),
                'learning_rate_init': (1e-4, 1, 'log-uniform'),
                'alpha': (1e-4, 0.5, 'log-uniform'),
                'max_iter': (200, 10000),
            }
    }
}


class Hyperoptimizer():
    class BayesSearchCV(BayesSearchCV):
        def _run_search(self, x): raise BaseException('Use newer skopt')

    @timing
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

    def get_best_algorithm(self, sort='desc'):
        if sort == 'desc':
            return self.results_table.iloc[0].to_dict()
        if sort == 'asc':
            return self.results_table.iloc[-1].to_dict()

    @timing
    def plot_comparison(self, mesh_resolution=None):

        x_min, x_max = self.x_train[:, 0].min(), self.x_train[:, 0].max()
        y_min, y_max = self.x_train[:, 1].min(), self.x_train[:, 1].max()
        if mesh_resolution is None:
            mesh_resolution = min((x_max - x_min) / 100, (y_max - y_min) / 100)  # step size in the mesh
        xx, yy = np.meshgrid(np.arange(x_min, x_max, mesh_resolution),
                             np.arange(y_min, y_max, mesh_resolution))

        le = preprocessing.LabelEncoder()
        le.fit(self.y_train.values)
        numeric_labels_train = le.transform(self.y_train.values)
        numeric_labels_test = le.transform(self.y_test.values)

        figure = plt.figure(figsize=(27, 9))
        i = 1
        # just plot the dataset first
        cm = plots.cmap_discretize('jet', len(np.unique(self.y_train.values)))
        # cm_bright = plots.cmap_discretize('jet', len(np.unique(self.y_train.values)) + 1).set_gamma(0.5)
        ax = plt.subplot(1, len(list(self.results.keys())) + 1, i)

        # Plot the training points
        ax.scatter(self.x_train[:, 0], self.x_train[:, 1], c=numeric_labels_train, cmap=cm,
                   edgecolors='k')
        # Plot the testing points
        ax.scatter(self.x_test[:, 0], self.x_test[:, 1], marker='s', c=numeric_labels_test, cmap=cm,
                   edgecolors='k')
        ax.set_xlim(xx.min(), xx.max())
        ax.set_ylim(yy.min(), yy.max())
        ax.set_xticks(())
        ax.set_yticks(())
        ax.set_title("Input data")
        i += 1
        cont = {}
        # iterate over classifiers
        for name in list(self.results.keys()):
            ax = plt.subplot(1, len(list(self.results.keys())) + 1, i)
            self.results[name]['model'].fit(self.x_train, self.y_train)
            score = self.results[name]['model'].score(self.x_test, self.y_test)

            # Plot the decision boundary. For that, we will assign a color to each
            # point in the mesh [x_min, x_max]x[y_min, y_max].
            try:
                if hasattr(self.results[name]['model'], "decision_function"):
                    Z = self.results[name]['model'].decision_function(np.c_[xx.ravel(), yy.ravel()])
                else:
                    # Z = self.results[name]['model'].predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
                    Z = np.argmax(self.results[name]['model'].predict_proba(np.c_[xx.ravel(), yy.ravel()]), axis=1)

                # Put the result into a color plot
                Z = Z.reshape(xx.shape)
                ax.contourf(xx, yy, Z, levels=len(np.unique(numeric_labels_train)), cmap=cm, alpha=0.3)
                cont[name] = Z
            except Exception as e:
                print(e.args[0])

            # Plot the training points
            ax.scatter(self.x_train[:, 0], self.x_train[:, 1], c=numeric_labels_train, cmap=cm,
                       edgecolors='k')
            # Plot the testing points
            ax.scatter(self.x_test[:, 0], self.x_test[:, 1], marker='s', c=numeric_labels_test, cmap=cm,
                       edgecolors='k')

            ax.set_xlim(xx.min(), xx.max())
            ax.set_ylim(yy.min(), yy.max())
            ax.set_xticks(())
            ax.set_yticks(())
            ax.set_title(name)
            ax.text(xx.max() - .3, yy.min() + .3, ('%.3f' % score).lstrip('0'),
                    size=15, horizontalalignment='right')
            i += 1

        plt.tight_layout()
        plt.show()
        return self


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
        # columns = [c for c in self.data.data.columns if c not in self.data.target]
        columns = [c for c in self.data.variables if c not in self.data.target]

        self.ptsne.fit(training_data=self.data.train_set[columns].values, epochs=epochs, **kwargs)
        return self

    @timing
    def transform_ptsne(self, dataset=['train', 'test']):
        # columns = [c for c in self.data.data.columns if c not in self.data.target]
        columns = [c for c in self.data.variables if c not in self.data.target]
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


class TsneClassifier(BaseEstimator, ClassifierMixin):

    def __init__(self, input_dset, n_components=2, perplexity=30, early_exaggeration=12.0, learning_rate=200.0,
                 n_iter=1000,
                 random_state=1):
        self.data = input_dset
        self.n_components = n_components
        self.perplexity = perplexity
        self.early_exaggeration = early_exaggeration
        self.learning_rate = learning_rate
        self.n_iter = n_iter
        self.random_state = random_state

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

    def tsne_fit(self, kwargs={}):
        """
        Fit t-SNE to training dataset
        Parameters
        ----------
        kwargs

        Returns
        -------

        """
        self.tsne_model = sklearn.manifold.TSNE(n_components=self.n_components, perplexity=self.perplexity,
                                                early_exaggeration=self.early_exaggeration,
                                                learning_rate=self.learning_rate,
                                                n_iter=self.n_iter,
                                                random_state=self.random_state, **kwargs)
        self.tsne = {'train_set': self.tsne_model.fit_transform(self.data.train_set)}
        return self

    def plot_tsne_fit(self, color_by=None):
        """
        Visualize t-SNE on training dataset
        Parameters
        ----------
        color_by

        Returns
        -------

        """
        fig = plt.figure()
        ax = fig.gca()
        if color_by is None:
            color_by = self.data.target
        le = preprocessing.LabelEncoder()
        le.fit(self.labels_train.values)
        numeric_labels_train = le.transform(self.labels_train.values)
        # numeric_labels_test = le.transform(self.labels_test.values)
        cmap = plots.cmap_discretize('jet', len(np.unique(self.labels_train.values)) + 1)
        if self.tsne['train_set'].shape[1] == 2:
            mappable = ax.scatter(self.tsne['train_set'][:, 0], self.tsne['train_set'][:, 1], c=numeric_labels_train,
                                  cmap=cmap)
        if self.tsne['train_set'].shape[1] == 3:
            ax = fig.gca(projection='3d')
            mappable = ax.scatter(self.tsne['train_set'][:, 0], self.tsne['train_set'][:, 1],
                                  self.tsne['train_set'][:, 2],
                                  c=numeric_labels_train,
                                  cmap=cmap)
        cb = plots.colorbar_index(len(np.unique(self.labels_train.values)), cmap)

        plt.title('TSNE training set')
        plt.tight_layout()

    def euclidean_distance_loss(self, y_true, y_pred):
        """
        Euclidean distance loss
        https://en.wikipedia.org/wiki/Euclidean_distance
        :param y_true: TensorFlow/Theano tensor
        :param y_pred: TensorFlow/Theano tensor of the same shape as y_true
        :return: float
        """
        return K.sqrt(K.sum(K.square(y_pred - y_true), axis=-1))

    def init_keras_model(self, hidden_neurons, output_neurons, dropout=None, regularizer=None, activation='relu',
                         loss='mean_squared_error', metrics=['mse']):
        model = Sequential()
        for hn in hidden_neurons:
            model.add(Dense(hn, activation=activation))
            if regularizer is not None:
                regularizers.l1_l2(l1=regularizer, l2=regularizer)
            if dropout is not None:
                model.add(Dropout(dropout))
        model.add(Dense(output_neurons, activation='linear'))
        model.compile(loss=loss, optimizer='adam', metrics=metrics)
        return model

    @timing
    def fit(self, X=None, y=None, hidden_neurons=[500, 250, 150], output_neurons=2, dropout=None, epochs=1000,
            batch_size=512, validation_split=None, regularizer=None, activation='relu', verbose=1,
            loss='mean_sqared_error', metrics=['mse']):
        self.keras_model = self.init_keras_model(hidden_neurons=hidden_neurons, output_neurons=output_neurons,
                                                 dropout=dropout, regularizer=regularizer, activation=activation,loss=loss, metrics=metrics)
        if X is None:
            X = self.data.train_set[self.data.variables].values
        if y is None:
            y = self.tsne['train_set']
        es = EarlyStopping(monitor='val_loss', mode='min', patience=200, verbose=verbose)
        mc = ModelCheckpoint('best_tsne_model.h5', monitor='val_loss', mode='min', verbose=verbose,
                             save_best_only=True)
        self.keras_model_history = self.keras_model.fit(x=X, y=y, epochs=epochs, batch_size=batch_size,
                                                        validation_split=validation_split, shuffle=False,
                                                        callbacks=[es, mc], verbose=verbose)
        self.keras_model = load_model('best_tsne_model.h5', custom_objects={'euclidean_distance_loss': self.euclidean_distance_loss})
        os.remove('best_tsne_model.h5')
        return self

    def predict(self, X):
        return self.keras_model.predict(x=X)

    def transform2tsne(self):
        self.fitted_train = self.predict(self.data.train_set[self.data.variables].values)
        self.fitted_test = self.predict(self.data.test_set[self.data.variables].values)


if __name__ == "__main__":
    # file = r'C:\Users\i503207\Documents\RompetrolCloudPoint\CloudPointTrain.xlsx'

    # ds = dset(file, target=["target"], id="id", header=0,
    #           variables=['feat1', 'feat13', 'feat15', 'feat17']
    #           )
    #
    # data = sns.load_dataset('iris')
    # ds = dset(data, target=["species"], header=0, target_type=''
    #           # variables=['feat1', 'feat13', 'feat15', 'feat17']
    #           )

    from sklearn.datasets import make_classification

    X1, Y1 = make_classification(n_samples=1000, n_features=10, n_redundant=4, n_informative=6,
                                 n_clusters_per_class=1, n_classes=4)
    data = pd.DataFrame(X1, columns=['feat' + str(i) for i in range(X1.shape[1])])
    data_labels = pd.DataFrame(Y1, columns=['target'])
    data = data.join(data_labels)

    ds = dset(data, target=["target"], header=0, target_type='')

    (ds
     .remove_nan_rows(columns=[ds.target], any_nan=True)
     .remove_nan_columns(any_nan=True)
     .categorical2numeric(exclude_target_column=False)
     .split_train_test_sets(train_test_split=0.7, shuffle_split=True, seed=0)
     .normalize(type='unitnorm')
     .summary()
     )

    # from sklearn.datasets import load_breast_cancer
    # data = load_breast_cancer()
    # data = pd.concat([pd.DataFrame(data.data, columns=data.feature_names), pd.DataFrame(data.target, columns=['cancer'])],axis=1)
    #
    # ds = dset(data, target=["cancer"], header=0, target_type=''
    #           # variables=['feat1', 'feat13', 'feat15', 'feat17']
    #           )
    #
    # (ds
    #  .remove_nan_rows(columns=[ds.target], any_nan=True)
    #  .remove_nan_columns(any_nan=True)
    #  .categorical2numeric(exclude_target_column=False)
    #  .split_train_test_sets(train_test_split=0.2, shuffle_split=True, seed=0)
    #  .normalize(type='gaussian')
    #  .summary()
    #  )

    # ptsne = Ptsne(ds, perplexities=None, n_components=2, kwargs={'batch_size' :64})
    # ptsne.fit_ptsne(epochs=25000)  # , kwargs={'verbose' : True})
    # ptsne.transform_ptsne()
    # # ptsne.plot2d()
    #
    # hopt = (Hyperoptimizer(candidates=candidates, type='classification',
    #                        x_train=ptsne.fitted_train,
    #                        y_train=ptsne.labels_train,
    #                        x_test=ptsne.fitted_test,
    #                        y_test=ptsne.labels_test,
    #                        n_iter=20)
    #         .run()
    #         .summary()
    #         )
    # hopt.get_best_algorithm()

    # variables = [c for c in ds.data.columns if c not in ds.target]
    # hopt = (Hyperoptimizer(candidates=candidates, type='classification',
    #                        x_train=ds.train_set[variables],
    #                        y_train=ds.train_set[ds.target],
    #                        x_test=ds.test_set[variables],
    #                        y_test=ds.test_set[ds.target],
    #                        n_iter=20)
    #         .run()
    #         .summary()
    #         )
    # hopt.get_best_algorithm()

    t = TsneClassifier(input_dset=ds, n_components=2, perplexity=10, early_exaggeration=12, n_iter=10000,
                       random_state=seed)
    t.tsne_fit()
    t.plot_tsne_fit()
    t.fit(epochs=5000)  # , hidden_neurons=[50,100, 150], output_neurons=2, batch_size=1000, dropout=0.25)
    t.transform2tsne()

    le = preprocessing.LabelEncoder()
    train_labels = le.fit_transform(t.labels_train.values)
    test_labels = le.fit_transform(t.labels_test.values)
    plt.figure()
    plt.scatter(t.tsne['train_set'][:, 0], t.tsne['train_set'][:, 1], c=train_labels)
    plt.scatter(t.fitted_train[:, 0], t.fitted_train[:, 1], marker='s', c=train_labels)
    plt.scatter(t.fitted_test[:, 0], t.fitted_test[:, 1], marker='d', c=test_labels, edgecolors='k')

    hopt = (Hyperoptimizer(candidates=candidates, type='classification',
                           x_train=t.fitted_train,
                           y_train=t.labels_train,
                           x_test=t.fitted_test,
                           y_test=t.labels_test,
                           n_iter=20, seed=seed)
            .run()
            .summary()
            )
    hopt.get_best_algorithm()
    cont = hopt.plot_comparison(mesh_resolution=None)
