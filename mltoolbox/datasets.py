import os
import sys
from functools import wraps

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn.manifold
import sklearn.model_selection
import sklearn.preprocessing
import sklearn.decomposition
from mltoolbox.misc import plots

if not sys.warnoptions:
    import warnings

    warnings.simplefilter("ignore")


class dset:
    def log_history(f):
        """
        Decorator for saving a history of called methods during the class instance lifespan
        """

        @wraps(f)
        def wrapper(instance, *args, **kwargs):
            instance.history.append([f.__qualname__.split('.')[1], args, kwargs])
            return f(instance, *args, **kwargs)

        return wrapper

    def __init__(
        self,
        data,
        id=None,
        target=None,
        variables=None,
        target_type="continuous",
        skiprows=None,
        header=True,
    ):
        """

        Parameters
        ----------
        data
        id
        target
        variables
        target_type
        skiprows
        header
        """
        self.id = id
        # self.target = target
        try:
            target[0]
            self.target = target[0]
        except:
            self.target = target
        self.skiprows = skiprows
        self.header = header
        self.target_type = target_type
        self.history = []
        self.train_set = None
        self.test_set = None
        self.scaler = None
        self.scaler_columns = None
        self.pca_scaler=None
        self.pca_columns=None

        if isinstance(data, pd.DataFrame):
            self.data = data
        else:
            # data is filepath
            loaded = False
            filename, file_extension = os.path.splitext(data)
            if 'csv' in file_extension:
                data = pd.read_csv(data, skiprows=self.skiprows, header=self.header, index_col=self.id)
                loaded = True
            if "xls" in file_extension:
                data = pd.read_excel(
                    data, skiprows=self.skiprows, header=self.header, index_col=self.id
                )
                loaded = True
            if not loaded:
                raise RuntimeError(
                    "\n Data argument is not a Pandas dataframe of path to xls/xlsx/csv file."
                )
            self.data = data

        if variables is None:
            self.variables = [c for c in self.data.columns if c not in [self.target, self.id]]
        else:
            tmp = [x for x in [self.target] if x is not None]
            tmp.extend(variables)
            self.variables = variables
            self.data = self.data[tmp]

        # self.categorical_variables = [v for v in self.variables if pd.api.types.is_string_dtype(self.data[v])]
        # self.numeric_variables = [v for v in self.variables if pd.api.types.is_numeric_dtype(self.data[v])]

    @log_history
    def split_train_test_sets(self, train_test_split=0.2, shuffle_split=False, stratify= None, seed=None):
        """
        Split data in train-test sets.
        Parameters
        ----------
        train_test_split : float, int or None, optional (default = 0.2)
            If float, should be between 0.0 and 1.0 and represent the proportion
            of the dataset to include in the test split. If int, represents the
            absolute number of test samples. If None, the value is set to the
            complement of the train size.
        shuffle_split : bool
            If True, data samples will be shuffled before train-test split.
        stratify: array or None. If array will be provided, will attempt to preserve the class label ratios in the
            train-test split.
        seed: int or None
            Random state

        Returns
        -------

        """
        self.train_set, self.test_set = sklearn.model_selection.train_test_split(self.data, test_size=train_test_split,
                                                                                 shuffle=shuffle_split, stratify=stratify,
                                                                                 random_state=seed)
        self.labels_train = self.train_set[self.target]
        try:
            # if dataframe has on column that it's shape is (x,), which causes tuple index error
            if self.labels_train.shape[1] > 1:
                # convert from one-hot vectors to labels
                self.labels_train = self.labels_train.dot(self.labels_train.columns)
        except:
            pass

        self.labels_test = self.test_set[self.target]
        try:
            if self.labels_test.shape[1] > 1:
                # convert from one-hot vectors to labels
                self.labels_test = self.labels_test.dot(self.labels_test.columns)
        except:
            pass
        return self

    @log_history
    def deblank_column_names(self):
        """
        Replace spaces with underscores in column names
        """
        columns = [c.replace(" ", "_") for c in self.data.columns]
        self.data.columns = columns
        if self.id:
            self.id = self.id.replace(" ", "_")
        if self.target:
            self.target = self.target.replace(" ", "_")
        # if self.variables:
        #     self.variables = self.variables.replace(" ", "_")
        return self

    @log_history
    def remove_nan_columns(self, columns=None, any_nan=False):
        """
        Remove empty columns (all nans)
        Parameters
        ----------
        columns: list
            Column names in which to look for nans
        any_nan: bool
            If true columns will be discarded if any nan is found.
        """
        if any_nan:
            self.data = self.data.dropna(axis=1, how="any", subset=columns)
        else:
            self.data = self.data.dropna(axis=1, how="all", subset=columns)
        return self

    @log_history
    def remove_nan_rows(self, columns=None, any_nan=False):
        """
        Remove rows with nans in columns.

        Parameters
        ----------
        columns: list
            Column names in which to look for nans
        any_nan: bool
            If true rows will be discarded if any nan is found. If false, all columns are required to have nans.
        """
        if any_nan:
            self.data = self.data.dropna(axis=0, how="any", subset=columns)
        else:
            self.data = self.data.dropna(axis=0, how="all", subset=columns)
        return self

    @log_history
    def drop_column(self, columns):
        """
        Drop Columns

        Parameters
        ----------
        columns: list
            List of columns to drop
        """
        self.data = self.data.drop(axis=1, labels=columns)
        return self

    @log_history
    def reindex(self):
        """
        Reindex dataframe
        """
        self.data = self.data.reset_index(drop=True)
        return self

    @log_history
    def normalize(self, columns=None, type='zscore', inplace=True, kwargs={}):
        if type == 'zscore':
            self.scaler = sklearn.preprocessing.StandardScaler(copy=(not inplace), **kwargs)
        elif type == 'robust':
            self.scaler = sklearn.preprocessing.RobustScaler(copy=(not inplace), **kwargs)
        elif type == 'uniform':
            self.scaler = sklearn.preprocessing.QuantileTransformer(copy=(not inplace), **kwargs)
        elif type == 'gaussian':
            self.scaler = sklearn.preprocessing.PowerTransformer(copy=(not inplace), **kwargs)
        elif type == 'unitnorm':
            self.scaler = sklearn.preprocessing.Normalizer(copy=(not inplace), **kwargs)

        if columns is None:
            # print('\nColumns for normalization not explicitely specified. Will use all columns defined as variables.')
            # columns = self.variables
            print('\nColumns for normalization not explicitely specified. Will use all columns defined as variables.')
            columns = self.variables #[c for c in self.data.columns if c not in self.target]


        self.scaler_columns = columns

        if self.train_set is None:
            print('\nNo training dataset is defined: applying normalization to the whole dataset. '
                  '\nYou should not split in train / test datasets after this step.')
            self.data.loc[:, self.scaler_columns] = self.scaler.fit_transform(self.data[self.scaler_columns])
        else:
            print(
                '\nApplying normalization fitting a scaler to the training dataset and using the same transformation on the test set.')
            self.train_set.loc[:, self.scaler_columns] = self.scaler.fit_transform(self.train_set[self.scaler_columns])
            self.test_set.loc[:, self.scaler_columns] = self.scaler.transform(self.test_set[self.scaler_columns])
        return self

    @log_history
    def pca(self,n_components=None, columns=None, seed=0):
        if columns is None:
            # print('\nColumns for normalization not explicitely specified. Will use all columns defined as variables.')
            # columns = self.variables
            print('\nColumns for PCA not explicitely specified. Will use all columns defined as variables.')
            columns = self.variables #[c for c in self.data.columns if c not in self.target]
        self.pca_columns = columns

        self.pca_scaler = sklearn.decomposition.PCA(n_components=n_components,random_state=seed)

        if self.train_set is None:
            print('\nNo training dataset is defined: applying normalization to the whole dataset. '
                  '\nYou should not split in train / test datasets after this step.')
            self.data.loc[:, self.pca_columns] = self.pca_scaler.fit_transform(self.data[self.pca_columns])
        else:
            print(
                '\nApplying normalization fitting a scaler to the training dataset and using the same transformation on the test set.')
            train_pca = self.pca_scaler.fit_transform(self.train_set[self.pca_columns])
            test_pca  = self.pca_scaler.transform(self.test_set[self.pca_columns])
            new_columns = ['pca' + str(i) for i in range(n_components)]
        self.train_set = self.train_set.drop(columns=self.pca_columns)
        # self.train_set[new_columns] = train_pca
        self.train_set = pd.DataFrame(np.hstack([self.train_set.values, train_pca]), columns=self.target + new_columns,
                     index=self.train_set.index)
        self.test_set = self.test_set.drop(columns=self.pca_columns)
        # self.test_set[new_columns] = train_pca
        self.test_set = pd.DataFrame(np.hstack([self.test_set.values, test_pca]), columns=self.target + new_columns,
                     index=self.test_set.index)
        self.variables = new_columns
        return self

    @log_history
    def categorical2numeric(self, columns=None, prefix_sep='_', exclude_target_column=True,
                            kwargs={}):
        if columns is None:
            columns = list(self.data.columns[self.data.dtypes == object])
            # columns = [c for c in self.data.columns if any([isinstance(c,o) for o in  [object, pd.api.types.CategoricalDtype]])]
        if (exclude_target_column):
            columns.remove(self.target)
        prefix = columns
        self.data = pd.get_dummies(self.data, columns=columns, prefix=prefix, prefix_sep=prefix_sep, **kwargs)
        if not exclude_target_column:
            self.target = [c for c in self.data.columns if self.target in c]
        return self

    def tsne_fit(self, data, n_components=2, perplexity=30, early_exaggeration=12.0, learning_rate=200.0, n_iter=1000,
                 random_state=1, kwargs={}):

        self.tsne_model = sklearn.manifold.TSNE(n_components=n_components, perplexity=perplexity,
                                                early_exaggeration=early_exaggeration, learning_rate=learning_rate,
                                                n_iter=n_iter,
                                                random_state=random_state, **kwargs)
        self.tsne = {data: self.tsne_model.fit_transform(getattr(self, data))}
        return self

    def plot_tsne(self, color_by=None):
        data = list(self.tsne.keys())[0]
        fig = plt.figure()
        ax = fig.gca()
        if color_by is None:
            if self.tsne[data].shape[1] == 2:
                mappable = ax.scatter(self.tsne[data][:, 0], self.tsne[data][:, 1])
            if self.tsne[data].shape[1] == 3:
                ax = fig.gca(projection='3d')
                mappable = ax.scatter(self.tsne[data][:, 0], self.tsne[data][:, 1], self.tsne[data][:, 2])

        else:
            if self.tsne[data].shape[1] == 2:
                mappable = ax.scatter(self.tsne[data][:, 0], self.tsne[data][:, 1], c=getattr(self, data)[color_by],
                                      cmap=plots.cmap_discretize('jet', len(getattr(self, data)[color_by].unique())))
            if self.tsne[data].shape[1] == 3:
                ax = fig.gca(projection='3d')
                mappable = ax.scatter(self.tsne[data][:, 0], self.tsne[data][:, 1], self.tsne[data][:, 2],
                                      c=getattr(self, data)[color_by],
                                      cmap=plots.cmap_discretize('jet', len(getattr(self, data)[color_by].unique())))

        cb = plt.colorbar(mappable, ax=ax)
        labels = np.arange(int(cb.vmin), int(cb.vmax) + 1, 1)
        loc = labels + .5
        cb.set_ticks(loc)
        cb.set_ticklabels(labels)
        cb.set_label(color_by)
        plt.title('TSNE %s' % data)
        plt.tight_layout()

    def summary(self):
        """
        Brief summary of dataset
        """
        columns = self.data.columns
        print("\nDataset with %d columns." % len(columns))
        if self.id:
            print("  - ID column: %s." % self.id)
        if self.target:
            print("  - Target column(s): %s." % self.target)
        # if self.variables:
        #     print("  - Variables: %s    (numeric: %s; string/categorical: %s)" % (
        #         self.variables, self.numeric_variables, self.categorical_variables))


if __name__ == "__main__":
    pass
