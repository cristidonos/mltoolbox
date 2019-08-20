import os
import sys

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn import preprocessing

import mltoolbox
from mltoolbox.algorithms import Hyperoptimizer, Ptsne, TsneMapper, classifier_candidates
from mltoolbox.datasets import dset
from mltoolbox.exploration import FacetsExploration
from mltoolbox.forecasts import ForecastProphet
from mltoolbox.misc.generics import timing

if __name__ == '__main__':

    seed = 1

    # 1. Set `PYTHONHASHSEED` environment variable at a fixed value
    os.environ['PYTHONHASHSEED'] = str(seed)

    # 2. Set `python` built-in pseudo-random generator at a fixed value
    import random

    random.seed(seed)

    # 3. Set `numpy` pseudo-random generator at a fixed value
    import numpy as np
    np.random.seed(seed)

    # 4. Set `tensorflow` pseudo-random generator at a fixed value
    import tensorflow as tf
    tf.set_random_seed(seed)

    # 5. Configure a new global `tensorflow` session
    from keras import backend as K

    session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
    sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
    K.set_session(sess)

    def test1_fbprophet():
        if 'fbprophet' not in sys.modules:
            print(
                'Missing "fbprophet" module, will skip "Test 1" which involve this module.'
            )
        else:

            print(
                '''
                --> Test #1: Forecasting with fbprophet.

                We use footfal data from the streets of York (open-source dataset from
                https://data.yorkopendata.org/dataset/footfall) to test the ForecastProphet class.
                For test purpose, only a subset of the dataset is used (Coney Street, last 120 days).
                Data split in training / test sets, z-scored, and UK holidays are added before fitting the model.
                Footfall forecast is showed in Figure 1, along with  goodness of fit and
                residual plots.
                '''
            )

            package_path = os.path.dirname(mltoolbox.__file__).replace(os.path.sep + 'lib' + os.path.sep,
                                                                       os.path.sep + 'Lib' + os.path.sep)

            # data = pd.read_csv(r'./datasets/footfallhourly.csv')
            filepath = package_path + os.path.join(os.sep, 'datasets', 'footfallhourly.csv')
            data = pd.read_csv(filepath)
            data = data[data['LocationName'] == 'Coney Street'][
                ['Date', 'TotalCount']
            ].copy()

            data = data.tail(24 * 120)  # keep 120 days
            data.Date = pd.to_datetime(data.Date, format='%d/%m/%Y %H:%M:%S').dt.strftime(
                '%Y-%m-%d %H:%M:%S'
            )
            data.index = data.Date
            data = data.drop(columns=['Date'])

            # convert to dset
            ds = dset(data=data, target=['TotalCount'])

            (ds
             .remove_nan_rows(any_nan=True)
             .split_train_test_sets(train_test_split=0.2, shuffle_split=False, seed=0)
             .normalize(columns=[ds.target], type='zscore')
             .summary()
             )

            ff = ForecastProphet(
                input_dset=ds, input_y_column='TotalCount', input_ds_column='index'
            )

            # ff.add_seasonality(name='monthly', period=12 * 30.5, fourier_order=5)
            ff.add_country_holidays(country_name='UK')

            ff.evaluate_train_test()
            ff.prophet.plot_components(ff.forecast)
        return ff


    def test2_facets():
        print(
            '''
            --> Test #2: Data exploration with Facets

            Dataset: Diamonds dataset from seaborn.
            Data split 40%-60% to showcase comparison of two datasets.
            '''
        )

        data = sns.load_dataset('diamonds')
        fo = FacetsExploration(
            input_data={
                'dataset1': data.sample(n=6000, random_state=0),
                'dataset2': data.sample(n=9000, random_state=0),
            }
        )
        fo.generate_overview_html_string()
        fo.generate_dive_html_string(data=fo.data['dataset2'])

        fo.show(type='overview')
        fo.show(type='dive')
        return fo


    def test3_ptsne():
        '''
        Function "Ptsne.fit_ptsne" - Elapsed time: 2620.52 seconds.
        Transforming train.
        Transforming test.
        Function "Ptsne.transform_ptsne" - Elapsed time: 0.20 seconds.
        Starting hyperoptimization of 2 algorithms.
        Model: Training xgb... took 31.790 seconds.
          * Test score: [0.9386]
        Model: Training gpc... took 11.130 seconds.
          * Test score: [0.9211]
        Function "Hyperoptimizer.run" - Elapsed time: 43.14 seconds.
                score runtime_seconds
        xgb  0.938596         31.7903
        gpc  0.921053         11.1302
        '''
        from sklearn.datasets import load_breast_cancer
        data = load_breast_cancer()
        data = pd.concat(
            [pd.DataFrame(data.data, columns=data.feature_names), pd.DataFrame(data.target, columns=['cancer'])],
            axis=1)

        ds = dset(data, target=["cancer"], header=0, target_type='')

        (ds
         .remove_nan_rows(columns=[ds.target], any_nan=True)
         .remove_nan_columns(any_nan=True)
         .categorical2numeric(exclude_target_column=False)
         .split_train_test_sets(train_test_split=0.2, shuffle_split=True, seed=0)
         .normalize(type='gaussian')
         .summary()
         )

        ptsne = Ptsne(ds, perplexities=None, n_components=2, kwargs={'batch_size': 256})
        ptsne.fit_ptsne(epochs=10000, kwargs={'verbose': True})
        ptsne.transform_ptsne()
        ptsne.plot2d()

        hopt = (Hyperoptimizer(candidates=classifier_candidates, type='classification',
                               x_train=ptsne.fitted_train,
                               y_train=ptsne.labels_train,
                               x_test=ptsne.fitted_test,
                               y_test=ptsne.labels_test,
                               n_iter=25)
                .run()
                .plot_comparison(mesh_resolution=None)
                .summary()
                )
        hopt.get_best_algorithm()

        return ptsne, hopt

    @timing
    def test4_tsneclassifier():

        print(
            '''
            --> Test #4: t-SNE classifier
            
            t-SNE reduces dimensionality of the dataset, usually to ndim=2, which allows better visualization.
            As t-SNE has no analytical way of representing new data into the low dimensional space, 
            a neural network is used the learn the t-SNE mapping from high to low dimensional space.
            Once a dataset (train + test) has its dimension reduced, a hyperoptimizer is used to find
            the classifier that best learns the data. 

            Dataset: Generated using sklearn's make_classification: 1000 samples, 4 classes, 10 features (6 information features, 4 redundant features).
            Data split 50%-50% in training / test.
            '''
        )

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
         .split_train_test_sets(train_test_split=0.5, shuffle_split=True, seed=0)
         .normalize(type='robust')
         .summary()
         )

        variables = [c for c in ds.variables if c not in ds.target]
        hopt_regular = (Hyperoptimizer(candidates=classifier_candidates, type='classification',
                                       x_train=ds.train_set[variables],
                                       y_train=ds.labels_train,
                                       x_test=ds.test_set[variables],
                                       y_test=ds.labels_test,
                                       n_iter=20)
                        .run()
                        )

        t = TsneMapper(input_dset=ds, n_components=2, perplexity=12, early_exaggeration=12, n_iter=1000,
                       random_state=0)
        t.tsne_fit()
        t.plot_tsne_fit()
        t.fit(epochs=1500, hidden_neurons=[200, 200, 200], activation='tanh', dropout=0.25, validation_split=0.2,
              verbose=0, metrics=[t.euclidean_distance_loss])
        t.transform2tsne()

        plt.figure()
        plt.plot(t.keras_model_history.history['loss'],)
        plt.plot(t.keras_model_history.history['val_loss'])
        plt.title('Loss')
        plt.ylabel('Euclidean distance')
        plt.xlabel('epoch')
        plt.legend(['train', 'val'], loc='upper right')

        le = preprocessing.LabelEncoder()
        train_labels = le.fit_transform(t.labels_train.values)
        test_labels = le.fit_transform(t.labels_test.values)
        plt.figure()
        plt.scatter(t.tsne['train_set'][:, 0], t.tsne['train_set'][:, 1], c=train_labels)
        plt.scatter(t.fitted_train[:, 0], t.fitted_train[:, 1], marker='s', c=train_labels, edgecolors='r')
        plt.scatter(t.fitted_test[:, 0], t.fitted_test[:, 1], marker='d', c=test_labels, edgecolors='k')
        plt.title(
            't-SNE parametrization (circle - tSNE, square - Keras mapping (train), diamond - Keras mapping (test) )')

        hopt_tsne = (Hyperoptimizer(candidates=classifier_candidates, type='classification',
                                    x_train=t.fitted_train,
                                    y_train=t.labels_train,
                                    x_test=t.fitted_test,
                                    y_test=t.labels_test,
                                    n_iter=20, seed=0)
                     .run()
                     .plot_comparison(mesh_resolution=None)
                     )




        print('\n Classification results on high-dimensional dataset  (with hyperoptimization)')
        hopt_regular.summary()
        print('\n Classification results on low-dimensional dataset (with hyperoptimization)')
        hopt_tsne.summary()


    print(
        '''
        *** Handyman's Toolbox showcase ***
        
        Type:
        test1_fbprophet()       for fbprophet forcast 
        test2_facets()          for facets exploration 
        test3_ptsne()           for classification with parametric t-SNE (slow and not as good as t-SNE classifier below)
        test4_tsneclassifier()  for t-SNE classifier and hyperoptimization
        '''
    )
