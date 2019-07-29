import sys
import os

import pandas as pd
import seaborn as sns

import mltoolbox
from mltoolbox.exploration import FacetsExploration
from mltoolbox.forecasts import ForecastProphet
from mltoolbox.classifiers import Ptsne, Hyperoptimizer, candidates
from mltoolbox.datasets import dset

if __name__ == '__main__':

    def test1_fbprophet():
        package_path = os.path.dirname(mltoolbox.__file__).replace(os.path.sep + 'lib' + os.path.sep,
                                                                   os.path.sep + 'Lib' + os.path.sep)

        # data = pd.read_csv(r'./datasets/footfallhourly.csv')
        filepath = package_path + os.path.join(os.sep, 'datasets', 'footfallhourly.csv')
        data = pd.read_csv(filepath)

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
                Data is z-scored and monthly seasonality and UK holidays are added before fitting the model.
                Footfall forecast for 7 days is showed in Figure 1, along with  goodness of fit and
                residual plots.
                '''
            )

            data = data[data['LocationName'] == 'Coney Street'][
                ['Date', 'TotalCount']
            ].copy()

            data = data.tail(24 * 120)  # keep last year
            data.Date = pd.to_datetime(data.Date, format='%d/%m/%Y %H:%M:%S').dt.strftime(
                '%Y-%m-%d %H:%M:%S'
            )

            ff = ForecastProphet(
                input_data=data, input_ds_column='Date', input_y_column='TotalCount'
            )
            ff.zscore_filter(df='data')

            ff.add_seasonality(name='monthly', period=12 * 30.5, fourier_order=5)
            ff.add_country_holidays(country_name='UK')

            ev, train, test = ff.evaluate_train_test(test_size=7 * 24, verbose=False)

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
                'dataset1': data.sample(n=6000),
                'dataset2': data.sample(n=9000),
            }
        )
        fo.generate_overview_html_string()
        fo.generate_dive_html_string(data=fo.data['dataset2'])

        fo.show(type='overview')
        fo.show(type='dive')

    def test3_ptsne():
        '''
        Function "Ptsne.fit_ptsne" - Elapsed time: 492.06 seconds.
        Transforming train.
        Transforming test.
        Function "Ptsne.transform_ptsne" - Elapsed time: 0.12 seconds.
        Starting hyperoptimization of 2 algorithms.
        Model: Training xgb... took 65.154 seconds.
          * Test score: [0.9386]
        Model: Training gpc... took 5.198 seconds.
          * Test score: [0.9298]
        Function "Hyperoptimizer.run" - Elapsed time: 70.56 seconds.
                score runtime_seconds
        xgb  0.938596         65.1541
        gpc  0.929825         5.19844
        '''
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
         .normalize(type='unitnorm')
         .summary()
         )

        ptsne = Ptsne(ds, perplexities=None, n_components=2, kwargs={'batch_size' :256})
        ptsne.fit_ptsne(epochs=5000)  # , kwargs={'verbose' : True})
        ptsne.transform_ptsne()
        ptsne.plot2d()

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



    print(
        '''
        *** Handyman's Toolbox showcase ***
        '''
    )






