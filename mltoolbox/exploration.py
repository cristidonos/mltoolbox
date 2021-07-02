import base64
import os
import sys
import tempfile
import webbrowser
from time import sleep

import numpy as np
import pandas as pd

import mltoolbox
from mltoolbox.misc.generics import timing

# sys.path.insert(0, './external/facets_old/facets_overview/python/')
# package_path = os.path.dirname(mltoolbox.__file__).replace(os.sep + 'lib' + os.sep,
#                                                            os.sep + 'Lib' + os.sep)
# sys.path.insert(0, package_path + os.sep + os.sep.join(['external', 'facets_old', 'facets_overview', 'python']) + os.sep)

from mltoolbox.external.facets.generic_feature_statistics_generator import GenericFeatureStatisticsGenerator


def mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


class FacetsExploration:
    """
    Class for interactive data exploration.py with Facets Overview  (https://github.com/PAIR-code/facets_old)

    Requires Jupyter Notebooks
    """

    def __init__(self, input_data):

        data, protostr = self.prepare_data(input_data)
        self.data = data
        self.protostr = protostr
        self.overview_html_string = None
        self.dive_html_string = None

    @timing
    def prepare_data(self, input_data):
        data = None
        protostr = None
        if isinstance(input_data, pd.DataFrame):
            data = {'data01': input_data}
        else:
            if (isinstance(input_data, dict)) & (
                    all(
                        [isinstance(input_data[k], pd.DataFrame) for k in input_data.keys()]
                    )
            ):
                data = input_data
            else:
                raise Exception(
                    'Input data for FacetsOverview should be a pandas pataframe or a dictionary of dataframes'
                )

        proto = GenericFeatureStatisticsGenerator().ProtoFromDataFrames(
            [{'name': k, 'table': v} for k, v in data.items()]
        )
        protostr = base64.b64encode(proto.SerializeToString()).decode('utf-8')
        return data, protostr

    def generate_overview_html_string(self):
        if self.protostr is not None:
            HTML_TEMPLATE = """
                    <script src="https://cdnjs.cloudflare.com/ajax/libs/webcomponentsjs/1.3.3/webcomponents-lite.js"></script>
                    <link rel="import" href="https://raw.githubusercontent.com/PAIR-code/facets/master/facets-dist/facets-jupyter.html" >
                    <facets-overview id="elem"></facets-overview>
                    <script>
                      document.querySelector("#elem").protoInput = "{protostr}";
                    </script>"""
            html_string = HTML_TEMPLATE.format(protostr=self.protostr)
            self.overview_html_string = html_string
        else:
            print('Cant generate HTML, probably a serialization error.')

    def generate_dive_html_string(self, data=None):
        if data is not None:
            sprite_size = 32 if len(data.index) > 50000 else 64
            jsonstr = data.to_json(orient='records')
            HTML_TEMPLATE = """
                    <script src="https://cdnjs.cloudflare.com/ajax/libs/webcomponentsjs/1.3.3/webcomponents-lite.js"></script>
                    <link rel="import" href="https://raw.githubusercontent.com/PAIR-code/facets/master/facets-dist/facets-jupyter.html" >
                    <facets-dive sprite-image-width="{sprite_size}" sprite-image-height="{sprite_size}" id="elem" height="1000"></facets-dive>
                    <script>
                      document.querySelector("#elem").data = {jsonstr};
                    </script>
                    """
            html_string = HTML_TEMPLATE.format(jsonstr=jsonstr, sprite_size=sprite_size)
            self.dive_html_string = html_string
        else:
            print('Cant generate HTML, probably a serialization error.')

    def show(self, type, save_to=None):
        if type == 'overview':
            html_string = self.overview_html_string
        if type == 'dive':
            html_string = self.dive_html_string
        if save_to is None:
            htmlfile = tempfile.NamedTemporaryFile(
                mode='wt', suffix='.html', delete=False
            )
        else:
            htmlfile = open(save_to,'wt')

        htmlfile.write(html_string)
        htmlfile.close()
        webbrowser.open_new(htmlfile.name)
        sleep(3)
        if save_to is None:
            os.remove(htmlfile.name)


if __name__ == '__main__':
    pass
