from setuptools import setup


requirements = ['Click>=6.0',
                'pandas==0.24',
                'seaborn',
                'scipy',
                'scikit-learn<=0.20.3',
                'jupyterlab',
                'scikit-optimize',
                'xgboost',
                'keras',
                'tensorflow',
                'nbparameterise @ git+https://github.com/takluyver/nbparameterise.git',
                ]


setup(
    name='mltoolbox',
    version='',
    # packages=['external.facets_old.facets_overview.python', 'external.parametric_tsne.parametric_tSNE', 'mltoolbox',
    #           'mltoolbox.misc'],
    packages=['mltoolbox',
              'mltoolbox.misc',
              'mltoolbox.external',
              'mltoolbox.external.parametric_tsne.parametric_tSNE',
              'mltoolbox.external.facets',
              ],
    url='',
    license='',
    author='Cristian Donos',
    author_email='',
    description='',
    install_requires = requirements,
)
