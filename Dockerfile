FROM continuumio/miniconda3:4.8.3

WORKDIR home

RUN apt-get -y install libc-dev
RUN apt-get -y install gcc g++
RUN pip install -U pip

RUN pip install convertdate lunarcalendar holidays hana_ml pystan

RUN conda install pandas numpy scipy scikit-learn seaborn jupyter -y \
    && conda clean -afy \
    && find /opt/conda/ -follow -type f -name '*.a' -delete \
    && find /opt/conda/ -follow -type f -name '*.pyc' -delete \
    && find /opt/conda/ -follow -type f -name '*.js.map' -delete \
    && find /opt/conda/lib/python*/site-packages/bokeh/server/static -follow -type f -name '*.js' ! -name '*.min.js' -delete

RUN pip install fbprophet

RUN pip install git+https://github.com/cristidonos/mltoolbox

