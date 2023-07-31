FROM pytorch/pytorch:1.11.0-cuda11.3-cudnn8-runtime AS base
RUN conda update conda && conda install pip && conda clean -afy
WORKDIR /herdnet
COPY environment-dev.yml ./
RUN conda env update -f environment-dev.yml -n base && conda clean -afy


#FROM base as dep_builder
## Some deps must be built (e.g. against the conda GDAL)
#RUN apt-get update \
#    && apt-get install -y gcc build-essential \
#    && rm -rf /var/lib/apt/lists/*
#COPY pyproject.toml setup.cfg ./
#COPY src/stactools/core/__init__.py src/stactools/core/
## Install dependencies but remove the actual package
#RUN pip install --prefix=/install .[all] \
#    && rm -r /install/lib/*/site-packages/stactools*


FROM base AS dev
# Install make for the docs build
RUN apt-get update \
    && apt-get install -y gcc make build-essential git \
    && rm -rf /var/lib/apt/lists/*
#COPY --from=dep_builder /install /opt/conda
#RUN conda install -c conda-forge pandoc && conda clean -af
COPY requirements-dev.txt ./
RUN pip install -r requirements-dev.txt
COPY . ./
# pre-commit run --all-files fails w/o this line
RUN git init
RUN pip install -e .


#FROM base AS main
#COPY --from=dep_builder /install /opt/conda
#COPY src ./src
#COPY pyproject.toml setup.cfg ./
#RUN pip install .[all]
