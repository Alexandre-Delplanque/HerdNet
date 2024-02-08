
ARG PYTORCH="1.11.0"
ARG CUDA="11.3"
ARG CUDNN="8"

FROM pytorch/pytorch:${PYTORCH}-cuda${CUDA}-cudnn${CUDNN}-devel AS base
ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0+PTX 9.0"
ENV TORCH_NVCC_FLAGS="-Xfatbin -compress-all"

RUN conda update conda && conda install pip && conda clean -afy
WORKDIR /herdnet
COPY environment-dev.yml ./
RUN conda env update -f environment-dev.yml -n base && conda clean -afy

FROM base AS dev
# Install make for the docs build
RUN apt-get update \
    && apt-get install -y gcc make build-essential git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-dev.txt ./
RUN pip install -r requirements-dev.txt
COPY . ./
# pre-commit run --all-files fails w/o this line
RUN git init
RUN pip install -e .

