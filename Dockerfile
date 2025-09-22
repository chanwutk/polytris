FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04

RUN sed -i 's/#force_color_prompt=yes/force_color_prompt=yes/g' ~/.bashrc

# Set the working directory
WORKDIR /polyis

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update -qq && \
    apt-get install -qq -y git ninja-build gettext cmake unzip curl && \
    git clone https://github.com/neovim/neovim && \
    cd neovim && \
    git checkout stable && \
    make CMAKE_BUILD_TYPE=Release && \
    make -j40 install && \
    cd .. && \
    rm -rf neovim && \
    apt-get remove -qq -y ninja-build gettext cmake unzip curl && \
    apt-get clean && \
    apt-get -y autoremove --purge && \
    rm -rf /var/lib/apt/lists/* /var/log/dpkg.log

# Install Conda
RUN apt-get -qq update && apt-get -qq -y install curl bzip2 \
    && curl -sSL https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -o /tmp/miniconda.sh \
    && bash /tmp/miniconda.sh -bfp /usr/local \
    && rm -rf /tmp/miniconda.sh \
    && conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main \
    && conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r \
    && conda install -y python=3.13 \
    && conda update conda \
    && apt-get -qq -y remove curl bzip2 \
    && apt-get -qq -y autoremove --purge \
    && apt-get -qq -y autoclean \
    && rm -rf /var/lib/apt/lists/* /var/log/dpkg.log \
    && conda clean --all --yes

ENV PATH /opt/conda/bin:$PATH

# Install FNM
RUN apt-get -qq update && apt-get -qq -y install curl unzip && \
    curl -fsSL https://fnm.vercel.app/install | bash && \
    ~/.local/share/fnm/fnm install --lts && \
    apt-get -qq -y remove curl unzip && \
    apt-get -qq -y autoremove --purge && \
    apt-get -qq -y autoclean && \
    rm -rf /var/lib/apt/lists/* /var/log/dpkg.log

# Install dependencies
RUN apt-get -qq update -y && \
    apt-get -qq -y install --no-install-recommends \
    build-essential \
    git \
    ffmpeg \
    curl \
    make && \
    rm -rf /var/lib/apt/lists/*

ENV PATH="/usr/local/cuda/bin:${PATH}"
ENV LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH}"

ENV PYTHONPATH="/polyis/python:${PYTHONPATH}"

RUN conda init bash

COPY ./environment.yml /polyis/environment.yml
RUN conda env update --file /polyis/environment.yml --prune --name base && \
    conda clean --all --yes

COPY ./requirements.txt /polyis/requirements.txt
RUN pip install --no-build-isolation -r /polyis/requirements.txt

RUN git clone https://github.com/chanwutk/lazyvim.git ~/.config/nvim

# RUN echo "127.0.0.1 host.docker.internal" | tee -a /etc/hosts
