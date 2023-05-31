# FROM pytorch/pytorch:1.10.0-cuda11.3-cudnn8-runtime
FROM singularitybase.azurecr.io/validations/base/singularity-tests:20210729T152030606 AS validator
FROM singularitybase.azurecr.io/base/job/pytorch/1.8.0-cuda11.1-cudnn8-devel:20211115T220336575

# Fix pubkey not available error
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub
# Install some basic utilities
RUN apt-get update && apt-get install -y software-properties-common && \
    add-apt-repository -y ppa:redislabs/redis && apt-get update && \
    apt-get install -y curl ca-certificates sudo git ssh bzip2 libx11-6 gcc iputils-ping \
    libsm6 libxext6 libxrender-dev graphviz tmux htop build-essential wget cmake libgl1-mesa-glx redis && \
    rm -rf /var/lib/apt/lists/*
# Manually install libffi7. This is a workaround for azureml.
RUN wget -q http://cz.archive.ubuntu.com/ubuntu/pool/main/libf/libffi/libffi7_3.3-4_amd64.deb && \
    yes | dpkg -i libffi7_3.3-4_amd64.deb && \
    rm -rf libffi7_3.3-4_amd64.deb

# Install Miniconda and Python 3.8
# ENV CONDA_AUTO_UPDATE_CONDA=false
# ENV PATH=/miniconda/bin:$PATH
# RUN cd / && curl -sLo /miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-py38_4.10.3-Linux-x86_64.sh \
#     && chmod +x /miniconda.sh \
#     && /miniconda.sh -b -p /miniconda \
#     && rm /miniconda.sh \
#     && conda install -y python==3.8 \
#     && conda clean -ya \
#     && ln -s /miniconda/bin/python /usr/local/bin/python \
#     && ln -s /miniconda/bin/python3 /usr/local/bin/python3

# A workaround to install ruamel.yaml
RUN pip install --ignore-installed --no-cache-dir ruamel.yaml llvmlite

# RUN conda install -y tensorboard scikit-learn numpy requests scipy numba\
#     && conda clean -ya

RUN pip install --no-cache-dir tqdm pyyaml utilsd sktime reformer_pytorch scipy numpy==1.21 pandas==1.4.1 --upgrade
RUN wget -q -O azcopy.tar.gz https://aka.ms/downloadazcopy-v10-linux && \
    tar -xf azcopy.tar.gz && \
    cp azcopy_*/azcopy /usr/local/bin && \
    rm -r azcopy.tar.gz azcopy_* && \
    chmod +x /usr/local/bin/azcopy

COPY dist/physiopro-*.whl .
RUN pip install *.whl && rm *.whl

WORKDIR /workspace
RUN chmod -R a+w /workspace

COPY --from=validator /validations /opt/microsoft/_singularity/validations/

ENV SINGULARITY_IMAGE_ACCELERATOR=NVIDIA

RUN /opt/microsoft/_singularity/validations/validator.sh
