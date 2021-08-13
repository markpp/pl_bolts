FROM nvidia/cuda:11.1.1-cudnn8-devel-ubuntu20.04

LABEL maintainer "Mark - Auto generated"

ENV DEBIAN_FRONTEND=noninteractive

# Install standard tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    apt-utils software-properties-common dirmngr build-essential \
    gpg-agent lsb-release git curl unzip wget gedit nano cmake \
    pkg-config checkinstall yasm libopenblas-dev

# install python3
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-dev python3-tk python3-setuptools

# Update pip
RUN pip3 --no-cache-dir install --upgrade \
    pip setuptools

# Install basal python packages
RUN pip3 install numpy scipy sklearn gensim umap-learn pandas matplotlib

RUN apt-get update && apt-get install -y libopencv-dev python3-opencv

# Install pytorch
RUN pip3 install torch==1.8.0+cu111 torchvision==0.9.0+cu111 -f https://download.pytorch.org/whl/torch_stable.html

RUN pip3 install pytorch-lightning wandb

# Install extra python packages
RUN pip3 install packaging imageio tqdm dlutils scikit-image ipdb lmdb easing-functions

RUN rm -rf /var/lib/apt/lists/*


COPY user.sh /usr/local/bin/
CMD bash -C '/usr/local/bin/user.sh'
