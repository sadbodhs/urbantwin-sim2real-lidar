# OpenPCDet (pinned commit) on PyTorch 2.4.1 / CUDA 12.1, built for the RTX 3090 (sm_86).
FROM pytorch/pytorch:2.4.1-cuda12.1-cudnn9-devel

ENV DEBIAN_FRONTEND=noninteractive TORCH_CUDA_ARCH_LIST="8.6" FORCE_CUDA=1 PIP_NO_CACHE_DIR=1
RUN apt-get update && apt-get install -y --no-install-recommends git build-essential ninja-build libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install "numpy==1.26.4" "numba==0.60.0" "llvmlite==0.43.0" "spconv-cu120==2.3.6" \
    "tensorboardX==2.6.2.2" "easydict==1.13" "pyyaml==6.0.2" "scikit-image==0.24.0" "tqdm==4.66.5" \
    "SharedArray==3.2.4" "opencv-python-headless==4.10.0.84" "pyquaternion==0.9.9" "plyfile==1.1" \
    "scipy==1.14.1" "pandas==2.2.3" "POT==0.9.5" "matplotlib==3.9.2"

ARG PCDET_COMMIT=233f849829b6ac19afb8af8837a0246890908755
RUN git clone https://github.com/open-mmlab/OpenPCDet /opt/OpenPCDet \
    && cd /opt/OpenPCDet && git checkout $PCDET_COMMIT \
    && python setup.py develop \
    && sed -i -e '/argo2_dataset import Argo2Dataset/d' -e "/'Argo2Dataset': Argo2Dataset/d" pcdet/datasets/__init__.py
# (the Argoverse-2 loader needs the av2 package; it is not used here)

RUN pip install "pytest==8.3.3"

ENV PYTHONPATH=/workspace:/opt/OpenPCDet
WORKDIR /workspace
