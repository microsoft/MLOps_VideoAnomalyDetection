FROM mcr.microsoft.com/azureml/base-gpu:intelmpi2018.3-cuda9.0-cudnn7-ubuntu16.04

# Install Horovod, temporarily using CUDA stubsddd
RUN ldconfig /usr/local/cuda/lib64/stubs && \
    conda install -y python=3.6.2 && conda clean -ay && \
    # Install AzureML SDK
    pip install --no-cache-dir azureml-defaults && \
    # Install PyTorch
    pip install --no-cache-dir tensorflow==1.8.0 tensorflow-gpu==1.8.0 keras==2.0.8 matplotlib==3.0.3 seaborn==0.9.0 requests==2.21.0 bs4==0.0.1 imageio==2.5.0 sklearn pandas==0.24.2 numpy==1.16.2 hickle==3.4.3 && \
    # Install Horovod
    pip install --no-cache-dir horovod==0.13.5 && \
    ldconfig
