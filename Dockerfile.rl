FROM python:3.10-slim

WORKDIR /workspace

RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    cmake \
    libboost-all-dev \
    && rm -rf /var/lib/apt/lists/*

RUN python -m pip install --upgrade pip setuptools wheel

RUN git clone https://github.com/cityflow-project/CityFlow.git /tmp/CityFlow && \
    sed -i.bak -E 's/cmake_minimum_required\(VERSION[[:space:]]+[0-9]+\.[0-9]+\)/cmake_minimum_required(VERSION 3.5)/' /tmp/CityFlow/CMakeLists.txt && \
    cd /tmp/CityFlow && \
    python -m pip install . && \
    rm -rf /tmp/CityFlow

RUN python -m pip install --index-url https://download.pytorch.org/whl/cpu torch
RUN python -m pip install stable-baselines3 gymnasium tensorboard numpy pandas matplotlib

WORKDIR /sample-code