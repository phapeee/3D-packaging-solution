FROM ubuntu:20.04

# Prevent interactive tzdata prompt
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    wget \
    curl \
    git \
    ca-certificates \
    llvm \
    libssl-dev \
    zlib1g-dev \
    libncurses5-dev \
    libncursesw5-dev \
    libreadline-dev \
    libsqlite3-dev \
    libgdbm-dev \
    libdb5.3-dev \
    libbz2-dev \
    libexpat1-dev \
    liblzma-dev \
    tk-dev \
    uuid-dev \
    libffi-dev \
    python3-openssl \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /usr/src

ENV PYVER=3.6.5

# Download, build, install Python
RUN wget https://www.python.org/ftp/python/${PYVER}/Python-${PYVER}.tgz \
 && tar xzf Python-${PYVER}.tgz \
 && cd Python-${PYVER} \
 && ./configure --enable-optimizations \
 && make -j"$(nproc)" altinstall \
 && cd /usr/src \
 && rm -rf Python-${PYVER} Python-${PYVER}.tgz

# Ensure pip and verify
RUN /usr/local/bin/python3.6 -m ensurepip --upgrade \
 && /usr/local/bin/python3.6 -m pip install --upgrade pip

WORKDIR /app

# Copy requirements first
COPY requirements.txt .

RUN pip install -r requirements.txt

# Copy app
COPY . .

EXPOSE 8002

CMD ["gunicorn", "--bind", "0.0.0.0:8002", "--workers", "2", "--threads", "4", "main:app"]