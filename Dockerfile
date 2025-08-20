# Dockerfile.py36
FROM ubuntu:18.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHON_VERSION=3.6.5 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# Build Python 3.6.5
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl ca-certificates zlib1g-dev libffi-dev \
    libssl1.0-dev libbz2-dev libreadline-dev libsqlite3-dev \
    libncurses5-dev libncursesw5-dev xz-utils && \
    curl -fsSLo /tmp/Python.txz https://www.python.org/ftp/python/3.6.5/Python-3.6.5.tar.xz && \
    mkdir -p /tmp/Python-src && tar -xf /tmp/Python.txz -C /tmp/Python-src --strip-components=1 && \
    cd /tmp/Python-src && ./configure --enable-optimizations && make -j"$(nproc)" && make install && \
    rm -rf /var/lib/apt/lists/* /tmp/Python* && \
    ln -s /usr/local/bin/python3.6 /usr/local/bin/python && \
    ln -s /usr/local/bin/pip3.6 /usr/local/bin/pip

WORKDIR /app

# Copy requirements first
COPY requirements.txt .
RUN pip install --upgrade "pip<21" "setuptools<58" "wheel<0.38" && \
    pip install -r requirements.txt

# Copy app
COPY . .

EXPOSE 8002

# Gunicorn works with py3.6; use gthread worker
CMD ["gunicorn", "--bind", "0.0.0.0:8002", "--workers", "2", "--threads", "4", "app:app"]
