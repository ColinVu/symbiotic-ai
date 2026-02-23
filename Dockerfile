# Dockerfile for Symbiotic AI with HTK HMM State Detection
# This container includes Python, all dependencies, and HTK toolkit

FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    wget \
    libgl1 \
    libglx0 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install HTK 3.4.1
# Note: HTK requires registration. You need to download HTK-3.4.1.tar.gz from http://htk.eng.cam.ac.uk/
# and place it in the same directory as this Dockerfile, OR use a pre-built version
RUN mkdir -p /opt/htk && \
    cd /opt/htk

# Option 1: If you have HTK-3.4.1.tar.gz in the build context
# COPY HTK-3.4.1.tar.gz /opt/htk/
# RUN cd /opt/htk && \
#     tar -xzf HTK-3.4.1.tar.gz && \
#     cd htk && \
#     ./configure --prefix=/usr/local --disable-hlmtools && \
#     make all && \
#     make install && \
#     cd / && \
#     rm -rf /opt/htk

# Option 2: Placeholder for manual HTK installation
# You will need to manually install HTK after building the container
# See DOCKER_SETUP.md for instructions

# Add HTK to PATH (once installed)
ENV PATH="/usr/local/bin:${PATH}"

# Copy only requirements first for better caching
COPY requirements_docker.txt /app/requirements_docker.txt

# Install Python dependencies
RUN pip install --no-cache-dir -r /app/requirements_docker.txt

# Copy the entire symbiote codebase
COPY symbiote/ /app/symbiote/

# Create directories for data, models, and outputs
RUN mkdir -p /data/videos /data/annotations /data/aruco_config /models /outputs

# Set Python path
ENV PYTHONPATH="/app:${PYTHONPATH}"

# Default command shows help
CMD ["python", "-m", "symbiote.cli.main", "--help"]
