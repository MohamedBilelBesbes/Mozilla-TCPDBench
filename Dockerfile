FROM ubuntu:20.04

RUN apt-get update && \
	DEBIAN_FRONTEND=noninteractive apt-get install -y tzdata && \
	apt-get remove -y python && \
	apt-get install -y --no-install-recommends \
        wget \
        software-properties-common \
		git \
		build-essential \
		r-base \
		r-base-dev \
		r-cran-rcppeigen \
		latexmk \
		texlive-latex-extra \
		libopenmpi-dev \
		liblzma-dev \
		libgit2-dev \
		libxml2-dev \
		libcurl4-openssl-dev \
		libssl-dev \
		libopenblas-dev \
		libfreetype6-dev \
		libv8-dev

RUN add-apt-repository ppa:deadsnakes/ppa && apt-get update && apt-get install -y --no-install-recommends \
	python3.9 \
	python3.9-dev \
	python3.9-tk \
	python3.9-distutils \
    python3.9-venv \
    python3-pip && \
    python3.9 -m pip install --no-cache-dir --upgrade setuptools && \
    python3.9 -m pip install virtualenv abed wheel jsonschema

# Clone the repo
RUN git clone https://github.com/simontrapp/TCPDBench

# copy the datasets into the benchmark dir, overwrite annotations.json/make_table.py/abed_conf.py
ADD datasets /TCPDBench/datasets
COPY annotations.json /TCPDBench/analysis/annotations/
COPY make_table.py /TCPDBench/analysis/scripts/
COPY abed_conf.py /TCPDBench/

# create analysis/output/summaries/
RUN mkdir -p /TCPDBench/analysis/output/summaries && mkdir -p /TCPDBench/abed_results

# Install Python dependencies
RUN python3.9 -m pip install -r /TCPDBench/analysis/requirements.txt

# Set the working directory
WORKDIR TCPDBench
