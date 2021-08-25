FROM ubuntu:20.04

RUN apt-get update && \
	DEBIAN_FRONTEND=noninteractive apt-get install -y tzdata && \
	apt-get remove -y python && \
	apt-get install -y --no-install-recommends \
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

# TODO: install python 3.9 and make sure python means python3.9!
RUN apt-get install -y --no-install-recommends \
	python3 \
	python3-dev \
	python3-tk \
	python3-venv \
	python3-pip && \
    pip3 install --no-cache-dir --upgrade setuptools && \
	echo "alias python='python3'" >> /root/.bash_aliases && \
	echo "alias pip='pip3'" >> /root/.bash_aliases && \
	cd /usr/local/bin && ln -s /usr/bin/python3 python && \
	cd /usr/local/bin && ln -s /usr/bin/pip3 pip && \
    pip install virtualenv abed wheel

# Clone the repo
RUN git clone --recurse-submodules https://github.com/simontrapp/TCPDBench

# TODO: COPY the datasets into the benchmark dir, overwrite annotations.json/make_table.py/abed_conf.py
# TODO: create analysis/output/summaries/
RUN mkdir -p /TCPDBench/datasets && cp TCPD/export/*.json /TCPDBench/datasets/

# Install Python dependencies
RUN pip install -r /TCPDBench/analysis/requirements.txt

# Set the working directory
WORKDIR TCPDBench
