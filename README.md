# Turing Change Point Detection Benchmark
This is a fork of the original [TCPDBench](https://github.com/alan-turing-institute/TCPDBench) with the goal to make it easier to configure and deploy. This fork only supports one-dimensional datasets!

## Running the experiments with Docker
1. Create a `Dockerfile` like the following example:
```dockerfile
FROM simont1105/tcpdbench:latest
# copy the datasets into the benchmark dir, overwrite annotations.json/make_table.py/abed_conf.py
ADD datasets /TCPDBench/datasets
COPY annotations.json /TCPDBench/analysis/annotations/
COPY make_table.py /TCPDBench/analysis/scripts/
COPY abed_conf.py /TCPDBench/
# Set the working directory
WORKDIR TCPDBench
```
2. Inside the Dockerfile directory:
   1. create a folder `datasets` and store your data in it in this [format](https://github.com/alan-turing-institute/TCPD/blob/master/datasets/bank/bank.json) (`n_dim` must be 1).
   2. create a file `annotations.json` and modify it like [this](https://github.com/alan-turing-institute/TCPD/blob/master/annotations.json) for your data.
   3. create files [abed_conf.py](abed_conf.py) and [make_table.py](analysis/scripts/make_table.py) and configure their `TODOs` according to your datasets.
3. Run the commands below in the Dockerfile directory to execute the experiments:
```shell
# build a container from your created Dockerfile
docker build -t tcpdexperiment .
# make results persist to host
mkdir ${HOME}/docker_results
docker volume create --driver local --opt type=none --opt device=${HOME}/docker_results --opt o=bind tcpdbench_vol
# reproduce all experiments (-np sets number of threads)
docker run -i -t -v tcpdbench_vol:/TCPDBench/docker_results tcpdexperiment /bin/bash -c "abed reload_tasks && abed status && mpiexec --allow-run-as-root -np 4 abed local && make results && cp -r /TCPDBench/abed_results /TCPDBench/docker_results && cp -r /TCPDBench/analysis/output /TCPDBench/docker_results"
```

## License
The code in this repository is licensed under the MIT license, unless 
otherwise specified. See the [LICENSE file](LICENSE) for further details. 
Reuse of the code in this repository is allowed, but should cite [our 
paper](https://arxiv.org/abs/2003.06222).
```bib
@article{vandenburg2020evaluation,
        title={An Evaluation of Change Point Detection Algorithms},
        author={{Van den Burg}, G. J. J. and Williams, C. K. I.},
        journal={arXiv preprint arXiv:2003.06222},
        year={2020}
}
```