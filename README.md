# Turing Change Point Detection Benchmark

Change point detection focuses on accurately detecting moments of abrupt 
change in the behavior of a time series. While many methods for change point 
detection exist, past research has paid little attention to the evaluation of 
existing algorithms on real-world data. This work introduces a benchmark study 
and a dataset that are explicitly designed for the evaluation of change point detection algorithms. 
We hope that our work becomes a proving ground for the comparison and 
development of change point detection algorithms that work well in practice.

This repository contains the code necessary to evaluate and analyze a 
significant number of change point detection algorithms on the TCPD, and 
serves to reproduce the work in [Van den Burg and Williams 
(2020)](https://arxiv.org/abs/2003.06222). Note that work based on either the 
dataset or this benchmark should cite that paper:

```bib
@article{vandenburg2020evaluation,
        title={An Evaluation of Change Point Detection Algorithms},
        author={{Van den Burg}, G. J. J. and Williams, C. K. I.},
        journal={arXiv preprint arXiv:2003.06222},
        year={2020}
}
```

For the experiments we've used the [abed](https://github.com/GjjvdBurg/abed) 
command line program, which makes it easy to organize and run the experiments. 
This means that all experiments are defined through the 
[abed_conf.py](abed_conf.py) file. In particular, the hyperparameters and the 
command line arguments to all methods are defined in that file. Next, all 
methods are called as command line scripts and they are defined in the 
[execs](execs) directory. The raw results from the experiments are collected 
in JSON files and placed in the [abed_results](abed_results) directory, 
organized by dataset and method. Finally, we use 
[Make](https://www.gnu.org/software/make/) to coordinate our analysis scripts: 
first we generate [summary files](analysis/output/summaries) using 
[summarize.py](analysis/scripts/summarize.py), and then use these to generate 
all the tables and figures in the paper.

## Getting Started

This repository contains all the code to generate the results 
(tables/figures/constants) from the paper, as well as to reproduce the 
experiments entirely. You can either install the dependencies directly on your 
machine or use the provided Dockerfile (see below). If you don't use Docker, 
first clone this repository using:

```
$ git clone --recurse-submodules https://github.com/simontrapp/TCPDBench
```

### Running the experiments with Docker

If you like to use [Docker](https://www.docker.com/) to manage the environment 
and dependencies, you can do so easily with the provided Dockerfile. You can 
build the Docker image using:

```
# TODO: add the datasets/*.json and analysis/annotations/annotations.json, edit the DATASETS in abed_conf.py and modifiy dataset enum in analysis/scripts/make_table.py
$ docker build -t tcpdbench .
# make results persist to host
$ mkdir docker_results
$ docker volume create --driver local --opt type=none --opt device=./docker_results --opt o=bind tcpdbench_vol
# OPTION 1: reproduce figures
$ docker run -i -t -v tcpdbench_vol:/TCPDBench tcpdbench /bin/bash -c "make results"
# OPTION 2: reproduce all experiments (-np sets number of threads)
$ docker run -i -t -v tcpdbench_vol:/TCPDBench tcpdbench /bin/bash -c "mv abed_results old_abed_results && mkdir abed_results && abed reload_tasks && abed status && make venvs && mpiexec --allow-run-as-root -np 4 abed local && make results"
```

## Extending the Benchmark

### Adding a new method

To add a new method to the benchmark, you'll need to write a script in the 
``execs`` folder that takes a dataset file as input and computes the change 
point locations.  Currently the methods are organized by language (R and 
python), but you don't necessarily need to follow this structure when adding a 
new method. Please do check the existing code for inspiration though, as 
adding a new method is probably easiest when following the same structure.

Experiments are managed using the [abed](https://github.com/GjjvdBurg/abed) 
command line application. This facilitates running all the methods with all 
their hyperparameter settings on all datasets.

Note that currently the methods print the output file to stdout, so if you 
want to print from your script, use stderr.

#### Python

When adding a method in Python, you can start with the 
[cpdbench_zero.py](./execs/python/cpdbench_zero.py) file as a template, as 
this contains most of the boilerplate code. A script should take command line 
arguments where ``-i/--input`` marks the path to a dataset file and optionally 
can take further command line arguments for hyperparameter settings. 
Specifying these items from the command line facilitates reproducibility.

If you need to add a timeout to your method, take a look at the 
[BOCPDMS](./execs/python/cpdbench_bocpdms.py) example.

#### R

Adding a method implemented in R to the benchmark can be done similarly to how 
it is done for Python. Again, the input file path and the hyperparameters are 
specified by command line arguments, which are parsed using 
[argparse](https://cran.r-project.org/web/packages/argparse/index.html). For R 
scripts we use a number of utility functions in the 
[utils.R](./execs/R/utils.R) file. To reliably load this file you can use the 
``load.utils()`` function available in all R scripts.

#### Adding the method to the experimental configuration

When you've written the command line script to run your method and verified 
that it works correctly, it's time to add it to the experiment configuration. 
For this, we'll have to edit the [abed_conf.py](./abed_conf.py) file.

1. To add your method, located the ``METHODS`` list in the configuration file 
   and add an entry ``best_<yourmethod>`` and ``default_<yourmethod>``, 
   replacing ``<yourmethod>`` with the name of your method (without spaces or 
   underscores).
2. Next, add the method to the ``PARAMS`` dictionary. This is where you 
   specify all the hyperparameters that your method takes (for the ``best`` 
   experiment). The hyperparameters are specified with a name and a list of 
   values to explore (see the current configuration for examples). For the 
   default experiment, add an entry ``"default_<yourmethod>" : {"no_param": 
   [0]}``. This ensures it will be run without any parameters.
3. Finally, add the command that needs to be executed to run your method to 
   the ``COMMANDS`` dictionary. You'll need an entry for ``best_<yourmethod>`` 
   and for ``default_<yourmethod>``. Please use the existing entries as 
   examples. Methods implemented in R are run with Rscript. The ``{execdir}``, 
   ``{datadir}``, and ``{dataset}`` values will be filled in by abed based on 
   the other settings. Use curly braces to specify hyperparameters, matching 
   the names of the fields in the ``PARAMS`` dictionary.


#### Dependencies

If your method needs external R or Python packages to operate, you can add 
them to the respective dependency lists.

* For R, simply add the package name to the [Rpackages.txt](./Rpackages.txt) 
  file. Next, run ``make clean_R_venv`` and ``make R_venv`` to add the package 
  to the R virtual environment. It is recommended to be specific in the 
  version of the package you want to use in the ``Rpackages.txt`` file, for 
  future reference and reproducibility.
* For Python, individual methods use individual virtual environments, as can 
  be seen from the bocpdms and rbocpdms examples. These virtual environments 
  need to be activated in the ``COMMANDS`` section of the ``abed_conf.py`` 
  file. Setting up these environments is done through the Makefile. Simply add 
  a ``requirements.txt`` file in your package similarly to what is done for 
  bocpdms and rbocpdms, copy and edit the corresponding lines in the Makefile, 
  and run ``make venv_<yourmethod>`` to build the virtual environment.


#### Running experiments

When you've added the method and set up the environment, run

```
$ abed reload_tasks
```

to have abed generate the new tasks for your method (see above under [Getting 
Started](#getting-started)). Note that abed automatically does a Git commit 
when you do this, so you may want to switch to a separate branch. You can see 
the tasks that abed has generated (and thus the command that will be executed) 
using the command:

```
$ abed explain_tbd_tasks
```

If you're satisfied with the commands, you can run the experiments using:

```
$ mpiexec -np 4 abed local
```

You can subsequently use the Makefile to generate updated figures and tables 
with your method or dataset.

### Adding a new dataset

To add a new dataset to the benchmark you'll need both a dataset file (in JSON 
format) and annotations (for evaluation). More information on how the datasets 
are constructed can be found in the 
[TCPD](https://github.com/alan-turing-institute/TCPD) repository, which also 
includes a schema file. A high-level overview is as follows:

* Each dataset has a short name in the ``name`` field and a longer more 
  descriptive name in the ``longname`` field. The ``name`` field must be 
  unique.
* The number of observations and dimensions is defined in the ``n_obs`` and 
  ``n_dim`` fields.
* The time axis is defined in the ``time`` field. This has at least an 
  ``index`` field to mark the indices of each data point. At the moment, these 
  indices need to be consecutive integers. This entry mainly exist for a 
  future scenario where we may want to consider non-consecutive time axes. If 
  the time axis can be mapped to a date or time, then a type and format of 
  this field can be specified (see e.g. the [nile 
  dataset](https://github.com/alan-turing-institute/TCPD/blob/master/datasets/nile/nile.json#L8), 
  which has year labels).
* The actual observations are specified in the ``series`` field. This is an 
  ordered list of JSON objects, one for each dimension. Every dimension has a 
  label, a data type, and a ``"raw"`` field with the actual observations. 
  Missing values in the time series can be marked with ``null`` (see e.g. 
  [uk_coal_employ](https://github.com/alan-turing-institute/TCPD/blob/master/datasets/uk_coal_employ/uk_coal_employ.json#L236) 
  for an example).

If you want to evaluate the methods in the benchmark on a new dataset, you may 
want to collect annotations for the dataset. These annotations can be 
collected in the [annotations.json](./analysis/annotations/annotations.json) 
file, which is an object that maps each dataset name to a map from the 
annotator ID to the marked change points. You can collect annotations using 
the [annotation tool](https://github.com/alan-turing-institute/annotatechange) 
created for this project.

Finally, add your method to the ``DATASETS`` field in the ``abed_conf.py`` 
file. Proceed with running the experiments as described above.

## License

The code in this repository is licensed under the MIT license, unless 
otherwise specified. See the [LICENSE file](LICENSE) for further details. 
Reuse of the code in this repository is allowed, but should cite [our 
paper](https://arxiv.org/abs/2003.06222).
