# Turing Change Point Detection Benchmark
Note that work based on either the dataset or this benchmark should cite that paper:
```bib
@article{vandenburg2020evaluation,
        title={An Evaluation of Change Point Detection Algorithms},
        author={{Van den Burg}, G. J. J. and Williams, C. K. I.},
        journal={arXiv preprint arXiv:2003.06222},
        year={2020}
}
```

## Running the experiments with Docker
1. Download the Dockerfile from this repository into a directory
2. Inside this directory, create a folder `datasets`, where your data is stored
3. create files `annotations.json`, `abed_conf.py` and `make_table.py` and configure them according to your datasets
4. Run the commands below to execute the experiments
```shell
docker build -t tcpdbench .
# make results persist to host
mkdir docker_results
docker volume create --driver local --opt type=none --opt device=./docker_results --opt o=bind tcpdbench_vol
# reproduce all experiments (-np sets number of threads)
docker run -i -t -v tcpdbench_vol:/TCPDBench/docker_results tcpdbench /bin/bash -c "mv abed_results old_abed_results && mkdir abed_results && abed reload_tasks && abed status && make venvs && mpiexec --allow-run-as-root -np 4 abed local && make results && cp -r /TCPDBench/abed_results /TCPDBench/docker_results && && cp -r /TCPDBench/analysis/output /TCPDBench/docker_results"
```

## Extending the Benchmark
### Adding a new method
#### Python
When adding a method in Python, you can start with the 
[cpdbench_zero.py](./execs/python/cpdbench_zero.py) file as a template, as 
this contains most of the boilerplate code. A script should take command line 
arguments where ``-i/--input`` marks the path to a dataset file and optionally 
can take further command line arguments for hyperparameter settings. 
Specifying these items from the command line facilitates reproducibility.

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

Finally, add your method to the ``DATASETS`` field in the ``abed_conf.py`` 
file. Proceed with running the experiments as described above.

## License
The code in this repository is licensed under the MIT license, unless 
otherwise specified. See the [LICENSE file](LICENSE) for further details. 
Reuse of the code in this repository is allowed, but should cite [our 
paper](https://arxiv.org/abs/2003.06222).
