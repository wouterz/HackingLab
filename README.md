# HackingLab : Reprogramming Deep Neural Networks

### Prerequisites

```
python 3.6
```

create data:

```
python3 ./make_squares.py
```

### Installing

Create virtual env and activate

```
python36 -m venv /path/to/new/virtual/environment
source /path/to/new/virtual/environment/bin/activate
```
Install dependencies
```
python -m pip install -r requirements.txt
```

### Running

Include path to folder to save models to **(end with /)**
```
python adv_model_script_tf.py /path/save/models/
```

or to run with slurm
```
sbatch jobscript_opt_tf.sbatch /path/save/models/
```