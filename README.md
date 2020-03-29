# HackingLab : Reprogramming Deep Neural Networks


### HRS

ADV Model Inception with parameter hrs_defense=True. Use function **construct_hrs_model** (from keras_utils.py).
In **AdvModel.py**:
```
def make_image_model(self, model_name, model_indicator):
    inception = []
    if model_name == "inception_v3":
        if self.hrs_defense:
            blocks_definition = get_split("default", 'IMAGENET')
            inception = construct_hrs_model(dataset='IMAGENET', model_indicator=model_indicator,
                                            blocks_definition=blocks_definition)
```

construct_hrs_model use **get_split()** (from block_split_config.py). Return two generator function for the two block of InceptionV3.
```
if dataset == 'IMAGENET':
    # block definitions
    def block_0():
        return split_InceptionV3()[0]
    def block_1():
        return split_InceptionV3()[1]
    generate_blocks = [block_0, block_1]
    return generate_blocks
```

**train_hrs.py** Training the InceptionV3 HRS version. For for each**block**, for each **channels** save weights of channel in Model/DATASET_models. Training algorithm starting at line 41.
**TODO**: 1) retrieve Imagenet data and set [X_train, X_test, Y_train, Y_test] = get_data(IMAGENET)
          2) Define loss function and training settings for Imagenet training.

After training, **construct_hrs_model()** read weights from  Model/DATASET_models given a model model_indicator (e.g test_hrs[10][10] as model_indicator means a model with two blocks, each one with 10 channels)


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
Required parameters:  
* Include path to folder to save models to **(end with /)**  
* CLASS_IMAGES = amount of images per class  
* IMAGE_TYPE = mnist or squares

for optional paramters see --help

```
python adv_model_script_tf.py /path/save/models/ CLASS_IMAGES IMAGE_TYPE
```

or to run with slurm

```
sbatch jobscript_opt_tf.sbatch /path/save/models/ CLASS_IMAGES IMAGE_TYPE
```
