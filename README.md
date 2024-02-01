# MindSet
![](https://i.ibb.co/pvTVHKw/0-05254-67.png)     ![](https://i.ibb.co/4SvMvCt/28.png)![](https://i.ibb.co/9N4YVxF/c-0.png)


### TL;DR: Just gimme the datasets!
**[MindSet Large on Kaggle](https://www.kaggle.com/datasets/valerio1988/mindset)** (~1.3 GB)


**[MindSet Lite on Kaggle](https://www.kaggle.com/datasets/valerio1988/mindset-lite)**  (59 MB)


## Overview
The MindSet datasets are designed to facilitate the testing of DNNs against controlled experiments in psychology. MindSet will focus on a range of low-, middle-, and high-level visual findings that provide important constraints for theory. It also provides materials for DNN testing and demonstrates how to evaluate a DNN for each experiment using DNNs pretrained on ImageNet.

> **Note:** This README is under active development and will be updated regularly.


[Generalisation in Mind and Machine, Bristol University, UK](https://mindandmachine.blogs.bristol.ac.uk/) 

## Datasets

MindSet datasets are divided into three categories: `low_mid_level_vision`, `visual_illusions`, and `shape_and_object_recognition`. Each of this category contains many datasets. You can explore the datasets in Kaggle without downloading them. 
Every dataset is further structured into subfolders (conditions), which are organized based on the dataset's specific characteristics. At the root of each dataset, there's an `annotation.csv` file. This file lists the paths to individual images (starting from the dataset folder) along with their associated parameters. Such organization enables users to use the datasets either exploting their folder structure (e.g. through PyTorch's  ImageFolder) or by directly referencing the annotation file. In our provided Decoder, Classification and Similarity Judgment methods we always use the annotation.csv approach.

### Ready-To-Download Version

MindSet is model-agnostic and offers flexibility in the way each dataset is employed. Depending on the testing method, you may need a few samples or several thousand images. To cater to these needs, we provide two variants of the dataset on Kaggle:

- [Large Version](https://www.kaggle.com/datasets/valerio1988/mindset) with ~5000 samples for each condition.
- [Lite Version](https://www.kaggle.com/datasets/valerio1988/mindset-lite) with ~100 samples for each condition.


Both versions of the MindSet dataset are structured into folders, each containing a specific dataset. Due to Kaggle's current limitations, it's not possible to download these folders individually. Hence, if you need access to a specific dataset, you'll have to download the entire MindSet version. Alternatively, you can generate the desired dataset on your own following the provided guidelines in the next section.

Similarly, if your research or project requires datasets with more than the provided 5000 samples, you can regenerate the datasets with a specific sample size. 

### Generate datasets from scratch
We provide an intuitive interface to generate each dataset from scratch, allowing users to modify various parameters. This is done through a `toml` configuration file, a simple text file that specifies what datasets should be generated, and what parameters to use for each one of them. The `toml` file used to generate the lite and the full version uploaded to Kaggle are provided in the root folder: [`generate_all_datasets.toml`](generate_all_datasets.toml) and [`generate_all_datasets_lite.toml`](generate_all_datasets_lite.toml).
The file contains a series of config options for each dataset. For example, the dataset `un_crowding` in the category `low_mid_level_vision` is specified as: 
```toml
["low_mid_level_vision/un_crowding"]
# The size of the canvas. If called through command line, a string in the format NxM eg `224x224`.
canvas_size = [ 224, 224,]
# Specify the background color. Could be a list of RGB values, or `rnd-uniform` for a random (but uniform) color. If called from command line, the RGB value must be a string in the form R_G_B
background_color = [ 0, 0, 0,]
# Specify whether we want to enable antialiasing
antialiasing = false
# What to do if the dataset folder is already present? Choose between [overwrite], [skip]
behaviour_if_present = "overwrite"
# The number of samples for each vernier type (left/right orientation) and condition. The vernier is places inside a flanker.
num_samples_vernier_inside = 5000
# The number of samples for each vernier type (left/right orientation) and condition. The vernier is placed outside of the flankers
num_samples_vernier_outside = 5000
# Specify whether the size of the shapes will vary across samples
random_size = true
# The folder containing the data. It will be created if doesn't exist. The default will match the folder structure of the generation script 
output_folder = "data/low_mid_level_vision/un_crowding"
file = "src/generate_datasets/low_mid_level_vision/un_crowding/generate_dataset.py"
```

To regenerate datasets:

1. We suggest to not modify the original toml files but duplicate them: create a file  `my_datasets.toml`.
2. Copy over from `generate_all_datasets.toml` only the config options for the datasets you want to generate.
3. Adjust parameters in the config as needed. 
4. From the root directory, execute `python -m src.generate_datasets -tomlf my_datasets.toml`.

The generated dataset will be saved in the `output_folder` specified in the toml file.

## Generate the same dataset multiple times from the same file
If you need to generate the same dataset multiple times, each with different configurations or parameters, you can include multiple configurations for the same dataset within a single TOML file. However, TOML requires each table (denoted by names within [square brackets]) to have a unique name. To accomplish this, you can use different suffixes or identifiers for each configuration, like so:
```TOML
["low_mid_level_vision/un_crowding.TRAIN"]
...
output_folder = 'blabla/train'

["low_mid_level_vision/un_crowding.EVAL"]
output_folder = 'blabla/eval'
```
In this example, `TRAIN` and `EVAL` are distinct identifiers that allow you to define different settings for the same dataset under the `low_mid_level_vision/un_crowding` category. Ensure that the main name remains consistent, as it is used to locate the corresponding dataset generation file in the `src/generate_datasets` folder.


# Testing Methods
Although we encourage researchers to use MindSet datasets in a variety of different experimental setups to compare DNNs to humans, we provide the resources to perform a set of basic comparisons between humans and DNNs outputs.
The three methods employed are: 


- **[`Similarity Judgment`](src/utils/similarity_judgment/README.md)**: Compute a distance metric (e.g. `euclidean distance`) between the internal activation of a DNN across different stimulus set. Compare the results with human similarity judgments. In `utils/similarity_judgment`. Works with a wide variety of DNNs. 
- **[`Decoder Method`](src/utils/decoder/README.md)**: Attach simple linear decoders at several processing stages of a ResNet152 network pretrained on ImageNet. The idea is that the researcher trains the decoders on a task (either regression or classification), and then tested on some target condition such an illusory configuration. In `utils/decoder`. 
- **[`ImageNet Classification`](src/utils/imagenet_classification/README.md)**: Test DNNs on unfamiliar data, such as texturized images. in `utils/imagenet_classification`.

Each dataset has a suggested method it could be evaluated on. We provide examples for each method in the corresponding folder.

<!-- - [Similarity Judgment](https://github.com/ValerioB88/mind-set/tree/master/src/utils/similarity_judgment)
- [Decoder Approach](https://github.com/ValerioB88/mind-set/tree/master/src/utils/decoder) -->

**Note:** Always set the working directory to the project root (`MindSet`). To manage module dependencies, run the script as a module, e.g., `python -m src.generate_datasets --toml_file generate_all_datasets_lite.toml`.


### Publications 
- [Conference Paper](https://psyarxiv.com/cneyp/): **Introducing the MindSet benchmark for comparing DNNs to human vision**, [Conference on Cognitive Computational Neuroscience, Oxford, UK 2013](https://2023.ccneuro.org/view_paper.php?PaperNum=1127)
_Valerio Biscione, Don Yin, Gaurav Malhotra, Marin Dujmović, Milton Montero, Guillermo Puebla, Federico Adolfi, Christian Tsvetkov, Benjamin Evans, Jeffrey Bowers_

## Supported Operating Systems
The scripts and functionalities have been tested and are confirmed to work on *macOS 13 Ventura*, *Windows 11*, and *Ubuntu 20.04*.

<!-- ## Similarity Judgement
[![Demo for Similarity Judgement](assets/similarity_judgement.png)](https://youtu.be/a7k5viGmxnk)
 -->

