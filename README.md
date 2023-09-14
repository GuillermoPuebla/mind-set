# MindSet
![](https://i.ibb.co/pvTVHKw/0-05254-67.png)     ![](https://i.ibb.co/4SvMvCt/28.png)![](https://i.ibb.co/9N4YVxF/c-0.png)


### TL;DR: Just gimme the datasets!
**[MindSet Large on Kaggle](https://www.kaggle.com/datasets/valerio1988/mindset)** (~1.3 GB)


**[MindSet Lite on Kaggle](https://www.kaggle.com/datasets/valerio1988/mindset-lite)**  (59 MB)


## Overview
The MindSet datasets are designed to facilitate the testing of DNNs against controlled experiments in psychology. MindSet will focus on a range of low-, middle-, and high-level visual findings that provide important constraints for theory. It also provides materials for DNN testing and demonstrates how to evaluate a DNN for each experiment using a pretrained ResNet152 on ImageNet.

> **Note:** This README is under active development and will be updated regularly.


<!-- [Generalisation in Mind and Machine, Bristol University, UK](https://mindandmachine.blogs.bristol.ac.uk/) -->

## Datasets

MindSet datasets are divided into four categories: `coding_of_shapes`, `gestalt`, `high_level_vision`, and `low_level_vision`. These categories are subject to future modifications. Each of this categor contains a dataset. You can explore the datasets in Kaggle without downloading them. 
Every dataset is further structured into subfolders, which are organized based on the dataset's specific characteristics. At the root of each dataset, there's an annotation.csv file. This file lists the paths to individual images (starting from the dataset folder) along with their associated parameters. Such organization enables users to harness PyTorch's Dataset objects like ImageFolder. Alternatively, users can design their experiments by directly referencing the annotation file.

### Ready-To-Download Version

MindSet is model-agnostic and offers flexibility in the way each dataset is employed. Depending on the testing method, you may need a few samples or several thousand images. To cater to these needs, we provide two variants of the dataset on Kaggle:

- [Large Version](https://www.kaggle.com/datasets/valerio1988/mindset) with ~5000 samples for each condition.
- [Lite Version](https://www.kaggle.com/datasets/valerio1988/mindset-lite) with ~100 samples for each condition.


Both versions of the MindSet dataset are structured into folders, each containing a specific dataset. Due to Kaggle's current limitations, it's not possible to download these folders individually. Hence, if you need access to a specific dataset, you'll have to download the entire MindSet version. Alternatively, you can generate the desired dataset on your own following the provided guidelines in the next section.

Similarly, if your research or project requires datasets with more than the provided 5000 samples, you can regenerate the datasets with a specific sample size. 

### Generate datasets from scratch
We provide an intuitive interface to generate each dataset from scratch, allowing users to modify various parameters. This is done through a `toml` configuration file, a simple text file that specifies what datasets should be generated, and what parameters to use for each one of them. The `toml` file used to generate the lite and the full version uploaded to Kaggle are provided in the root folder: `generate_all_datasets.toml` and `generate_all_datasets_lite.toml`.
The file contains a series of config options for each dataset. For example, the dataset `un_crowding` in the category `gestalt` is specified as: 
```toml
["src/generate_datasets/gestalt/un_crowding/generate_dataset.py"]
canvas_size = [ 224, 224,]
background_color = [ 0, 0, 0,]
antialiasing = false
regenerate = true
num_samples_vernier_inside = 5000
num_samples_vernier_outside = 5000
random_size = true
output_folder = "datasets/gestalt/un_crowding"
```

To regenerate datasets:

1. Duplicate the `toml` file, e.g., `my_subset.toml`.
2. Retain only the datasets you want and remove the rest.
3. Adjust parameters as needed. Comprehensive documentation on these parameters will be available shortly.
4. From the root directory, execute `python -m src.generate_datasets --toml_file my_subset.toml`.

The generated dataset will be saved in the specified `output_folder`.


 


# Testing Methods
Although we encourage researchers to use MindSet datasets in a variety of different experimental setups to compare DNNs to humans, we provide the resources to perform a set of basic comparisons between humans and DNNs outputs.
The three methods employed are: 


- **Similarity Judgment**: Compute a distance metric (e.g. `euclidean distance`) between the internal activation of a DNN across different stimulus set. Compare the results with human similarity judgments. In `utils/similarity_judgment`. Works with a wide variety of DNNs. 
- **Decoder Method**: Attach simple linear decoders at several processing stages of a ResNet152 network pretrained on ImageNet. The Decoders are trained on a certain task, and then tested on some target condition such an illusory configuration. In `utils/decoder`. 
- **Classification of Out-of-Distribution Samples**: Test DNNs on unfamiliar data, such as texturized images. Not yet implemented.

You can refer to the examples in each folder to get started. We will soon provide a more detailed documentations about these three methods. 

<!-- - [Similarity Judgment](https://github.com/ValerioB88/mind-set/tree/master/src/utils/similarity_judgment)
- [Decoder Approach](https://github.com/ValerioB88/mind-set/tree/master/src/utils/decoder) -->

**Note:** Always set the working directory to the project root (`MindSet`). To manage module dependencies, run the script as a module, e.g., `python -m src.generate_datasets --toml_file generate_all_datasets_lite.toml`.


### Publications 
- [Conference Paper](https://psyarxiv.com/cneyp/): **Introducing the MindSet benchmark for comparing DNNs to human vision**, [Conference on Cognitive Computational Neuroscience, Oxford, UK 2013](https://2023.ccneuro.org/view_paper.php?PaperNum=1127)
_Valerio Biscione, Don Yin, Gaurav Malhotra, Marin Dujmović, Milton Montero, Guillermo Puebla, Federico Adolfi, Christian Tsvetkov, Benjamin Evans, Jeffrey Bowers_

## Supported Operating Systems
The scripts and functionalities have been tested and are confirmed to work on *macOS 13 Ventura*, *Windows 11*, and *Ubuntu 20.04*.