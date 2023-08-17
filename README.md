# MindSet
![](https://storage.googleapis.com/kagglesdsdata/datasets/3633702/6315113/low_level_vision/ebbinghaus_illusion/scrambled_circles/0.07255_27.png?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=databundle-worker-v2%40kaggle-161607.iam.gserviceaccount.com%2F20230816%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20230816T215807Z&X-Goog-Expires=345600&X-Goog-SignedHeaders=host&X-Goog-Signature=4138d24bb641b28f4183f346a8229978d1d12832af160f800f02f59ddfa626bc4da41e39194ddba2b249ccabf26b053d621bf2eb18830d059bf868e751dfa914978912f408ffb6fa5ccabc7e01cdb08abdb13016f6978d4e22a7d98c07d2e41cfec74a21995e84a1abff55ca279fef564bcc01a2edd2cffc331da103871a2a289a4cf4bbcc2e021799199e2feadd3a4dfa474b2f96137ceb107698df665d6e6b3af2a5cd9335d5c9edc568b0b2706d82c6b78f37d2b15a8cdd7416384ed992b30099a0d4fadf5c40be7a30dad2d9c5de8caba080595ceb75b41af67c1b15be1abc8f9c6d35a3c9f0e0278dcfa659cc962384a4b5e753d24a340fd7ea4b3eee97) ![](https://storage.googleapis.com/kagglesdsdata/datasets/3633702/6315113/gestalt/texturized_linedrawings_chars/Airplane/1.png?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=databundle-worker-v2%40kaggle-161607.iam.gserviceaccount.com%2F20230816%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20230816T220054Z&X-Goog-Expires=345600&X-Goog-SignedHeaders=host&X-Goog-Signature=4830942d84a0035079e51ef89882c47f0def2b4188334f99afafafcaf024da11b39f704a104148f51825373dd441c8d4b458e48032c12e1d72b922a4705af5bac13d0a52333ba9bc1f169e770e02432cff076fa9ddb77b350f3cb0efd2d694ff9ad68be5333861c12b9e937ee8ac7e5024f5bc3ce7dca6a397360faf495e9c9347ad84b0f3542cf818fc3b5d4b861ff3dbd3dab938bf282ed71a84503a525f92012458e4fde614bb80e3db9302755e2ddab66f501f9b708e5ce2b4b4fd702038ed5afeef1d72552643c2cd2e04ca85cb043e523a8625aa2374e7db49574dc607b52f8ff9a1f9f79aa88d0bb5ac1b2489937999fed8bfaf92829a3f2cd8994b4d) ![](https://storage.googleapis.com/kagglesdsdata/datasets/3633702/6315113/low_level_vision/contour_completion/notched/c_15.png?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=databundle-worker-v2%40kaggle-161607.iam.gserviceaccount.com%2F20230816%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20230816T220121Z&X-Goog-Expires=345600&X-Goog-SignedHeaders=host&X-Goog-Signature=276cb6f6cdeb64fd36abcbf443a9eeb0360c8b85bb4f66030983104bd838fd74a5cf7d7110b272e504189dad8922708b146a6f861da0d286436937c0922e8f2259be8d97b766dacc2281179420063e2152063b4f11efc70766a2e24a3f3ba2c91401e73391ca9057f60a2c6cbc7aa63251c980d3b147c1bfe77023aa6bf2f9d092406a41cdfe65f1465b99044aa31be90ef9026c9331a8ea7e9f9b827d98d956eb0e6e17ac2f4731ecb28902fa3e5af11f8956dee642db211206115807c5cc66f835fadaf0b231b13b41631eea420b2c33c219c572f24c11ae5dd1d6a266b03addc6da464d15b54584a5b0086d8f95ccb36da91fc876b17cb35146979714b462)



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
["src/datasets_generation/gestalt/un_crowding/generate_dataset.py"]
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
