
## Distance Similarity
-----THIS DOCUMENTATION IS NO LONGER VALID AND NEEDS TO BE REWRITTEN TO ACCOUNT FOR THE TOML CONFIG SETUP WE USE NOW. HOWEVER, TO GET STARTED, USE THE EXAMPLES iIN utils/similarity_judgment/examples. THEY ARE UP TO DATE!  

[//]: # (All the relevant scripts are in `src/utils/compute_distance` and `src/utils`. Examples are in `src/utils/compute_distance/examples`.  )
You can compute the distance between activation with `cosine similarity` or `euclideaen distance`. Use the command line argument `--distance_metric cossim` or `--distance_metric euclidean`. The latter is the default. 

There are two ways to compute the activation distance. One is by comparing _one base image_ with a set of images contained in a folder (`image_vs_folder` method). The other one is to compare a set of images in a base folder with corresponding images in other folders (`folder_vs_folder` method). You will find an example for each one in the folder `src/utils/compute_distance/examples/`. 

For both cases, you use the script `src/utils/compute_distance/run.py`. The folder structure provided will determine the method used. 
Next is how to organize the dataset to use the scripts:


### Dataset: Base image vs set of images
* **Example dataset in `data/examples/closure`**
* **Example usage in `src/utils/compute_distance/examples/image_vs_folder.py`**

This is the simplest way of computing the distance. Simply specify the path of the base image (`base_name` argument) and the `folder` containing images you want to compare the base image with. An example of this dataset is `data/examples/closure/`, which contains the base image `square.png` and a folder with a bunch of other images (image names here don't matter). 


The meaning of the `--affine_transf_code` optional args is explained below. 

### Dataset: Folder vs Folder
* **Example dataset in `data/examples/NAPvsMP_standard`**
* **Example usage `src/utils/compute_distance/examples/folder_vs_folder.py`**

This method is more complex but allows for more interesting comparisons. 
Here you compute distance between images in a base folder (`base_name` argument) and corresponding images in other folders, all contained in a parent folder specified in the `folder` argument. An example is in `data/examples/NAPvsMPlines` folder. In the example, `data/examples/NAPvsMPlines/NS` is the base folder, and `data/examples/NAPvsMPlines` is the parent folder containing folders against which `NS` images will be compared (notice that the `base_folder` does not need to be a subfolder of `folder`).  

Images names are important in this mode: the names need to match across folders: `NAPvsMPlines/NS/1.png` will be compared with `NAPvsMPlines/S/1.png` and with `NAPvsMPlines/NSmore/1.png`. The image name doesn't need to be a number - but the name needs to match across folders.

However, there are cases in which you want to compare each image in the base folder with each other image in all the other folders. For example, you want to compare `NAPvsMPlines/NS/1.png` with `NAPvsMPlines/S/1.png` but also with `NAPvsMPlines/NS/2.png` and so on. To do that, set the option `match_mode` in the TOML file to `all` (default is `same_name`). In this mode, having matchin name doesn't matter and is not checked for.

(Note: by carefully arranging the folders and setting `match_more` to `all`, you could perfectly recreate the `base image vs set of images` mode. Thus, `base image vs set of images` mode is a special case of `folder vs folder` mode. We decide to keep them separated for simplicity). 

## Dataset: All vs All
* **Example dataset in TBD 
* **Example usage TBD 
This computes all paired comparisons for all iamges within one folder. In this case the user just needs to specify the `folder` arugment, leaving the `base_name` empty. 

## Optional Arguments
All default arguments are specified in `src/utils/compute_distance/default_distance_config.toml`. Here are some info about some of those options.

### Network_name
`--network_name`: Choose between `alexnet`, `vgg11`, `vgg16`, `vgg11bn`, `vgg16bn`, `vgg19bn`, `resnet18`, `resnet50`, `resnet152`, `inception_v3` , `densenet121`, `densenet201`, `googlenet`. Default is `resnet152`. 

### Affine_transf_code
You may want to augment the samples by using an affine transformation augmentation. Use the `--affine_transf_code` for that.
 `t`, `s` and `r` indicate that you want to apply translation, scale and rotation. If you do not indicate anything, default values will be used. For example you can pass `ts` to apply translation and scaling but no rotation. You can be more specific with the parameters, for example `t[-0.2,0.2]r[0,45]` will translate and rotate by the specified amount (translation parameters are fraction of total image size, rotation in degree). 
By default, the *same* transformation is applied across all pairs of a comparison. If that's not what you want, set the optional argument `--matching_transform` to `False`.

Another related optional argument is `--affine_transf_background`. The colour specified here (default is black: `(0, 0, 0)`) will "fill in" the area outside the transformed image (for example, when you apply a rotation to the image. Otherwise it will be white.

### Save_Layers
`--save_layers` Indicate what layer type you want to compute the distance metric on. In practice, we check whether the layer name contains any of the string indicated here. For example, you can specify `--set_layers MaxP ReLU Conv2d`. Default is `Conv2d Linear`.


## Output
Both `folder_vs_folder` and `image_vs_folder` will return a pandas dataframe, and a list of all layers which activation has been used for computing the distance metric. 
These scripts will save the dataframe as a pickle object in the `--result_folder`. They'll also write images in a `debug_img` folder (within the `--result_folder`)  showing samples of the pairs used for computing distance metric. Really useful to check that it all makes sense, expecially with respect to the affine transformations.

## Analysis
Both scripts will generate a pickle file containing a panda dataset that you can use for analysis. It is assumed that you will write your own analysis but, as an example of how you can use the generated file, you can take a look at the script `src/utils/compute_distance/examples/analysis_folder_vs_folder.py`

