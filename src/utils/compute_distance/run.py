"""
This will compute the folder-to-folder version of distance.
Given a dataset folder `./data/xx/` and a base folder, it compares each sample in each folder with the base folder, e.g
    `./data/xx/base/0.png` vs `./data/xx/comp1/0.png`,
    `./data/xx/base/1.png` vs  `./data/xx/comp1/1.png`,
    .
    .
    .
    `./data/xx/base/0.png` vs `./data/xx/comp2/0.png`,
    .
    .
The number of samples in each folder must match.
Each comparison is done multiple time at different transformations
"""
import argparse

import toml
import torch
from src.utils.net_utils import GrabNet, prepare_network
from src.utils.misc import update_dict, pretty_print_dict
from sty import fg, rs
import pickle
import os
import pathlib
import torchvision.transforms as transforms
import torchvision
from src.utils.compute_distance.misc import (
    has_subfolders,
)
from src.utils.compute_distance.activation_recorder import (
    RecordDistanceAcrossFolders,
    RecordDistanceImgBaseVsFolder,
)


def compute_distance(input_paths, options, saving_folders, transformation):
    with open(os.path.dirname(__file__) + "/default_distance_config.toml", "r") as f:
        toml_config = toml.load(f)
        toml_config["transformation"]["affine_transf_background"] = tuple(
            toml_config["transformation"]["affine_transf_background"]
        )
    update_dict(
        toml_config,
        {
            "input_paths": input_paths if input_paths else {},
            "options": options if options else {},
            "saving_folders": saving_folders if saving_folders else {},
            "transformation": transformation if transformation else {},
        },
    )

    network, norm_values, resize_value = GrabNet.get_net(
        toml_config["options"]["network_name"],
        imagenet_pt=True
        if toml_config["options"]["pretraining"] == "ImageNet"
        else False,
    )
    torch.cuda.set_device(
        toml_config["training"]["gpu_num"]
    ) if torch.cuda.is_available() else None

    if has_subfolders(toml_config["input_paths"]["folder"]):
        assert os.path.isdir(
            toml_config["input_paths"]["base_name"]
        ), f"{toml_config['input_paths']['folder']} contains other folders, and so you are in folders vs folders mode. However, { toml_config['input_paths']['base_name']} should be a path but it's not!"
        toml_config["run_info"]["type"] = "folder_vs_folder"
    else:
        assert os.path.isfile(
            toml_config["input_paths"]["base_name"]
        ), f"Folder {toml_config['input_paths']['folder']} does not have subfolders, so you are in `image vs folder mode. However, base_name {toml_config['input_paths']['base_name']} is NOT a path to a file!"
        toml_config["run_info"]["type"] = "image_vs_folder"

    pathlib.Path(toml_config["saving_folders"]["result_folder"]).mkdir(
        parents=True, exist_ok=True
    )

    toml.dump(
        {
            **toml_config,
            "transformation": {
                **toml_config["transformation"],
                "affine_transf_background": list(
                    toml_config["transformation"]["affine_transf_background"]
                ),
            },
        },
        open(
            toml_config["saving_folders"]["result_folder"] + "/train_config.toml", "w"
        ),
    )
    pretty_print_dict(toml_config)

    prepare_network(
        network,
        toml_config["options"]["pretraining"],
        train=False,
    )

    transf_list = [
        transforms.Resize(resize_value),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(norm_values["mean"], norm_values["std"]),
    ]

    transform = torchvision.transforms.Compose(transf_list)

    debug_image_path = toml_config["saving_folders"]["result_folder"] + "/debug_img/"
    pathlib.Path(os.path.dirname(toml_config["saving_folders"]["result_folder"])).mkdir(
        parents=True, exist_ok=True
    )
    pathlib.Path(os.path.dirname(debug_image_path)).mkdir(parents=True, exist_ok=True)
    if toml_config["run_info"]["type"] == "folder_vs_folder":
        recorder = RecordDistanceAcrossFolders(
            distance_metric=toml_config["options"]["distance_metric"],
            net=network,
            use_cuda=False,
            only_save=toml_config["options"]["save_layers"],
        )
    elif toml_config["run_info"]["type"] == "image_vs_folder":
        recorder = RecordDistanceImgBaseVsFolder(
            distance_metric=toml_config["options"]["distance_metric"],
            net=network,
            use_cuda=torch.cuda.is_available(),
            only_save=toml_config["options"]["save_layers"],
        )

    distance_df, layers_names = recorder.compute_random_set(
        folder=toml_config["input_paths"]["folder"],
        transform=transform,
        matching_transform=toml_config["transformation"]["matching_transform"],
        fill_bk=toml_config["transformation"]["affine_transf_background"],
        affine_transf=toml_config["transformation"]["affine_transf_code"],
        N=toml_config["transformation"]["repetitions"],
        path_save_fig=debug_image_path,
        base_name=toml_config["input_paths"]["base_name"],
    )

    save_path = toml_config["saving_folders"]["result_folder"] + "/dataframe.pickle"
    print(fg.red + f"Saved in " + fg.green + f"{save_path}" + rs.fg)

    pickle.dump(
        {"layers_names": layers_names, "dataframe": distance_df}, open(save_path, "wb")
    )
    return distance_df, layers_names


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--toml_config_path",
        "-toml",
        default=f"{os.path.dirname(__file__)}/default_distance_config.toml",
    )
    args = parser.parse_known_args()[0]
    with open(args.toml_config_path, "r") as f:
        toml_config = toml.load(f)
    print(f"**** Selected {args.toml_config_path} ****")
    compute_distance(**toml_config)
