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
    PasteOnCanvas,
)
from src.utils.compute_distance.activation_recorder import (
    RecordDistanceAcrossFolders,
    RecordDistanceImgBaseVsFolder,
)
import inspect


def compute_distance(
    input_paths=None,
    options=None,
    saving_folders=None,
    transformation=None,
    folder_vs_folder=None,
):
    with open(os.path.dirname(__file__) + "/default_distance_config.toml", "r") as f:
        toml_config = toml.load(f)
        toml_config["transformation"]["affine_transf_background"] = tuple(
            toml_config["transformation"]["affine_transf_background"]
        )
    # update the toml_config file based on the input args to this function
    local_vars = locals()
    update_dict(
        toml_config,
        {
            i: local_vars[i] if local_vars[i] else {}
            for i in inspect.getfullargspec(compute_distance)[0]
        },
    )

    network, norm_values, resize_value = GrabNet.get_net(
        toml_config["options"]["network_name"],
        imagenet_pt=True
        if toml_config["options"]["pretraining"] == "ImageNet"
        else False,
    )
    torch.cuda.set_device(
        toml_config["options"]["gpu_num"]
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
        x
        for x in [
            (
                PasteOnCanvas(
                    toml_config["transformation"]["canvas_to_image_ratio"],
                    toml_config["transformation"]["affine_transf_background"],
                )
                if toml_config["transformation"]["copy_on_bigger_canvas"]
                else None
            ),
            transforms.Resize(resize_value),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(norm_values["mean"], norm_values["std"]),
        ]
        if x is not None
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
            match_mode=toml_config["folder_vs_folder"]["match_mode"],
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