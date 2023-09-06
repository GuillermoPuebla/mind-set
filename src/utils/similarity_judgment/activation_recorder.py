import glob
import os
from pathlib import Path

import pandas as pd
import torch
import sty
import numpy as np
from typing import List

import PIL.Image as Image
from matplotlib import pyplot as plt
from torchvision.transforms import InterpolationMode, transforms
from tqdm import tqdm

from src.utils.similarity_judgment.misc import (
    my_affine,
    get_new_affine_values,
    save_figs,
)
from src.utils.misc import conditional_tqdm, conver_tensor_to_plot
from src.utils.net_utils import make_cuda
from copy import deepcopy
import csv


class RecordActivations:
    def __init__(
        self, net, use_cuda=None, only_save: List[str] = None, detach_tensors=True
    ):
        if only_save is None:
            self.only_save = ["Conv2d", "Linear"]
        else:
            self.only_save = only_save
        self.cuda = False
        if use_cuda is None:
            if torch.cuda.is_available():
                self.cuda = True
            else:
                self.cuda = False
        else:
            self.cuda = use_cuda
        self.net = net
        self.detach_tensors = detach_tensors
        self.activation = {}
        self.last_linear_layer = ""
        self.all_layers_names = []
        self.setup_network()

    def setup_network(self):
        self.was_train = self.net.training
        self.net.eval()  # a bit dangerous
        print(
            sty.fg.yellow + "Network put in eval mode in Record Activation" + sty.rs.fg
        )
        all_layers = self.group_all_layers()
        self.hook_lists = []
        for idx, i in enumerate(all_layers):
            name = "{}: {}".format(idx, str.split(str(i), "(")[0])
            if np.any([ii in name for ii in self.only_save]):
                ## Watch out: not all of these layers will be used. Some networks have conditional layers depending on training/eval mode. The best way to get the right layers is to check those that are returned in "activation"
                self.all_layers_names.append(name)
                self.hook_lists.append(
                    i.register_forward_hook(self.get_activation(name))
                )
        self.last_linear_layer = self.all_layers_names[-1]

    def get_activation(self, name):
        def hook(model, input, output):
            self.activation[name] = output.detach() if self.detach_tensors else output

        return hook

    def group_all_layers(self):
        all_layers = []

        def recursive_group(net):
            for layer in net.children():
                if not list(layer.children()):  # if leaf node, add it to list
                    all_layers.append(layer)
                else:
                    recursive_group(layer)

        recursive_group(self.net)
        return all_layers

    def remove_hooks(self):
        for h in self.hook_lists:
            h.remove()
        if self.was_train:
            self.net.train()


class RecordDistance(RecordActivations):
    def __init__(
        self,
        annotation_filepath,
        match_factors,
        factor_variable,
        filter_factor_level,
        reference_level,
        distance_metric,
        *args,
        **kwargs,
    ):
        assert distance_metric in [
            "euclidean",
            "cossim",
        ], f"distance_metric must be one of ['euclidean', 'cossim'], instead is {distance_metric}"
        self.filter_factor_level = filter_factor_level
        self.distance_metric = distance_metric
        self.annotation_filepath = annotation_filepath
        self.match_factors = match_factors
        self.factor_variable = factor_variable
        self.reference_level = reference_level
        super().__init__(*args, **kwargs)

    def compute_distance_pair(self, image0, image1):  # path_save_fig, stats):
        distance = {}

        self.net(make_cuda(image0.unsqueeze(0), torch.cuda.is_available()))
        first_image_act = {}
        # activation_image1 = deepcopy(self.activation)
        for name, features1 in self.activation.items():
            if not np.any([i in name for i in self.only_save]):
                continue
            first_image_act[name] = features1.flatten()

        self.net(make_cuda(image1.unsqueeze(0), torch.cuda.is_available()))
        # activation_image2 = deepcopy(self.activation)

        second_image_act = {}
        for name, features2 in self.activation.items():
            if not np.any([i in name for i in self.only_save]):
                continue
            second_image_act[name] = features2.flatten()
            if name not in distance:
                if self.distance_metric == "cossim":
                    distance[name].append(
                        torch.nn.CosineSimilarity(dim=0)(
                            first_image_act[name], second_image_act[name]
                        ).item()
                    )
                if self.distance_metric == "euclidean":
                    distance[name] = torch.norm(
                        (first_image_act[name] - second_image_act[name])
                    ).item()
        return distance

    def compute_from_annotation(
        self,
        transform,
        matching_transform=False,
        fill_bk=None,
        affine_transf="",
        transformed_repetition=5,
        path_save_fig=None,
    ):
        norm = [i for i in transform.transforms if isinstance(i, transforms.Normalize)][
            0
        ]
        df = pd.read_csv(self.annotation_filepath)

        mask = pd.Series([True] * len(df), index=df.index, dtype="bool")

        for col, val in self.filter_factor_level.items():
            mask = mask & (df[col] == val)
        df = df.loc[mask]

        if self.match_factors:
            matching_levels = df[self.match_factors].drop_duplicates().values.tolist()
        else:
            matching_levels = df.values.tolist()
            print(
                sty.fg.red
                + "No MATCHING LEVELS. Are you sure this is correct?"
                + sty.rs.fg
            )

        all_other_levels = [
            i for i in df[self.factor_variable].unique() if i != self.reference_level
        ]
        pbar = tqdm(all_other_levels, desc="comparison levels")
        df_rows = []
        for comparison_level in pbar:
            pbar.set_postfix(
                {
                    sty.fg.blue
                    + f"{self.factor_variable}"
                    + sty.rs.fg: f"{self.reference_level} vs {comparison_level}"
                },
                refresh=True,
            )
            pbar2 = tqdm(matching_levels, desc="matching samples", leave=False)

            for mm in pbar2:
                pbar2.set_postfix(
                    {
                        sty.fg.blue + f"{k}" + sty.rs.fg: v
                        for k, v in zip(self.match_factors, mm)
                    },
                    refresh=True,
                )
                mask = pd.Series([True] * len(df), index=df.index)

                for col, val in zip(self.match_factors, mm):
                    mask = mask & (df[col] == val)

                comparison_paths = np.random.permutation(
                    df[df[self.factor_variable] == comparison_level]
                    .loc[mask]["Path"]
                    .values
                )
                reference_paths = np.random.permutation(
                    df[df[self.factor_variable] == self.reference_level]
                    .loc[mask]["Path"]
                    .values
                )

                if len(comparison_paths) != len(reference_paths):
                    print(
                        sty.fg.red
                        + f"The number of images satisfying the requirements is not the same for level {comparison_level} and {self.reference_level}. Some images won't be processed"
                        + sty.rs.fg
                    )

                for sample_num, selected_paths in conditional_tqdm(
                    enumerate(zip(reference_paths, comparison_paths)),
                    len(reference_paths) > 1,
                    desc="nth matching sample",
                    leave=False,
                ):
                    reference_path, comp_path = (
                        Path(self.annotation_filepath).parent / i
                        for i in selected_paths
                    )
                    save_num_image_sets = 5

                    save_sets = []
                    save_fig = True

                    for transform_idx in conditional_tqdm(
                        range(transformed_repetition),
                        transformed_repetition > 1,
                        desc="transformation rep.",
                        leave=False,
                    ):
                        im_0 = Image.open(reference_path).convert("RGB")
                        im_i = Image.open(comp_path).convert("RGB")
                        af = (
                            [get_new_affine_values(affine_transf) for i in [im_0, im_i]]
                            if not matching_transform
                            else [get_new_affine_values(affine_transf)] * 2
                        )
                        images = [
                            my_affine(
                                im,
                                translate=af[idx]["tr"],
                                angle=af[idx]["rt"],
                                scale=af[idx]["sc"],
                                shear=af[idx]["sh"],
                                interpolation=InterpolationMode.NEAREST,
                                fill=fill_bk,
                            )
                            for idx, im in enumerate([im_0, im_i])
                        ]

                        images = [transform(i) for i in images]

                        layers_distances = self.compute_distance_pair(
                            images[0], images[1]
                        )
                        df_rows.append(
                            {
                                "ReferencePath": str(reference_path),
                                "ComparisonPath": str(comp_path),
                                "ReferenceLevel": self.reference_level,
                                "ComparisonLevel": comparison_level,
                                "MatchingLevels": mm,
                                **{f"{i}": j for i, j in zip(self.match_factors, mm)},
                                "TransformerRep": transform_idx,
                                **layers_distances,
                            }
                        )
                        if save_fig:
                            save_sets.append(
                                [
                                    conver_tensor_to_plot(i, norm.mean, norm.std)
                                    for i in images
                                ]
                            )
                            if (
                                len(save_sets)
                                == min([save_num_image_sets, transformed_repetition])
                                and path_save_fig
                            ):
                                save_figs(
                                    path_save_fig
                                    + f"[{self.reference_level}]_{comparison_level}_{'-'.join([str(i) for i in mm])}",
                                    save_sets,
                                    extra_info=affine_transf,
                                )

                                save_fig = False
                                save_sets = []
        result_df = pd.DataFrame(df_rows)

        all_layers = list(layers_distances.keys())
        return result_df, all_layers
