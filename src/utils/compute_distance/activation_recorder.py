import glob
import os

import pandas as pd
import torch
import sty
import numpy as np
from typing import List

import PIL.Image as Image
from matplotlib import pyplot as plt
from torchvision.transforms import InterpolationMode, transforms
from tqdm import tqdm

from src.utils.compute_distance.misc import my_affine, get_new_affine_values, save_figs
from src.utils.misc import conver_tensor_to_plot
from src.utils.net_utils import make_cuda
from copy import deepcopy


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
    def __init__(self, distance_metric, *args, **kwargs):
        assert distance_metric in [
            "euclidean",
            "cossim",
        ], f"distance_metric must be one of ['euclidean', 'cossim'], instead is {distance_metric}"
        self.distance_metric = distance_metric
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
                distance[name] = []
                if self.distance_metric == "cossim":
                    distance[name].append(
                        torch.nn.CosineSimilarity(dim=0)(
                            first_image_act[name], second_image_act[name]
                        ).item()
                    )
                if self.distance_metric == "euclidean":
                    distance[name].append(
                        torch.norm(
                            (first_image_act[name] - second_image_act[name])
                        ).item()
                    )
        return distance


class RecordDistanceAcrossFolders(RecordDistance):
    def compute_random_set(
        self,
        folder,
        transform,
        matching_transform=False,
        fill_bk=None,
        affine_transf="",
        N=5,
        path_save_fig=None,
        base_name="base",
    ):
        norm = [i for i in transform.transforms if isinstance(i, transforms.Normalize)][
            0
        ]
        save_num_image_sets = 5
        all_files = glob.glob(folder + "/**")
        levels = [
            os.path.basename(i) for i in glob.glob(folder + "/**") if os.path.isdir(i)
        ]


        sets = [
            np.unique(
                [
                    os.path.splitext(os.path.basename(i))[0]
                    for i in glob.glob(folder + f"/{l}/*")
                ]
            )
            for l in levels
        ]
        assert np.all(
            [len(sets[ix]) == len(sets[ix - 1]) for ix in range(1, len(sets))]
        ), "Length for one of the folder doesn't match other folder in the dataset"
        assert np.all(
            [np.all(sets[ix] == sets[ix - 1]) for ix in range(1, len(sets))]
        ), "All names in all folders in the dataset needs to match. Some name didn't match"
        sets = sets[0]

        df = pd.DataFrame([])
        save_sets = []
        for s in tqdm(sets):
            plt.close("all")
            for a in levels:
                save_fig = True
                save_sets = []
                for n in range(N):
                    im_0 = Image.open(f"{base_name}/{s}.png").convert("RGB")
                    im_i = Image.open(folder + f"/{a}/{s}.png").convert("RGB")
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
                    df_row = {"set": s, "level": a, "n": n}
                    cs = self.compute_distance_pair(
                        images[0], images[1]
                    )  # , path_fig='')
                    df_row.update(cs)
                    df = pd.concat([df, pd.DataFrame.from_dict(df_row)])

                    if save_fig:
                        save_sets.append(
                            [
                                conver_tensor_to_plot(i, norm.mean, norm.std)
                                for i in images
                            ]
                        )
                        if len(save_sets) == min([save_num_image_sets, N]):
                            save_figs(
                                path_save_fig + f"{s}_{a}",
                                save_sets,
                                extra_info=affine_transf,
                            )
                            save_fig = False
                            save_sets = []
        all_layers = list(cs.keys())
        return df, all_layers


class RecordDistanceImgBaseVsFolder(RecordDistance):
    def compute_random_set(
        self,
        folder,
        transform,
        matching_transform=False,
        fill_bk=None,
        affine_transf="",
        N=5,
        path_save_fig=None,
        base_name="base.png",
    ):
        norm = [i for i in transform.transforms if isinstance(i, transforms.Normalize)][
            0
        ]
        save_num_image_sets = 5
        compare_images = glob.glob(folder + "/**")

        df = pd.DataFrame([])
        for s in tqdm(compare_images):
            save_sets = []
            plt.close("all")
            save_fig = True
            for n in range(N):
                im_0 = Image.open(s).convert("RGB")
                im_i = Image.open(base_name).convert("RGB")
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
                df_row = {"compare_img": os.path.basename(s), "n": n}
                cs = self.compute_distance_pair(images[0], images[1])  # , path_fig='')
                df_row.update(cs)
                df = pd.concat([df, pd.DataFrame.from_dict(df_row)])

                if save_fig:
                    save_sets.append(
                        [conver_tensor_to_plot(i, norm.mean, norm.std) for i in images]
                    )
                    if len(save_sets) == min([save_num_image_sets, N]):
                        save_figs(
                            path_save_fig + f"{os.path.basename(s)}",
                            save_sets,
                            extra_info=affine_transf,
                        )
                        save_fig = False
                        save_sets = []
        all_layers = list(cs.keys())
        return df, all_layers
