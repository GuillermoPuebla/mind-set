# # """
# # This will compute the base image vs folder distance.
# # Given a dataset folder `./data/xx/` and a base IMAGE, it compares the base image for each image in the target folder
# #     `./data/xx/base/0.png` vs `./data/xx/comp1/0.png`,
# #     `./data/xx/base/0.png` vs  `./data/xx/comp1/1.png`,
# #     .
# #     .
# #     .
# #     .
# # Each comparison is done multiple time at different transformations
# # """
# #
# # from torchvision.transforms.functional import InterpolationMode
# # import glob
# # import pandas as pd
# # from src.utils.net_utils import GrabNet, prepare_network
# # from src.utils.misc import conver_tensor_to_plot
# # import matplotlib.pyplot as plt
# # from sty import fg, rs
# # import pickle
# # import os
# # import pathlib
# # from tqdm import tqdm
# # import torchvision.transforms as transforms
# # import torchvision
# # import PIL.Image as Image
# # from src.utils.compute_distance.misc import (
# #     get_new_affine_values,
# #     my_affine,
# #     save_figs,
# #     get_distance_args,
# # )
# # from src.utils.compute_distance.activation_recorder import RecordDistance
# #
#
# # def compute_distance(config):
# #     config.model, norm_values, resize_value = GrabNet.get_net(
# #         config.network_name,
# #         imagenet_pt=True if config.pretraining == "ImageNet" else False,
# #     )
# #
# #     prepare_network(config.model, config, train=False)
# #
# #     transf_list = [
# #         transforms.Resize(resize_value),
# #         torchvision.transforms.ToTensor(),
# #         torchvision.transforms.Normalize(norm_values["mean"], norm_values["std"]),
# #     ]
# #
# #     transform = torchvision.transforms.Compose(transf_list)
# #
# #     debug_image_path = config.result_folder + "/debug_img/"
# #     pathlib.Path(os.path.dirname(config.result_folder)).mkdir(
# #         parents=True, exist_ok=True
# #     )
# #     pathlib.Path(os.path.dirname(debug_image_path)).mkdir(parents=True, exist_ok=True)
# #
#     recorder = RecordDistanceImgBaseVsFolder(
#         distance_metric=config.distance_metric,
#         net=config.model,
#         use_cuda=True,
#         only_save=config.save_layers,
#     )
# #     distance_df, layers_names = recorder.compute_random_set(
# #         folder=config.folder,
# #         transform=transform,
# #         matching_transform=config.matching_transform,
# #         fill_bk=config.affine_transf_background,
# #         affine_transf=config.affine_transf_code,
# #         N=config.repetitions,
# #         path_save_fig=debug_image_path,
# #         base_image=config.base_image,
# #     )
# #
# #     save_path = config.result_folder + "/dataframe.pickle"
# #     print(fg.red + f"Saved in " + fg.green + f"{save_path}" + rs.fg)
# #
# #     pickle.dump(
# #         {
# #             "layers_names": layers_names,
# #             "dataframe": distance_df,
# #             "folder": config.folder,
# #             "base_image": config.base_image,
# #         },
# #         open(save_path, "wb"),
# #     )
# #     return distance_df, layers_names
# #
# #
# # if __name__ == "__main__":
# #     import argparse
# #
# #     parser = argparse.ArgumentParser()
# #     parser.add_argument("--base_image")
# #     parser.add_argument("--folder")
# #     parser = get_distance_args(parser)
# #
# #     # config = parser.parse_args(['--base_image', './data/examples/closure/square.png', '--folder', './data/examples/closure/angles_rnd/', '--result_folder', './results/closure/square/segm15/full_vs_segm/', '--repetitions', '2'])  #, '--affine_transf_code', 't[-0.2, 0.2]s[0.5,0.9]r'])
# #
# #     config = parser.parse_known_args()[0]
# #     [
# #         print(fg.red + f"{i[0]}:" + fg.cyan + f" {i[1]}" + rs.fg)
# #         for i in config._get_kwargs()
# #     ]
# #     dataframe, layers_names = compute_distance(config)
