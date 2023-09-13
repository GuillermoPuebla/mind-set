import json
import os
import torch
import torch.backends.cudnn as cudnn

# from rich import print
from src.utils.callbacks import *
from src.utils.dataset_utils import get_dataloader
from src.utils.misc import pretty_print_dict, update_dict
from src.utils.net_utils import load_pretraining, make_cuda, GrabNet, ResNet152decoders
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from tqdm import tqdm
from pathlib import Path
import pandas
import toml
import inspect


def decoder_evaluate(
    task_type=None, gpu_num=None, eval=None, network=None, saving_folders=None, **kwargs
):
    with open(os.path.dirname(__file__) + "/default_decoder_config.toml", "r") as f:
        toml_config = toml.load(f)

    # update the toml_config file based on the input args to this function
    local_vars = locals()
    update_dict(
        toml_config,
        {i: local_vars[i] for i in inspect.getfullargspec(decoder_evaluate)[0]},
    )
    pretty_print_dict(toml_config, name="PARAMETERS")
    use_cuda = torch.cuda.is_available()
    torch.cuda.set_device(toml_config["gpu_num"]) if torch.cuda.is_available() else None

    test_loaders = [
        get_dataloader(
            toml_config["task_type"],
            ds_config=i,
            transf_config=toml_config["transformation"],
            batch_size=toml_config["network"]["batch_size"],
            return_path=True,
        )
        for i in toml_config["eval"]["datasets"]
    ]

    assert toml_config["network"]["architecture_name"] in [
        "resnet152_decoder",
        "resnet152_decoder_residual",
    ], f"Network.name needs to be either `resnet152_decoder` or `resnet152_decoder_residual`. You used {toml_config['network']['name']}"

    net, _, _ = GrabNet.get_net(
        toml_config["network"]["architecture_name"],
        imagenet_pt=True if toml_config["network"]["imagenet_pretrained"] else False,
        num_classes=toml_config["network"]["decoder_outputs"]
        if toml_config["task_type"] == "regression"
        else len(test_loaders[0].dataset.classes),
    )

    load_pretraining(
        net=net,
        optimizers=None,
        network_path=toml_config["network"]["load_path"],
        optimizers_path=None,
    )
    net.eval()

    cudnn.benchmark = True if use_cuda else False

    net.cuda() if use_cuda else None

    results_folder = pathlib.Path(toml_config["saving_folders"]["results_folder"])
    num_decoders = len(net.decoders)

    def evaluate_one_dataloader(dataloader):
        results_final = []
        (results_folder / dataloader.dataset.name).mkdir(parents=True, exist_ok=True)

        print(
            f"Evaluating Dataset "
            + sty.fg.green
            + f"{dataloader.dataset.name}"
            + sty.rs.fg
        )

        for _, data in enumerate(tqdm(dataloader, colour="yellow")):
            images, labels, path = data
            images = make_cuda(images, use_cuda)
            labels = make_cuda(labels, use_cuda)
            out_dec = net(images)
            for i in range(len(labels)):
                # if task_type == "classification":
                #     prediction = torch.argmax(out_dec[decoder_idx][i]).item()
                # else:
                #     try:
                #         prediction = out_dec[decoder_idx][i].item()
                #     except IndexError:
                #         prediction = [o.item() for o in out_dec][decoder_idx]
                results_final.append(
                    {
                        "image_path": path[i],
                        "label": labels[i].item(),
                        **{
                            f"prediction_dec_{dec_idx}": torch.argmax(
                                out_dec[dec_idx][i]
                            ).item()
                            if task_type == "classification"
                            else out_dec[dec_idx][i].item()
                            for dec_idx in range(num_decoders)
                        },
                    }
                )

        results_final_pandas = pandas.DataFrame(results_final)
        result_path = str(results_folder / dataloader.dataset.name / "predictions.csv")
        results_final_pandas.to_csv(
            result_path,
            index=False,
        )
        print(sty.fg.yellow + f"Result written in {result_path}" + sty.rs.fg)

    [evaluate_one_dataloader(dataloader) for dataloader in test_loaders]
