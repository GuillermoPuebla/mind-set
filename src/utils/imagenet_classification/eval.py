import json
import os
import torch
import torch.backends.cudnn as cudnn

# from rich import print
from src.utils.callbacks import *
from src.utils.dataset_utils import ImageNetClasses, get_dataloader
from src.utils.misc import pretty_print_dict, update_dict
from src.utils.net_utils import load_pretraining, make_cuda, GrabNet, ResNet152decoders
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from tqdm import tqdm
from pathlib import Path
import pandas
import toml
import inspect


def classification_evaluate(
    task_type=None, gpu_num=None, eval=None, network=None, saving_folders=None, **kwargs
):
    with open(
        os.path.dirname(__file__) + "/default_classification_config.toml", "r"
    ) as f:
        toml_config = toml.load(f)

    # update the toml_config file based on the input args to this function
    local_vars = locals()
    update_dict(
        toml_config,
        {i: local_vars[i] for i in inspect.getfullargspec(classification_evaluate)[0]},
    )
    pretty_print_dict(toml_config, name="PARAMETERS")
    use_cuda = torch.cuda.is_available()
    torch.cuda.set_device(toml_config["gpu_num"]) if torch.cuda.is_available() else None

    test_loaders = [
        get_dataloader(
            toml_config,
            "classification",
            ds_config=i,
            transf_config=toml_config["transformation"],
            batch_size=toml_config["network"]["batch_size"],
            return_path=True,
        )
        for i in toml_config["eval"]["datasets"]
    ]

    net, _, _ = GrabNet.get_net(
        toml_config["network"]["architecture_name"],
        imagenet_pt=True if toml_config["network"]["imagenet_pretrained"] else False,
        num_classes=1000,  # num classes of ImageNet
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
    imagenet_classes = ImageNetClasses()

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
            output = net(images)
            for i in range(len(labels)):
                prediction = torch.argmax(output[i]).item()

                results_final.append(
                    {
                        "image_path": path[i],
                        "label_idx": labels[i].item(),
                        "label_class_name": imagenet_classes.idx2label[
                            labels[i].item()
                        ],
                        "prediction_idx": prediction,
                        "prediction_class_name": imagenet_classes.idx2label[prediction],
                    }
                )

        results_final_pandas = pandas.DataFrame(results_final)
        results_final_pandas.to_csv(
            str(results_folder / dataloader.dataset.name / "predictions.csv"),
            index=False,
        )

    [evaluate_one_dataloader(dataloader) for dataloader in test_loaders]
