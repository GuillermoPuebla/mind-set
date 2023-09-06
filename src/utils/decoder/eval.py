import json
import os
import torch
import torch.backends.cudnn as cudnn
from rich import print
from src.utils.callbacks import *
from src.utils.decoder.data_utils import ImageRegressionDataset
from src.utils.decoder.train_utils import ResNet152decoders, fix_dataset
from src.utils.misc import pretty_print_dict, update_dict
from src.utils.net_utils import load_pretraining, make_cuda
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from tqdm import tqdm
from pathlib import Path
import pandas
import toml
import inspect


def decoder_evaluate(
    task_type=None,
    gpu_num=None,
    datasets=None,
    network=None,
    saving_folders=None,
):
    with open(os.path.dirname(__file__) + "/default_decoder_config.toml", "r") as f:
        toml_config = toml.load(f)

    # update the toml_config file based on the input args to this function
    local_vars = locals()
    update_dict(
        toml_config,
        {
            i: local_vars[i] if local_vars[i] is not None else {}
            for i in inspect.getfullargspec(decoder_evaluate)[0]
        },
    )
    pretty_print_dict(toml_config)
    use_cuda = torch.cuda.is_available()
    torch.cuda.set_device(toml_config["gpu_num"]) if torch.cuda.is_available() else None

    def load_dataset(ds_config):
        if toml_config["task_type"] == "classification":
            ds = ImageFolder(root=...)  # TODO: CLASSIFICATION!
        elif toml_config["task_type"] == "regression":
            ds = ImageRegressionDataset(
                csv_file=ds_config["annotation_file"],
                img_path_col=ds_config["img_path_col_name"],
                label_cols=ds_config["label_cols"],
                filters=ds_config["filters"],
            )
        return fix_dataset(ds, name_ds=ds_config["name"])

    test_datasets = [load_dataset(i) for i in toml_config["datasets"]["validation"]]

    net = ResNet152decoders(
        imagenet_pt=toml_config["network"]["imagenet_pretrained"],
        num_outputs=toml_config["network"]["decoder_outputs"]
        if toml_config["task_type"] == "regression"
        else len(test_datasets[0].classes),
        use_residual_decoder=toml_config["network"]["use_residual_decoder"],
    )
    num_decoders = len(net.decoders)

    load_pretraining(
        net=net,
        optimizers=None,
        network_path=toml_config["network"]["load_path"],
        optimizers_path=None,
    )
    net.eval()

    cudnn.benchmark = True if use_cuda else False

    test_loaders = [
        DataLoader(
            td,
            batch_size=toml_config["network"]["batch_size"],
            drop_last=False,
            num_workers=8 if use_cuda else 0,
            timeout=0,
            pin_memory=True,
        )
        for td in test_datasets
    ]
    print("STARTING EVALUATION")

    result_final = []

    def decoder_test(data, model, use_cuda):
        images, labels = data
        images = make_cuda(images, use_cuda)
        labels = make_cuda(labels, use_cuda)
        out_dec = model(images)
        for decoder_idx in range(num_decoders):
            for i in range(len(labels)):
                try:
                    prediction = out_dec[decoder_idx][i].item()
                except IndexError:
                    prediction = [o.item() for o in out_dec][decoder_idx]
                result_final.append(
                    {
                        "decoder": decoder_idx,
                        "label": labels[i].item(),
                        "prediction": prediction,
                    }
                )

    results_folder = pathlib.Path(toml_config["saving_folders"]["results_folder"])
    results_folder.mkdir(parents=True, exist_ok=True)

    def evaluate_one_dataloader(dataloader):
        for _, data in enumerate(tqdm(dataloader, colour="yellow")):
            decoder_test(data, net, use_cuda)

            result_final_pandas = pandas.DataFrame(result_final)
            result_final_pandas.to_csv(
                str(results_folder / dataloader.name / "predictions.csv"), index=False
            )

    [evaluate_one_dataloader(dataloader) for dataloader in test_loaders]
