import json
import os
import torch
import torch.backends.cudnn as cudnn
from rich import print
from src.utils.callbacks import *
from src.utils.decoder.data_utils import RegressionDataset
from src.utils.decoder.train_utils import ResNet152decoders, fix_dataset
from src.utils.net_utils import load_pretraining, make_cuda
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from tqdm import tqdm
from pathlib import Path
import pandas


def decoder_evaluate(
    save_name="result.json",
    results_folder="./results/tmp/",
    pretraining="",
    dataset_folder="",
    gpu_num=0,
    batch_size=64,
    use_residual_decoder=False,
):
    """
    Evaluate a trained decoder with ResNet152 on a test dataset, supports both residual and linear.
    The important parameters are the test_results_folder, pretraining and test dataset.
    Load the checkpoint from the pretraining path, the function will take a folder of iamges
    and run all the decoders for each image and save the results in a JSON file in the test_results_folder.

    The json file will have the following structure:
    [
        {
            "decoder": *,  # the decoder number (0-4, 5 decoders)
            "label": *,  # before the _ in the image name
            "prediction": *  # the prediction of the decoder
        },
        ...
    ]

    params:
        :param test_results_folder: str, optional, default="./results/tmp/"
            The folder path to save the results.
        :param test_dataset: str
            The path to the test dataset folder.
        :param gpu_num: int, optional, default=0
            The GPU number to use for computation.
        :param batch_size: int, optional, default=64
            The batch size for processing the test dataset.
        :param pretraining: str, optional, default=""
            The pre-training file path.
        :param use_residual_decoder: bool, optional, default=False
            Whether to use a residual decoder or not.

    :return: None
        Saves the evaluation results to a JSON file in the test_results_folder.
    """

    dataset_folder = str(Path(dataset_folder))

    def assert_exists(path):
        if not os.path.exists(path):
            assert False, f"Path {path} doesn't exist!"

    assert_exists(dataset_folder)

    dataset_folder = dataset_folder.rstrip("/")
    use_cuda = torch.cuda.is_available()
    torch.cuda.set_device(gpu_num) if use_cuda else None

    if not list(os.walk(dataset_folder))[0][1]:
        print(
            "You pointed to a folder containing only images, which means that you are going to run a REGRESSION method"
        )
        method = "regression"
        loading_method = RegressionDataset
    else:
        print(
            "You pointed to a dataset of folders, which means that you are going to run a CLASSIFICATION method"
        )
        method = "classification"
        loading_method = ImageFolder

    dataset_folder = fix_dataset(
        loading_method(root=dataset_folder),
        name_ds=os.path.basename(dataset_folder),
    )

    net = ResNet152decoders(
        imagenet_pt=True,
        num_outputs=1 if method == "regression" else len(dataset_folder.classes),
        use_residual_decoder=use_residual_decoder,
    )

    net.cuda() if use_cuda else None
    num_decoders = len(net.decoders)

    load_pretraining(net, pretraining, optimizer=None, use_cuda=use_cuda)

    # freeze the network -------------------------
    print("FREEZING BOTH NETWORKS")
    for param in net.parameters():
        param.requires_grad = False
    for param in net.decoders.parameters():
        param.requires_grad = False
    net.eval()  # set the model to training mode

    cudnn.benchmark = True if use_cuda else False

    test_loader = DataLoader(
        dataset_folder,
        batch_size=batch_size,
        drop_last=False,
        shuffle=True,
        num_workers=0,
        timeout=0,
        pin_memory=True,
    )

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

    with torch.no_grad():
        for _, data in enumerate(tqdm(test_loader, colour="yellow")):
            decoder_test(data, net, use_cuda)

    # save the results
    os.makedirs(results_folder, exist_ok=True)

    json.dump(
        result_final,
        open(os.path.join(results_folder, save_name + ".json"), "w"),
    )

    # make result_final a pandas dataframe
    result_final_pandas = pandas.DataFrame(result_final)
    result_final_pandas.to_pickle(os.path.join(results_folder, save_name + ".pkl"))


if __name__ == "__main__":
    # for testing a single folder -------------------------
    decoder_evaluate(
        save_name="red_top",
        results_folder="./results/jastrow/",
        dataset_folder="data/low_level_vision/jastrow_test/red_on_top_0_smaller/",
        pretraining="private/checkpoints/linear_decoder_jastrow__final_checkpoint.pt",
        gpu_num=0,
        batch_size=16,
        use_residual_decoder=True,
    )

    # # for testing multiple folders -------------------------
    # path_jastrow_test_data = Path("data", "low_level_vision", "jastrow_test")
    # dataset_folders = os.listdir(path_jastrow_test_data)
    # dataset_folders = [i for i in dataset_folders if not i.startswith(".")]

    # for dataset_folder in tqdm(dataset_folders):
    #     decoder_evaluate(
    #         save_name=dataset_folder,
    #         results_folder="./results/jastrow/",
    #         dataset_folder=path_jastrow_test_data / dataset_folder,
    #         pretraining="private/checkpoints/residual_decoder_color_picker.pt",
    #         gpu_num=0,
    #         batch_size=16,
    #         use_residual_decoder=True,
    #     )
