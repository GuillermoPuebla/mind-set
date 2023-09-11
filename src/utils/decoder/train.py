from typing import Any
import torchvision
import toml
from datetime import datetime
from src.utils.dataset_utils import load_dataset

from src.utils.decoder.train_utils import (
    decoder_step,
    log_neptune_init_info,
)
from src.utils.net_utils import ExpMovingAverage, CumulativeAverage, GrabNet, run
from torch.utils.data import DataLoader
from src.utils.misc import (
    assert_exists,
    weblog_dataset_info,
    pretty_print_dict,
    update_dict,
)
from src.utils.callbacks import *
import argparse
import torch.backends.cudnn as cudnn
from src.utils.net_utils import load_pretraining
from functools import partial
from torchvision.datasets import ImageFolder
import pathlib

try:
    import neptune
except:
    pass
import inspect


def decoder_train(
    task_type=None,
    gpu_num=None,
    train_info=None,
    eval=None,
    network=None,
    transformation=None,
    training=None,
    saving_folders=None,
):
    with open(os.path.dirname(__file__) + "/default_decoder_config.toml", "r") as f:
        toml_config = toml.load(f)

    # update the toml_config file based on the input args to this function
    local_vars = locals()
    update_dict(
        toml_config,
        {i: local_vars[i] for i in inspect.getfullargspec(decoder_train)[0]},
    )
    toml_config["training"].update(
        {
            "train_id": datetime.now().strftime("%d%m%Y_%H%M%S")
            if not toml_config["training"]["train_id"]
            else toml_config["training"]["train_id"],
            "completed": False,
        }
    )

    results_folder_id = (
        pathlib.Path(toml_config["saving_folders"]["results_folder"])
        / toml_config["training"]["train_id"]
    )
    model_output_folder_id = (
        pathlib.Path(toml_config["saving_folders"]["model_output_folder"])
        / toml_config["training"]["train_id"]
    )
    results_folder_id.mkdir(parents=True, exist_ok=True)
    model_output_folder_id.mkdir(parents=True, exist_ok=True)

    weblogger = False
    if toml_config["training"]["monitoring"]["neptune_proj_name"]:
        try:
            neptune_run = neptune.init_run(
                api_token=os.environ["NEPTUNE_API_TOKEN"],
                project=toml_config["training"]["monitoring"]["neptune_proj_name"],
            )
            weblogger = neptune_run
            print(sty.fg.blue + "~~ NEPTUNE LOGGING ACTIVE ~~" + sty.rs.fg)
        except:
            print(
                "Initializing neptune didn't work, maybe you don't have the neptune client installed or you haven't set up the API token (https://docs.neptune.ai/getting-started/installation). Neptune logging won't be used :("
            )

    log_neptune_init_info(weblogger, toml_config, tags=None) if weblogger else None

    toml.dump(
        toml_config,
        open(str(results_folder_id / "train_config.toml"), "w"),
    )
    pretty_print_dict(toml_config, name="PARAMETERS")

    use_cuda = torch.cuda.is_available()
    torch.cuda.set_device(toml_config["gpu_num"]) if torch.cuda.is_available() else None

    train_dataset = load_dataset(
        toml_config["task_type"],
        ds_config=toml_config["training"]["dataset"],
        transf_config=toml_config["transformation"],
    )

    test_datasets = (
        [
            load_dataset(
                toml_config["task_type"],
                ds_config=i,
                transf_config=toml_config["transformation"],
            )
            for i in toml_config["eval"]["datasets"]
        ]
        if toml_config["training"]["evaluate_during_training"] in toml_config
        else []
    )
    assert toml_config["network"]["architecture_name"] in [
        "resnet152_decoder",
        "resnet152_decoder_residual",
    ], f"Network.name needs to be either `resnet152_decoder` or `resnet152_decoder_residual`. You used {toml_config['network']['name']}"

    net, _, _ = GrabNet.get_net(
        toml_config["network"]["architecture_name"],
        imagenet_pt=True if toml_config["network"]["imagenet_pretrained"] else False,
        num_classes=toml_config["network"]["decoder_outputs"]
        if toml_config["task_type"] == "regression"
        else len(train_dataset.classes),
    )

    num_decoders = len(net.decoders)
    loss_fn = (
        torch.nn.MSELoss()
        if toml_config["task_type"] == "regression"
        else torch.nn.CrossEntropyLoss()
    )
    optimizers = [
        torch.optim.Adam(
            net.decoders[i].parameters(),
            lr=toml_config["training"]["learning_rate"],
            weight_decay=toml_config["training"]["weight_decay"],
        )
        for i in range(num_decoders)
    ]

    load_pretraining(
        net=net,
        optimizers=optimizers,
        network_path=toml_config["network"]["load_path"],
        optimizers_path=toml_config["training"]["optimizers_path"],
    )
    print(sty.ef.inverse + "FREEZING CORE NETWORK" + sty.rs.ef)

    for param in net.parameters():
        param.requires_grad = False
    for param in net.decoders.parameters():
        param.requires_grad = True

    net.cuda() if use_cuda else None

    cudnn.benchmark = False if use_cuda else False

    net.train()

    train_loader = DataLoader(
        train_dataset,
        batch_size=toml_config["network"]["batch_size"],
        drop_last=False,
        shuffle=True,
        num_workers=8 if use_cuda else 0,
        timeout=0,
        pin_memory=True,
    )

    weblog_dataset_info(
        train_loader, weblogger=weblogger, num_batches_to_log=1, log_text="train"
    ) if weblogger else None

    test_loaders = (
        [
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
        if toml_config["training"]["evaluate_during_training"]
        else []
    )

    [
        weblog_dataset_info(
            td, weblogger=weblogger, num_batches_to_log=1, log_text="test"
        )
        for td in test_loaders
    ]

    step = decoder_step

    def call_run(loader, train, callbacks, method, logs_prefix="", logs=None, **kwargs):
        if logs is None:
            logs = {}
        logs.update({f"{logs_prefix}ema_loss": ExpMovingAverage(0.2)})

        if train:
            logs.update(
                {
                    f"{logs_prefix}ema_{log_type}_{i}": ExpMovingAverage(0.2)
                    for i in range(6)
                }
            )
        else:
            logs.update({f"{logs_prefix}{log_type}": CumulativeAverage()})
            logs.update(
                {f"{logs_prefix}{log_type}_{i}": CumulativeAverage() for i in range(6)}
            )

        return run(
            loader,
            use_cuda=use_cuda,
            net=net,
            callbacks=callbacks,
            loss_fn=loss_fn,
            optimizer=optimizers,
            iteration_step=step,
            train=train,
            logs=logs,
            logs_prefix=logs_prefix,
            collect_data=kwargs.pop("collect_data", False),
            stats=train_dataset.stats,
            method=method,
        )

    def stop(logs, cb):
        logs["stop"] = True
        print("Early Stopping")

    log_type = (
        "acc" if toml_config["task_type"] == "classification" else "rmse"
    )  # rmse: Root Mean Square Error : sqrt(MSE)
    all_callbacks = [
        # StopFromUserInput(),
        ProgressBar(
            l=len(train_dataset),
            batch_size=toml_config["network"]["batch_size"],
            logs_keys=[
                "ema_loss",
                *[f"ema_{log_type}_{i}" for i in range(num_decoders)],
            ],
        ),
        PrintNeptune(id="ema_loss", plot_every=10, weblogger=weblogger),
        *[
            PrintNeptune(id=f"ema_{log_type}_{i}", plot_every=10, weblogger=weblogger)
            for i in range(num_decoders)
        ],
        TriggerActionWhenReachingValue(
            mode="max",
            patience=1,
            value_to_reach=toml_config["training"]["stopping_conditions"][
                "stop_at_epoch"
            ],
            check_after_batch=False,
            metric_name="epoch",
            action=stop,
            action_name=f"{toml_config['training']['stopping_conditions']['stop_at_epoch']} epochs",
        ),
        *[
            DuringTrainingTest(
                testing_loaders=tl,
                eval_mode=False,
                every_x_epochs=1,
                auto_increase=False,
                weblogger=weblogger,
                log_text="test during train TRAINmode",
                use_cuda=use_cuda,
                logs_prefix=f"{tl.dataset.name}/",
                call_run=partial(call_run, method=toml_config["task_type"]),
                plot_samples_corr_incorr=False,
                callbacks=[
                    SaveInfoCsv(
                        log_names=[
                            "epoch",
                            *[
                                f"{tl.dataset.name}/{log_type}_{i}"
                                for i in range(num_decoders)
                            ],
                        ],
                        path=str(results_folder_id / f"{tl.dataset.name}.csv"),
                    ),
                    # if you don't use neptune, this will be ignored
                    PrintNeptune(
                        id=f"{tl.dataset.name}/{log_type}",
                        plot_every=np.inf,
                        log_prefix="test_TRAIN",
                        weblogger=weblogger,
                    ),
                    PrintConsole(
                        id=f"{tl.dataset.name}/{log_type}",
                        endln=" -- ",
                        plot_every=np.inf,
                        plot_at_end=True,
                    ),
                    *[
                        PrintConsole(
                            id=f"{tl.dataset.name}/{log_type}_{i}",
                            endln=" " "/ ",
                            plot_every=np.inf,
                            plot_at_end=True,
                        )
                        for i in range(num_decoders)
                    ],
                ],
            )
            for tl in test_loaders
        ],
    ]

    all_callbacks.append(
        SaveModelAndOpt(
            net,
            str(model_output_folder_id),
            loss_metric_name="ema_loss",
            optimizers=optimizers,
            at_epoch_end=toml_config["training"]["save_at_epoch_end"],
        )
    ) if toml_config["training"]["save_trained_model"] else None

    all_callbacks.append(
        TriggerActionWhenReachingValue(
            mode="min",
            patience=20,
            value_to_reach=toml_config["training"]["stopping_conditions"][
                "stop_at_loss"
            ],
            check_every=10,
            metric_name="ema_loss",
            action=stop,
            action_name=f"goal [{toml_config['training']['stopping_conditions']['stop_at_loss']}]",
        )
    ) if toml_config["training"]["stopping_conditions"]["stop_at_loss"] else None

    net, logs = call_run(train_loader, True, all_callbacks, toml_config["task_type"])
    weblogger.stop() if weblogger else None
    toml_config["training"]["completed"] = True
    toml.dump(
        toml_config,
        open(str(results_folder_id / "train_config.toml"), "w"),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--toml_config_path",
        "-toml",
        default=f"{os.path.dirname(__file__)}/default_decoder_config.toml",
    )
    args = parser.parse_known_args()[0]
    with open(args.toml_config_path, "r") as f:
        toml_config = toml.load(f)
    print(f"**** Selected {args.toml_config_path} ****")
    decoder_train(**toml_config)
