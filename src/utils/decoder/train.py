"""
General training script for decoder approach. The only thing you need to change is the loading `DATASET` variable. Note that in this case the EbbinghausTrain dataset is always generated on the fly (but you could specify the kwarg "path" to save/load it on disk).
"""

import toml
from datetime import datetime

from src.utils.decoder.train_utils import (
    decoder_step,
    ResNet152decoders,
    fix_dataset,
    log_neptune_init_info,
)
from src.utils.net_utils import ExpMovingAverage, CumulativeAverage, run
from torch.utils.data import DataLoader
from src.utils.misc import (
    assert_exists,
    weblog_dataset_info,
    pretty_print_dict,
    update_dict,
)
from src.utils.callbacks import *
from src.utils.decoder.data_utils import ImageRegressionDataset
import argparse
import torch.backends.cudnn as cudnn
from src.utils.net_utils import load_pretraining
from functools import partial
from torchvision.datasets import ImageFolder

try:
    import neptune
except:
    pass
import inspect


def run_train(
    run_info=None,
    datasets=None,
    network=None,
    training=None,
    stopping_conditions=None,
    saving_folders=None,
    monitoring=None,
):
    with open(os.path.dirname(__file__) + "/default_train_config.toml", "r") as f:
        toml_config = toml.load(f)

    # update the toml_config file based on the input args to this function
    local_vars = locals()
    update_dict(
        toml_config,
        {
            i: local_vars[i] if local_vars[i] else {}
            for i in inspect.getfullargspec(run_train)[0]
        },
    )
    toml_config["run_info"] = {
        "run_id": datetime.now().strftime("%d%m%Y_%H%M%S")
        if not toml_config["run_info"]["run_id"]
        else toml_config["run_info"]["run_id"],
        "completed": False,
    }
    for f in toml_config["saving_folders"].keys():
        ff = {
            "result_folder": "results/decoder_training",
            "model_output_folder": "models/decoder_training",
        }[f]
        toml_config["saving_folders"][f] = (
            f"{ff}/{toml_config['run_info']['run_id']}"
            if not toml_config["saving_folders"][f]
            else toml_config["saving_folders"][f]
        )
        pathlib.Path(toml_config["saving_folders"][f]).mkdir(
            parents=True, exist_ok=True
        )

    weblogger = False
    if toml_config["monitoring"]["neptune_proj_name"]:
        try:
            neptune_run = neptune.init_run(
                api_token=os.environ["NEPTUNE_API_TOKEN"],
                project=toml_config["monitoring"]["neptune_proj_name"],
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
        open(
            toml_config["saving_folders"]["result_folder"] + "/train_config.toml", "w"
        ),
    )
    pretty_print_dict(toml_config)

    use_cuda = torch.cuda.is_available()
    torch.cuda.set_device(
        toml_config["training"]["gpu_num"]
    ) if torch.cuda.is_available() else None

    def load_dataset(ds_config):
        if toml_config["training"]["type_training"] == "classification":
            ds = ImageFolder(root=...)  # ToDo: CLASSIFICATION!
        elif toml_config["training"]["type_training"] == "regression":
            ds = ImageRegressionDataset(
                csv_file=ds_config["annotation_file"],
                img_path_col=ds_config["img_path_col_name"],
                label_cols=ds_config["label_cols"],
                filters=ds_config["filters"],
            )
        return fix_dataset(ds, name_ds=ds_config["name"])

    train_dataset = load_dataset(toml_config["datasets"]["training"])

    test_datasets = (
        [load_dataset(i) for i in toml_config["datasets"]["validation"]]
        if "validation" in toml_config["datasets"]
        else []
    )

    net = ResNet152decoders(
        imagenet_pt=toml_config["network"]["imagenet_pretrained"],
        num_outputs=toml_config["network"]["decoder_outputs"]
        if toml_config["training"]["type_training"] == "regression"
        else len(train_dataset.classes),
        use_residual_decoder=toml_config["network"]["use_residual_decoder"],
    )
    num_decoders = len(net.decoders)

    if toml_config["training"]["continue_train"]:
        pretraining = toml_config["training"]["continue_train"]
        load_pretraining(net, pretraining, optimizer=None, use_cuda=use_cuda)

    print(sty.ef.inverse + "FREEZING CORE NETWORK" + sty.rs.ef)

    for param in net.parameters():
        param.requires_grad = False
    for param in net.decoders.parameters():
        param.requires_grad = True

    net.cuda() if use_cuda else None

    cudnn.benchmark = False if use_cuda else False

    net.train()
    loss_fn = (
        torch.nn.MSELoss()
        if toml_config["training"]["type_training"] == "regression"
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

    train_loader = DataLoader(
        train_dataset,
        batch_size=toml_config["training"]["batch_size"],
        drop_last=False,
        shuffle=True,
        num_workers=8 if use_cuda else 0,
        timeout=0,
        pin_memory=True,
    )

    weblog_dataset_info(
        train_loader, weblogger=weblogger, num_batches_to_log=1, log_text="train"
    ) if weblogger else None

    test_loaders = [
        DataLoader(
            td,
            batch_size=toml_config["training"]["batch_size"],
            drop_last=False,
            num_workers=8 if use_cuda else 0,
            timeout=0,
            pin_memory=True,
        )
        for td in test_datasets
    ]

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
        "acc"
        if toml_config["training"]["type_training"] == "classification"
        else "rmse"
    )  # rmse: Root Mean Square Error : sqrt(MSE)
    all_callbacks = [
        StopFromUserInput(),
        ProgressBar(
            l=len(train_dataset),
            batch_size=toml_config["training"]["batch_size"],
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
            value_to_reach=toml_config["stopping_conditions"]["stop_at_epoch"],
            check_after_batch=False,
            metric_name="epoch",
            action=stop,
            action_name=f"{toml_config['stopping_conditions']['stop_at_epoch']} epochs",
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
                logs_prefix=f"{tl.dataset.name_ds}/",
                call_run=partial(
                    call_run, method=toml_config["training"]["type_training"]
                ),
                plot_samples_corr_incorr=False,
                callbacks=[
                    SaveInfoCsv(
                        log_names=[
                            "epoch",
                            *[
                                f"{tl.dataset.name_ds}/{log_type}_{i}"
                                for i in range(num_decoders)
                            ],
                        ],
                        path=toml_config["saving_folders"]["result_folder"]
                        + f"/{tl.dataset.name_ds}.csv",
                    ),
                    # if you don't use neptune, this will be ignored
                    PrintNeptune(
                        id=f"{tl.dataset.name_ds}/{log_type}",
                        plot_every=np.inf,
                        log_prefix="test_TRAIN",
                        weblogger=weblogger,
                    ),
                    PrintConsole(
                        id=f"{tl.dataset.name_ds}/{log_type}",
                        endln=" -- ",
                        plot_every=np.inf,
                        plot_at_end=True,
                    ),
                    *[
                        PrintConsole(
                            id=f"{tl.dataset.name_ds}/{log_type}_{i}",
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
        SaveModel(
            net,
            toml_config["saving_folders"]["model_output_folder"],
            loss_metric_name="ema_loss",
        )
    ) if toml_config["training"]["save_trained_model"] else None

    all_callbacks.append(
        TriggerActionWhenReachingValue(
            mode="min",
            patience=20,
            value_to_reach=toml_config["stopping_conditions"]["stop_at_loss"],
            check_every=10,
            metric_name="ema_loss",
            action=stop,
            action_name=f"goal [{toml_config['stopping_conditions']['stop_at_loss']}]",
        )
    ) if toml_config["stopping_conditions"]["stop_at_loss"] else None

    net, logs = call_run(
        train_loader, True, all_callbacks, toml_config["training"]["type_training"]
    )
    weblogger.stop() if weblogger else None
    toml_config["run_info"]["completed"] = True
    toml.dump(
        toml_config,
        open(
            toml_config["saving_folders"]["result_folder"] + "/train_config.toml", "w"
        ),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--toml_config_path",
        "-toml",
        default=f"{os.path.dirname(__file__)}/default_train_config.toml",
    )
    args = parser.parse_known_args()[0]
    with open(args.toml_config_path, "r") as f:
        toml_config = toml.load(f)
    print(f"**** Selected {args.toml_config_path} ****")
    run_train(**toml_config)
