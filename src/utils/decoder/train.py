"""
General training script for decoder approach. The only thing you need to change is the loading `DATASET` variable. Note that in this case the EbbinghausTrain dataset is always generated on the fly (but you could specify the kwarg "path" to save/load it on disk).
"""

import toml
from datetime import datetime

# from src.utils.Config import Config
from src.utils.decoder.train_utils import decoder_step, ResNet152decoders, fix_dataset, log_neptune_init_info
from src.utils.net_utils import ExpMovingAverage, CumulativeAverage, run
from torch.utils.data import DataLoader
from src.utils.misc import weblog_dataset_info, pretty_print_dict, update_dict
from src.utils.callbacks import *
from src.utils.decoder.data_utils import RegressionDataset
import argparse
import torch.backends.cudnn as cudnn
from src.utils.net_utils import load_pretraining
from functools import partial
from torchvision.datasets import ImageFolder
import neptune.new as neptune


def run_train(run_info=None, network=None, training=None, stopping_conditions=None, saving_folders=None, monitoring=None):
    with open(os.path.dirname(__file__) + '/default_train_config.toml', 'r') as f:
        toml_config = toml.load(f)
    update_dict(toml_config, {'run_info': run_info if run_info else {},
                              'network': network if network else {},
                              'training': training if training else {},
                              'stopping_conditions': stopping_conditions if stopping_conditions else {},
                              'saving_folders': saving_folders if saving_folders else {},
                              'monitoring': monitoring if monitoring else {}})

    toml_config['run_info'] = {'run_id':  datetime.now().strftime("%d%m%Y_%H%M%S") if toml_config['run_info']['run_id'] is None else toml_config['run_info']['run_id'], 'completed': False}
    toml_config['saving_folders']['result_folder'] = f"results/{toml_config['run_info']['run_id']}" if not toml_config['saving_folders']['result_folder'] else toml_config['saving_folders']['result_folder']


    train_dataset = toml_config['training']['train_dataset'].rstrip("/")
    test_datasets = [i.rstrip("/") for i in toml_config['training']['test_datasets']]
    [assert_exists(p) for p in [train_dataset, *test_datasets]]

    weblogger = False
    if toml_config['monitoring']['neptune_proj_name']:
        try:
            neptune_run = neptune.init_run(api_token=os.environ['NEPTUNE_API_TOKEN'], project=toml_config['monitoring']['neptune_proj_name'])
            weblogger = neptune_run
            print(sty.fg.blue + "~~ NEPTUNE LOGGING ACTIVE ~~" + sty.rs.fg)
        except:
            print(
                "Initializing neptune didn't work, maybe you don't have the neptune client installed or you haven't set up the API token (https://docs.neptune.ai/getting-started/installation). Neptune logging won't be used :("
            )

    log_neptune_init_info(weblogger, toml_config, tags=None) if weblogger else None
    [pathlib.Path(path).mkdir(parents=True, exist_ok=True) for path in list(toml_config['saving_folders'].values())]

    toml.dump(toml_config, open(toml_config['saving_folders']['result_folder'] + '/train_config.toml', 'w'))
    pretty_print_dict(toml_config)

    use_cuda = torch.cuda.is_available()
    is_pycharm = True if "PYCHARM_HOSTED" in os.environ else False
    torch.cuda.set_device(toml_config['training']['gpu_num']) if torch.cuda.is_available() else None

    if not list(os.walk(train_dataset))[0][1]:
        print(
            sty.fg.yellow
            + sty.ef.inverse
            + "You pointed to a folder containin only images, which means that you are going to run a REGRESSION method"
            + sty.rs.ef
        )
        method = "regression"
    else:
        print(
            sty.fg.yellow
            + sty.ef.inverse
            + "You pointed to a dataset of folders, which means that you are going to run a CLASSIFICATION method"
            + sty.rs.ef
        )
        method = "classification"


    ds = ImageFolder if method == "classification" else RegressionDataset
    train_dataset = fix_dataset(ds(root=train_dataset), name_ds=os.path.basename(train_dataset))

    net = ResNet152decoders(
        imagenet_pt=True,
        num_outputs=1 if method == "regression" else len(train_dataset.classes),
        use_residual_decoder=toml_config['network']['use_residual_decoder'],
    )
    num_decoders = len(net.decoders)

    if toml_config['training']['continue_train']:
        pretraining = toml_config['training']['continue_train']
        load_pretraining(net, pretraining, optimizer=None, use_cuda=use_cuda)

    print(sty.ef.inverse + "FREEZING CORE NETWORK" + sty.rs.ef)

    for param in net.parameters():
        param.requires_grad = False
    for param in net.decoders.parameters():
        param.requires_grad = True
    # for param in net.decoders_residual.parameters():
    #     param.requires_grad = True

    net.cuda() if use_cuda else None

    # cudnn.benchmark is a property of the cudnn library that determines whether to use a cached version of the best convolution algorithm for the hardware or to re-evaluate the convolution algorithm for each forward pass.
    cudnn.benchmark = False if use_cuda else False

    net.train()
    loss_fn = torch.nn.MSELoss() if method == "regression" else torch.nn.CrossEntropyLoss()
    optimizers = [
        torch.optim.Adam(net.decoders[i].parameters(),
                         lr=toml_config['training']['learning_rate'],
                         weight_decay=toml_config['training']['weight_decay'])
        for i in range(num_decoders)
    ]

    train_loader = DataLoader(
        train_dataset,
        batch_size=toml_config['training']['batch_size'],
        drop_last=False,
        shuffle=True,
        num_workers=0 if use_cuda and not is_pycharm else 0,
        timeout=0 if use_cuda and not is_pycharm else 0,
        pin_memory=True,
    )

    weblog_dataset_info(
        train_loader, weblogger=weblogger, num_batches_to_log=1, log_text="train"
    ) if weblogger else None

    ds_type = RegressionDataset if method == "regression" else ImageFolder
    test_datasets = [
        fix_dataset(ds_type(root=path), name_ds=os.path.splitext(os.path.basename(path))[0]) for path in test_datasets
    ]

    test_loaders = [
        DataLoader(
            td,
            batch_size=toml_config['training']['batch_size'],
            drop_last=False,
            num_workers=8 if use_cuda and not is_pycharm else 0,
            timeout=0 if use_cuda and not is_pycharm else 0,
            pin_memory=True,
        )
        for td in test_datasets
    ]

    [weblog_dataset_info(td, weblogger=weblogger, num_batches_to_log=1, log_text="test") for td in test_loaders]

    step = decoder_step

    def call_run(loader, train, callbacks, method, logs_prefix="", logs=None, **kwargs):
        if logs is None:
            logs = {}
        logs.update({f"{logs_prefix}ema_loss": ExpMovingAverage(0.2)})

        if train:
            logs.update({f"{logs_prefix}ema_{log_type}_{i}": ExpMovingAverage(0.2) for i in range(6)})
        else:
            logs.update({f"{logs_prefix}{log_type}": CumulativeAverage()})
            logs.update({f"{logs_prefix}{log_type}_{i}": CumulativeAverage() for i in range(6)})

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

    log_type = "acc" if method == "classification" else "rmse"  # rmse: Root Mean Square Error : sqrt(MSE)
    all_cb = [
        StopFromUserInput(),
        ProgressBar(
            l=len(train_dataset),
            batch_size=toml_config['training']['batch_size'],
            logs_keys=["ema_loss", *[f"ema_{log_type}_{i}" for i in range(num_decoders)]],
        ),
        PrintNeptune(id="ema_loss", plot_every=10, weblogger=weblogger),
        *[PrintNeptune(id=f"ema_{log_type}_{i}", plot_every=10, weblogger=weblogger) for i in range(num_decoders)],
        # Either train for X epochs
        TriggerActionWhenReachingValue(
            mode="max",
            patience=1,
            value_to_reach=toml_config['stopping_conditions']['stop_at_epoch'],
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
                call_run=partial(call_run, method=method),
                plot_samples_corr_incorr=False,
                callbacks=[
                    SaveInfoCsv(
                        log_names=["epoch", *[f"{tl.dataset.name_ds}/{log_type}_{i}" for i in range(num_decoders)]],
                        path=toml_config['saving_folders']['result_folder'] + f"/{tl.dataset.name_ds}.csv",
                    ),
                    # if you don't use neptune, this will be ignored
                    PrintNeptune(
                        id=f"{tl.dataset.name_ds}/{log_type}",
                        plot_every=np.inf,
                        log_prefix="test_TRAIN",
                        weblogger=weblogger,
                    ),
                    PrintConsole(id=f"{tl.dataset.name_ds}/{log_type}", endln=" -- ", plot_every=np.inf, plot_at_end=True),
                    *[
                        PrintConsole(
                            id=f"{tl.dataset.name_ds}/{log_type}_{i}", endln=" " "/ ", plot_every=np.inf, plot_at_end=True
                        )
                        for i in range(num_decoders)
                    ],
                ],
            )
            for tl in test_loaders
        ],
    ]

    all_cb.append(
        SaveModel(net, toml_config['saving_folders']['model_output_path'], loss_metric_name="ema_loss")
    ) if toml_config['saving_folders']['model_output_path'] and not is_pycharm else None
    all_cb.append(
        TriggerActionWhenReachingValue(
            mode="min",
            patience=20,
            value_to_reach=toml_config['stopping_conditions']['stop_at_loss'],
            check_every=10,
            metric_name="ema_loss",
            action=stop,
            action_name=f"goal [{toml_config['stopping_conditions']['stop_at_loss']}]",
        )
    ) if toml_config['stopping_conditions']['stop_at_loss'] else None

    net, logs = call_run(train_loader, True, all_cb, method)
    weblogger.stop() if weblogger else None
    toml_config['run_info']['completed'] = True
    toml.dump(toml_config, open(toml_config['saving_folders']['result_folder'] + '/train_config.toml', 'w'))


def assert_exists(path):
    if not os.path.exists(path):
        assert False, sty.fg.red +f"Path {path} doesn't exist!" + sty.rs.fg



if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--toml_config_path', '-toml', default=f'{os.path.dirname(__file__)}/default_train_config.toml')
    args = parser.parse_known_args()[0]
    with open(args.toml_config_path, 'r') as f:
        toml_config = toml.load(f)
    print(f'**** Selected {args.toml_config_path} ****')
    run_train(**toml_config)
