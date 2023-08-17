import torch
import torchvision
from torch import nn as nn
from src.utils.net_utils import make_cuda
from src.utils.misc import imshow_batch, convert_lists_to_strings
from copy import deepcopy
import numpy as np
from torch.nn import functional as F

# if method == "regression":
#     if train:
#         [
#             logs[f"{logs_prefix}ema_rmse_{idx}"].add(torch.sqrt(ms).item())
#             for idx, ms in enumerate(loss_decoder)
#         ]
#     else:
#         [
#             logs[f"{logs_prefix}rmse_{idx}"].add(torch.sqrt(ms).item())
#             for idx, ms in enumerate(loss_decoder)
#         ]

#         logs[f"{logs_prefix}rmse"].add(torch.sqrt(loss / num_decoders).item())

# elif method == "classification":
#     if train:
#         [
#             logs[f"{logs_prefix}ema_acc_{idx}"].add(
#                 torch.mean((torch.argmax(out_dec[idx], 1) == labels).float()).item()
#             )
#             for idx in range(len(loss_decoder))
#         ]
#     else:
#         [
#             logs[f"{logs_prefix}acc_{idx}"].add(
#                 torch.mean((torch.argmax(out_dec[idx], 1) == labels).float()).item()
#             )
#             for idx in range(len(loss_decoder))
#         ]
#         logs[f"{logs_prefix}acc"].add(
#             torch.mean(
#                 torch.tensor(
#                     [
#                         torch.mean(
#                             (torch.argmax(out_dec[idx], 1) == labels).float()
#                         ).item()
#                         for idx in range(len(loss_decoder))
#                     ]
#                 )
#             ).item()
#         )


def fix_dataset(dataset, name_ds=""):
    dataset.name_ds = name_ds
    dataset.stats = {"mean": [0.491, 0.482, 0.44], "std": [0.247, 0.243, 0.262]}
    add_resize = False
    if next(iter(dataset))[0].size[0] != 244:
        add_resize = True

    dataset.transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                mean=dataset.stats["mean"], std=dataset.stats["std"]
            ),
        ]
    )
    if add_resize:
        dataset.transform.transforms.insert(0, torchvision.transforms.Resize(224))
    return dataset


def update_logs(logs, loss_decoders, labels, method, logs_prefix, train=True):
    logs[f"{logs_prefix}ema_loss"].add(sum(loss_decoders))
    prefix = "ema_" if train else ""
    if method == "regression":
        for idx, ms in enumerate(loss_decoders):
            logs[f"{logs_prefix}{prefix}rmse_{idx}"].add(torch.sqrt(ms).item())
        if not train:
            logs[f"{logs_prefix}rmse"].add(
                torch.sqrt(sum(loss_decoders) / len(loss_decoders)).item()
            )
    elif method == "classification":
        for idx in range(len(loss_decoders)):
            acc = torch.mean(
                (torch.argmax(loss_decoders[idx], 1) == labels).float()
            ).item()
            logs[f"{logs_prefix}{prefix}acc_{idx}"].add(acc)
        average_acc = torch.mean(
            torch.tensor(
                [
                    logs[f"{logs_prefix}{prefix}acc_{idx}"].value
                    for idx in range(len(loss_decoders))
                ]
            )
        ).item()
        if not train:
            logs[f"{logs_prefix}acc"].add(average_acc)


def decoder_step(
    data,
    model,
    loss_fn,
    optimizers,
    use_cuda,
    logs,
    logs_prefix,
    train,
    method,
    **kwargs,
):
    num_decoders = len(model.decoders)

    images, labels = data
    images = make_cuda(images, use_cuda)
    labels = make_cuda(labels, use_cuda)
    if train:
        [optimizers[i].zero_grad() for i in range(num_decoders)]
    out_dec = model(images)
    loss = make_cuda(
        torch.tensor([0.0], dtype=torch.float, requires_grad=True), use_cuda
    )
    loss_decoder = []
    for _, od in enumerate(out_dec):
        loss_decoder.append(loss_fn(od, labels))
        loss = loss + loss_decoder[-1]

    update_logs(logs, loss_decoder, labels, method, logs_prefix, train)

    if "collect_data" in kwargs and kwargs["collect_data"]:
        logs["data"] = data

    if train:
        loss.backward()
        [optimizers[i].step() for i in range(num_decoders)]


def log_neptune_init_info(neptune_logger, toml_config, tags=None):
    tags = [] if tags is None else tags
    neptune_logger["sys/name"] = toml_config["train_info"][
        "run_id"
    ]  # to be consistent with before
    neptune_logger["toml_config"] = convert_lists_to_strings(toml_config)
    neptune_logger["toml_config_file"].upload(
        toml_config["train_info"]["save_folder"] + "/toml_config.txt"
    )
    neptune_logger["sys/tags"].add(tags)


def config_to_path_train(config):
    return f"/ebbinghaus/decoder/{config.network_name}"


def replace_layer(net, layer_class, new_layer_class):
    """This function replaces a specific layer class in a given neural network with a new layer class. The input parameters are:
    net: The neural network object in which the layer replacement is to be done
    layer_class: The class of the layer to be replaced
    new_layer_class: The class of the new layer to replace the old one
    """

    layers = deepcopy(
        list(net.named_modules())
    )  # copy to avoid RuntimeError: dictionary changed size during iteration
    for name, layer in layers:
        if isinstance(layer, layer_class):
            names = name.split(".")
            parent = net
            for n in names[:-1]:
                parent = getattr(parent, n)
            setattr(parent, names[-1], new_layer_class())


class ResidualBlockPreActivation(nn.Module):
    """
    Homemade implementation of residual block with pre-activation from He et al. (2016)
    """

    def __init__(self, channels1, channels2, res_stride=1):
        super().__init__()

        self.bn1 = nn.BatchNorm2d(channels1)
        self.conv1 = nn.Conv2d(
            channels1,
            channels2,
            kernel_size=3,
            stride=res_stride,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(channels2)
        self.conv2 = nn.Conv2d(
            channels2, channels2, kernel_size=3, stride=1, padding=1, bias=False
        )

        if res_stride != 1 or channels2 != channels1:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    channels1, channels2, kernel_size=1, stride=res_stride, bias=False
                ),
                nn.BatchNorm2d(channels2),
            )

        else:
            self.shortcut = nn.Sequential()

    def forward(self, x):
        # original forward pass: Conv2d > BatchNorm2d > ReLU > Conv2D >  BatchNorm2d > ADD > ReLU
        # pre-activation forward pass: BatchNorm2d > ReLU > Conv2d > BatchNorm2d > ReLU > Conv2d > ADD
        out = self.bn1(x)
        out = F.relu(out)
        out = self.conv1(out)
        out = self.bn2(out)
        out = F.relu(out)
        out = self.conv2(out)

        out += self.shortcut(x)

        return out


def make_layer(block, in_channel, out_channel, num_blocks, stride):
    """
    Make a layer of residual blocks
    """
    layers = []
    layers.append(block(channels1=in_channel, channels2=out_channel, res_stride=stride))
    for _ in np.arange(num_blocks - 1):
        layers.append(block(channels1=out_channel, channels2=out_channel))
    return nn.Sequential(*layers)


class ResNet152decoders(nn.Module):
    """
    ResNet152 with decoders
    """

    def __init__(
        self,
        imagenet_pt,
        num_outputs=1,
        disable_batch_norm=False,
        use_residual_decoder=False,
        **kwargs,
    ):
        super().__init__()

        pretrained_weights = "IMAGENET1K_V1" if imagenet_pt else None

        self.net = torchvision.models.resnet152(
            weights=pretrained_weights,
            progress=True,
            **kwargs,
        )

        if disable_batch_norm:
            replace_layer(self.net, nn.BatchNorm2d, nn.Identity)

        self.use_residual_decoder = use_residual_decoder

        if use_residual_decoder:
            decoder_1 = nn.Sequential(  # input: 3, 224, 224
                make_layer(
                    block=ResidualBlockPreActivation,
                    in_channel=3,
                    out_channel=64,
                    num_blocks=1,
                    stride=2,
                ),
                make_layer(
                    block=ResidualBlockPreActivation,
                    in_channel=64,
                    out_channel=64,
                    num_blocks=1,
                    stride=2,
                ),
                make_layer(
                    block=ResidualBlockPreActivation,
                    in_channel=64,
                    out_channel=64,
                    num_blocks=1,
                    stride=2,
                ),
                # here the input will be 64, 28, 28
                nn.Flatten(),
                nn.Linear(64 * 28 * 28, 1024),
                nn.ReLU(),
                nn.Linear(1024, 256),
                nn.ReLU(),
                nn.Linear(256, 64),
                nn.ReLU(),
                nn.Linear(64, num_outputs),
            )
            decoder_2 = nn.Sequential(  # input: 256, 56, 56
                make_layer(
                    block=ResidualBlockPreActivation,
                    in_channel=256,
                    out_channel=256,
                    num_blocks=1,
                    stride=2,
                ),
                make_layer(
                    block=ResidualBlockPreActivation,
                    in_channel=256,
                    out_channel=256,
                    num_blocks=1,
                    stride=2,
                ),
                make_layer(
                    block=ResidualBlockPreActivation,
                    in_channel=256,
                    out_channel=256,
                    num_blocks=1,
                    stride=2,
                ),
                nn.Flatten(),
                nn.Linear(256 * 7 * 7, 1024),
                nn.ReLU(),
                nn.Linear(1024, 256),
                nn.ReLU(),
                nn.Linear(256, 64),
                nn.ReLU(),
                nn.Linear(64, num_outputs),
            )

            decoder_3 = nn.Sequential(  # input: 512, 28, 28
                make_layer(
                    block=ResidualBlockPreActivation,
                    in_channel=512,
                    out_channel=512,
                    num_blocks=2,
                    stride=2,
                ),
                make_layer(
                    block=ResidualBlockPreActivation,
                    in_channel=512,
                    out_channel=512,
                    num_blocks=1,
                    stride=2,
                ),
                nn.Flatten(),
                nn.Linear(512 * 7 * 7, 1024),
                nn.ReLU(),
                nn.Linear(1024, 256),
                nn.ReLU(),
                nn.Linear(256, 64),
                nn.ReLU(),
                nn.Linear(64, num_outputs),
            )

            decoder_4 = nn.Sequential(  # input: 1024, 14, 14
                make_layer(
                    block=ResidualBlockPreActivation,
                    in_channel=1024,
                    out_channel=1024,
                    num_blocks=3,
                    stride=2,
                ),
                nn.Flatten(),
                nn.Linear(1024 * 7 * 7, 1024),
                nn.ReLU(),
                nn.Linear(1024, 256),
                nn.ReLU(),
                nn.Linear(256, 64),
                nn.ReLU(),
                nn.Linear(64, num_outputs),
            )
            decoder_5 = nn.Sequential(  # input: 2048, 7, 7
                make_layer(
                    block=ResidualBlockPreActivation,
                    in_channel=2048,
                    out_channel=2048,
                    num_blocks=3,
                    stride=1,
                ),
                nn.Flatten(),
                nn.Linear(2048 * 7 * 7, 1024),
                nn.ReLU(),
                nn.Linear(1024, 256),
                nn.ReLU(),
                nn.Linear(256, 64),
                nn.ReLU(),
                nn.Linear(64, num_outputs),
            )
            decoder_6 = nn.Sequential(  # input: 2048
                nn.Linear(2048, 1024),
                nn.ReLU(),
                nn.Linear(1024, num_outputs),
            )
            self.decoders = nn.ModuleList(
                [decoder_1, decoder_2, decoder_3, decoder_4, decoder_5, decoder_6]
            )

        else:
            self.decoders = nn.ModuleList(
                [
                    nn.Linear(3 * 224 * 224, num_outputs),
                    nn.Linear(802816, num_outputs),
                    nn.Linear(401408, num_outputs),
                    nn.Linear(200704, num_outputs),
                    nn.Linear(100352, num_outputs),
                    nn.Linear(2048, num_outputs),
                ]
            )

    def forward(self, x):
        if self.use_residual_decoder:
            return self._forward_residual_decoder(x)
        else:
            return self._forward(x)

    def _forward_residual_decoder(self, x):
        out_dec_res = []
        out_dec_res.append(self.decoders[0](x).squeeze())

        x = self.net.conv1(x)
        x = self.net.bn1(x)
        x = self.net.relu(x)
        x = self.net.maxpool(x)

        x = self.net.layer1(x)
        out_dec_res.append(self.decoders[1](x))

        x = self.net.layer2(x)
        out_dec_res.append(self.decoders[2](x))

        x = self.net.layer3(x)
        out_dec_res.append(self.decoders[3](x))

        x = self.net.layer4(x)
        out_dec_res.append(self.decoders[4](x))

        x = self.net.avgpool(x)
        x = torch.flatten(x, 1)
        out_dec_res.append(self.decoders[5](x))

        return out_dec_res

    def _forward(self, x):
        out_dec = []
        out_dec.append(self.decoders[0](torch.flatten(x, 1)))

        x = self.net.conv1(x)
        x = self.net.bn1(x)
        x = self.net.relu(x)
        x = self.net.maxpool(x)

        x = self.net.layer1(x)
        out_dec.append(self.decoders[1](torch.flatten(x, 1)))

        x = self.net.layer2(x)
        out_dec.append(self.decoders[2](torch.flatten(x, 1)))

        x = self.net.layer3(x)
        out_dec.append(self.decoders[3](torch.flatten(x, 1)))

        x = self.net.layer4(x)
        out_dec.append(self.decoders[4](torch.flatten(x, 1)))

        x = self.net.avgpool(x)
        x = torch.flatten(x, 1)
        out_dec.append(self.decoders[5](torch.flatten(x, 1)))

        return out_dec


if __name__ == "__main__":
    net = ResNet152decoders(imagenet_pt=False, num_outputs=1)
    print(net)
