import argparse
import math

def parse_option():
    """Parse arguments."""

    parser = argparse.ArgumentParser("argument for training")

    parser.add_argument("--classification", action="store_true", help="classification task")

    parser.add_argument("--global_feature", action="store_true", help="with global feature")
    parser.add_argument("--batch_size", type=int, default=256, help="batch_size")
    parser.add_argument("--num_workers", type=int, default=16, help="num of workers to use")
    parser.add_argument("--epochs", type=int, default=1000, help="number of training epochs")

    # optimization
    parser.add_argument("--learning_rate", type=float, default=0.05, help="learning rate")
    parser.add_argument(
        "--lr_decay_epochs", type=str, default="1000", help="where to decay lr, can be a list"
    )
    parser.add_argument(
        "--lr_decay_rate", type=float, default=0.1, help="decay rate for learning rate"
    )
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="weight decay")
    parser.add_argument("--momentum", type=float, default=0.9, help="momentum")

    parser.add_argument("--dataset", type=str, default="freesolv", help="dataset")
    parser.add_argument("--data_dir", type=str, default = "/home/UWO/ysun2443/code/trimol_dataset/", help="path to custom dataset")
    parser.add_argument("--num_tasks", type=int, default=1, help="parameter for task number")

    parser.add_argument("--mlp_layers", type=int, default=2)
    parser.add_argument("--num_gc_layers", type=int, default=3)
    parser.add_argument("--power", type=int, default="4", help="number of jump knowledge layer of gcn")
    parser.add_argument("--num_dim", type=int, default="64", help="dimension")

    # other setting
    parser.add_argument("--cosine", action="store_true", help="using cosine annealing")
    parser.add_argument("--warm", action="store_true", help="warm-up for large batch training")
    parser.add_argument("--trial", type=str, default="0", help="id for recording multiple runs")

    #xlstm configuration
    parser.add_argument("--num_blocks", type=int, default="2", help="num of blocks")
    parser.add_argument('--slstm', nargs='+', type=int, default=[0], help='Position of slstm')

    parser.add_argument('--dropout', type=float, default="0.5", help="classifier dropout")
    parser.add_argument("--mlp_layer", type=int, default=2, help="classifier mlp layer number")
    parser.add_argument("--num_experts", type=int, default=8, help="number of experts")
    parser.add_argument("--num_heads", type=int, default=8, help="number of moe heads")

    parser.add_argument("--save_model", type=int, default=0, help="save model or not")
    opt = parser.parse_args()

    opt.model_path = "./save/{}_models".format(opt.dataset)
    opt.tb_path = "./save/{}_tensorboard".format(opt.dataset)

    iterations = opt.lr_decay_epochs.split(",")
    opt.lr_decay_epochs = list([])
    for it in iterations:
        opt.lr_decay_epochs.append(int(it))

    if opt.classification:
        opt.model_name = (
            "lr_{}_bsz_{}_trial_{}_blocks_{}_slstm_{}_power_{}_dims_{}".format(
                opt.learning_rate,
                opt.batch_size,
                opt.trial,
                opt.num_blocks,
                opt.slstm,
                opt.power,
                opt.num_dim
            )
        )
    else:
        opt.model_name = "lr_{}_bsz_{}_trial_{}_blocks_{}_slstm_{}_power_{}_dims_{}".format(
            opt.learning_rate,
            opt.batch_size,
            opt.trial,
            opt.num_blocks,
            opt.slstm,
            opt.power,
            opt.num_dim
        )

    if opt.cosine:
        opt.model_name = "{}_cosine".format(opt.model_name)

    if opt.batch_size > 1024:
        opt.warm = True
    if opt.warm:
        opt.model_name = "{}_warm".format(opt.model_name)
        opt.warmup_from = 0.01
        opt.warm_epochs = 100
        if opt.cosine:
            eta_min = opt.learning_rate * (opt.lr_decay_rate**3)
            opt.warmup_to = (
                eta_min
                + (opt.learning_rate - eta_min)
                * (1 + math.cos(math.pi * opt.warm_epochs / opt.epochs))
                / 2
            )
        else:
            opt.warmup_to = opt.learning_rate_gcn

    return opt
