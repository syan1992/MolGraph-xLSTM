import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
import time
import json
import warnings
from tqdm import tqdm
from copy import deepcopy
from typing import Set, Callable, Any

import pandas as pd
import numpy as np
from transformers import optimization

import torch
from torch import Tensor
from torch.nn import Module
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torch.optim import Optimizer
from torch_geometric.data import DataLoader, Data
import tensorboard_logger as tb_logger

from utils.parsing import parse_option
from utils.evaluate import Evaluator
from utils.load_dataset import PygOurDataset
from utils.util import AverageMeter, set_optimizer, calmean
from loss.loss_scl_cls import SupConLoss
from loss.loss_scl_reg import SupConLossReg
from models.deepgcn import GraphxLSTM
from models.module import MLP, MLPMoE

warnings.filterwarnings("ignore")

def set_seed(num_seed):
    np.random.seed(num_seed)
    torch.manual_seed(num_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(num_seed)
        torch.cuda.manual_seed_all(num_seed) 

#parse arguments
opt = parse_option()
args_dict = vars(opt)

def set_loader(opt: Any, dataname: str) -> Set[Data]:
    """Load dataset from opt.datas_dir.

    Args:
        opt (Any): Parsed arguments.
        dataname (str): The folder name of the dataset.

    Returns:
        Set[Data]: train/validation/test sets.
    """

    train_dataset = PygOurDataset(root=opt.data_dir, phase="train", dataname=dataname)
    test_dataset = PygOurDataset(root=opt.data_dir, phase="test", dataname=dataname)
    val_dataset = PygOurDataset(root=opt.data_dir, phase="valid", dataname=dataname)

    return train_dataset, test_dataset, val_dataset

class ContrastiveEncode(torch.nn.Module):
    def __init__(self, dim_feat: int):
        super().__init__()
        self.encode = torch.nn.Sequential(
                torch.nn.Linear(dim_feat, dim_feat),
                torch.nn.ReLU(inplace=True),
                torch.nn.Linear(dim_feat, dim_feat)
        )

    def forward(self, x):
        x = self.encode(x)
        return x

class MolPropertyPrediction(torch.nn.Module):
    def __init__(self, molxlstm: Module, dim_feat: int, opt: Any):
        super(MolPropertyPrediction, self).__init__()

        self.molxlstm = molxlstm

        self.dropout = torch.nn.Dropout(0.5)

        emb_dim = opt.num_dim * opt.power
        self.enc_joint = ContrastiveEncode(emb_dim)
        self.enc_joint_1 = ContrastiveEncode(emb_dim)
        self.enc_joint_2 = ContrastiveEncode(emb_dim)
        self.enc_joint_3 = ContrastiveEncode(emb_dim)

        classifier_num = 11
        self.classifier = torch.nn.ModuleList()
        self.classifier.extend(MLPMoE(input_feat = opt.num_dim * opt.power, dim_feat = opt.num_dim * opt.power, num_tasks=opt.num_tasks, num_experts = opt.num_experts, num_heads = opt.num_heads) for _ in range(classifier_num))

    def forward(self, input_molecule: Tensor, phase: str = "train"):
        f_out, atom_feat, fg_feat, graph_feat = self.model_graph(input_molecule)

        atom_feat_norm = F.normalize(self.enc_joint_1(atom_feat), 1)
        fg_feat_norm = F.normalize(self.enc_joint_2(fg_feat), 1)
        graph_feat_norm = F.normalize(self.enc_joint_3(graph_feat), 1)
        f_out_norm = F.normalize(self.enc_joint(f_out), 1)

        output_atom, loss_atom = self.classifier[6](atom_feat.unsqueeze(1))
        output_fg, loss_fg = self.classifier[7](fg_feat.unsqueeze(1))
        output_gnn, loss_gnn = self.classifier[8](graph_feat.unsqueeze(1))
        output_final, loss_out = self.classifier[9](f_out.unsqueeze(1))
        loss_auc = (loss_atom + loss_fg + loss_out + loss_gnn)

        if phase == "train":
            return (
                output_final,
                output_atom,
                output_fg,
                output_gnn,
                atom_feat_norm,
                fg_feat_norm,
                graph_feat_norm,
                f_out_norm,
                loss_auc
            )
        else:
            return (
                output_final,
                output_atom,
                output_fg,
                output_gnn,
                atom_feat_norm,
                fg_feat_norm,
                graph_feat_norm,
                f_out_norm
            )

def set_model(opt: Any):
    """Initialization of the model and loss functions.

    Args:
        opt (Any): Parsed arguments.

    Returns:
        Return the model and the loss functions.
    """

    molgraph_xlstm = GraphxLSTM(opt)
    model = MolPropertyPrediction(molgraph_xlstm, opt)

    if opt.classification:
        criterion_task = torch.nn.BCEWithLogitsLoss()
    else:
        criterion_task = torch.nn.MSELoss()

    if torch.cuda.is_available():
        model = model.cuda()
        cudnn.benchmark = False

    return model, criterion_task

def train(
    train_loader: Any,
    model: torch.nn.Sequential,
    criterion_task: Callable,
    optimizer: Optimizer,
    scheduler: Any,
    opt: Any,
    mu: int = 0,
    std: int = 0,
    dynamic_t: int = 0, 
    max_dist: int = 0,
    epoch: int = 0
):
    """One epoch training.

    Args:
        train_dataset (Set[Data]): Train set.
        model (torch.nn.Sequential): Model
        criterion_task (Callable): Task loss function
        optimizer (Optimizer): Optimizer
        opt (Any): Parsed arguments
        mu (int, optional): Mean value of the train set for the regression task. Defaults to 0.
        std (int, optional): Standard deviation of the train set for the regression task.
            Defaults to 0.

    Returns:
        Losses.
    """
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()

    losses_task = AverageMeter()
    losses_scl = AverageMeter()
    losses = AverageMeter()

    end = time.time()
    for _, batch in enumerate(tqdm(train_loader, desc="Iteration")):
        batch = batch.to("cuda")
        data_time.update(time.time() - end)

        bsz = batch.y.shape[0]
        if not opt.classification:
            labels = (batch.y - mu) / std
        else:
            labels = batch.y
        # compute loss
        (
            output_final,
            output_atom,
            output_fg,
            output_gnn,
            atom_feat,
            fg_feat,
            graph_feat,
            f_out,
            loss_auc
        ) = model(batch, opt)


        if (opt.classification):
            criterion_cl = SupConLoss()
        else:
            criterion_cl = SupConLossReg(gamma1 = 1, gamma2 = 0)

        loss_cl_tmp = 0

        features_graph_1 = torch.cat([f_out.unsqueeze(1), f_out.unsqueeze(1)], dim=1)
        features_graph_2 = torch.cat([atom_feat.unsqueeze(1), atom_feat.unsqueeze(1)], dim=1)
        features_graph_3 = torch.cat([fg_feat.unsqueeze(1), fg_feat.unsqueeze(1)], dim=1)
        features_graph_4 = torch.cat([graph_feat.unsqueeze(1), graph_feat.unsqueeze(1)], dim=1)

        num_labels = 0
        loss_task_tmp_final = 0
        loss_task_tmp_atom = 0
        loss_task_tmp_fg = 0
        loss_task_tmp_gnn = 0

        for i in range(labels.shape[1]):
            is_labeled = batch.y[:, i] == batch.y[:, i]

            loss_task_final = criterion_task(
                    output_final[is_labeled, i].squeeze(), labels[is_labeled, i].squeeze()
            )

            loss_task_atom = criterion_task(
                    output_atom[is_labeled, i].squeeze(), labels[is_labeled, i].squeeze()
            )

            loss_task_fg = criterion_task(
                    output_fg[is_labeled, i].squeeze(), labels[is_labeled, i].squeeze()
            )

            loss_task_gnn = criterion_task(
                    output_gnn[is_labeled, i].squeeze(), labels[is_labeled, i].squeeze()
            )

            loss_task_tmp_final = loss_task_tmp_final + loss_task_final
            loss_task_tmp_atom = loss_task_tmp_atom + loss_task_atom
            loss_task_tmp_fg = loss_task_tmp_fg + loss_task_fg
            loss_task_tmp_gnn = loss_task_tmp_gnn + loss_task_gnn

            if labels[is_labeled, i].sum() == 0:
                continue

            num_labels = num_labels + 1

            if opt.classification:
                loss_cl_tmp = (loss_cl_tmp +
                               (criterion_cl(features_graph_1[is_labeled, :], labels[is_labeled, i].squeeze()) +
                                criterion_cl(features_graph_2[is_labeled, :], labels[is_labeled, i].squeeze()) +
                                criterion_cl(features_graph_3[is_labeled, :], labels[is_labeled, i].squeeze()) +
                                criterion_cl(features_graph_4[is_labeled, :], labels[is_labeled, i].squeeze())))
            else:
                loss_cl_tmp = (loss_cl_tmp +
                               criterion_cl(dynamic_t, max_dist, features_graph_1[is_labeled, :], labels[is_labeled, i].squeeze()) +
                               criterion_cl(dynamic_t, max_dist, features_graph_2[is_labeled, :], labels[is_labeled, i].squeeze()) +
                               criterion_cl(dynamic_t, max_dist, features_graph_3[is_labeled, :], labels[is_labeled, i].squeeze()) +
                               criterion_cl(dynamic_t, max_dist, features_graph_4[is_labeled, :], labels[is_labeled, i].squeeze()))

        if num_labels==0:
            loss_cl = loss_cl_tmp / labels.shape[1]
        else:
            loss_cl = loss_cl_tmp / num_labels

        loss_task = (loss_task_tmp_final + loss_task_tmp_atom + loss_task_tmp_fg + loss_task_tmp_gnn) / labels.shape[1]

        if opt.classification:
            loss = loss_task + loss_cl + loss_auc
        else:
            loss = loss_task + loss_cl + loss_auc

        # update metric
        losses_task.update(loss_task.item(), bsz)
        losses_scl.update(loss_cl.item(), bsz)
        losses.update(loss.item(), bsz)

        optimizer[0].zero_grad()
        loss.backward()
        optimizer[0].step()
        scheduler.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
    return losses_task.avg, losses_scl.avg, losses.avg


def validation(
    dataset: Set[Data],
    model: torch.nn.Sequential,
    opt: Any,
    mu: int = 0,
    std: int = 0,
    save_feature: int = 0,
    epoch: int =0
):
    """Calculate performance metrics.

    Args:
        dataset (Set[Data]): A dataset.
        model (torch.nn.Sequential): Model.
        opt (Any): Parsed arguments.
        mu (int, optional): Mean value of the train set for the regression task.
            Defaults to 0.
        std (int, optional): Standard deviation of the train set for the regression task.
            Defaults to 0.
        save_feature (int, optional): Whether save the learned features or not.
            Defaults to 0.

    Returns:
        auroc or rmse value.
    """
    model.eval()

    if opt.classification:
        evaluator = Evaluator(name=opt.dataset, num_tasks=opt.num_tasks, eval_metric="rocauc")
    else:
        evaluator = Evaluator(name=opt.dataset, num_tasks=opt.num_tasks, eval_metric="rmse")
    data_loader = DataLoader(
        dataset, batch_size=opt.batch_size, shuffle=False, follow_batch = ['fg_x', 'atom2fg_list']
    )

    with torch.no_grad():
        y_true = []
        y_pred = []
        for _, batch in enumerate(tqdm(data_loader, desc="Iteration")):
            batch = batch.to("cuda")
            (
                output_final,
                output_atom,
                output_fg,
                output_gnn,
                atom_feat,
                fg_feat, 
                graph_feat,
                f_out
            ) = model(batch, opt, "valid")

            if not opt.classification:
                output = (output_final) * std + mu

            if opt.classification:
                sigmoid = torch.nn.Sigmoid()
                output = sigmoid(output_final)

            y_true.append(batch.y.detach().cpu())
            y_pred.append(output.detach().cpu())

        y_true = torch.cat(y_true, dim=0).squeeze().unsqueeze(1).numpy()
        if opt.num_tasks > 1:
            y_pred = np.concatenate(y_pred)
            input_dict = {"y_true": y_true.squeeze(), "y_pred": y_pred.squeeze()}
        else:
            y_pred = np.expand_dims(np.concatenate(y_pred), 1)
            input_dict = {
                "y_true": np.expand_dims(y_true.squeeze(), 1),
                "y_pred": np.expand_dims(y_pred.squeeze(), 1),
            }

        if opt.classification:
            eval_result = evaluator.eval(input_dict)["rocauc"]
        else:
            eval_result = evaluator.eval(input_dict)["rmse"]

    return y_true, y_pred, eval_result


def main():

    for dataname in [opt.dataset + "_3", opt.dataset + "_1", opt.dataset + "_2"]:
        set_seed(10)
        # build data loader
        train_dataset, test_dataset, val_dataset = set_loader(opt, dataname)

        if opt.classification:
            mu, std, dynamic_t, max_dist = 0, 0, 0, 0
        else:
            mu, std, dynamic_t, max_dist = calmean(train_dataset)

        # build model and criterion
        model, criterion_task = set_model(opt)

        # build optimizer
        optimizer = set_optimizer(opt.learning_rate, opt.weight_decay, model)

        model_name = "{}_{}".format(opt.model_name, dataname)

        # save folder
        opt.tb_folder = os.path.join(opt.tb_path, model_name)
        if not os.path.isdir(opt.tb_folder):
            os.makedirs(opt.tb_folder)

        opt.save_folder = os.path.join(opt.model_path, model_name)
        if not os.path.isdir(opt.save_folder):
            os.makedirs(opt.save_folder)

        with open(opt.save_folder + "//runtime_params.json", "w") as f:
                json.dump(args_dict, f, indent=4)

        # tensorboard
        logger = tb_logger.Logger(logdir=opt.tb_folder, flush_secs=2)

        if opt.classification:
            best_acc = 0
        else:
            best_acc = float('inf')

        best_model = model
        best_epoch = 0

        train_loader = DataLoader(train_dataset, batch_size=opt.batch_size, drop_last=True, shuffle=True, follow_batch=['fg_x', 'atom2fg_list'])
        num_training_steps =  len(train_loader) * opt.epochs
        num_warmup_steps = int(num_training_steps * 0.1)
        scheduler = optimization.get_linear_schedule_with_warmup(optimizer[0], num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps)

        # training routine
        for epoch in range(opt.epochs):
            torch.cuda.empty_cache()
            # train for one epoch
            time1 = time.time()
            loss_task, loss_scl, loss = train(
                train_loader,
                model,
                criterion_task,
                optimizer,
                scheduler,
                opt,
                mu,
                std,
                dynamic_t, 
                max_dist,
                epoch
            )
            time2 = time.time()
            print("epoch {}, total time {:.2f}".format(epoch, time2 - time1))

            _, _, acc = validation(val_dataset, model, opt, mu, std, 0, epoch)

            # tensorboard logger
            logger.log_value("task loss", loss_task, epoch)
            logger.log_value("supervised contrastive loss", loss_scl, epoch)
            logger.log_value("overall loss", loss, epoch)
            logger.log_value("validation auroc/rmse", acc, epoch)
            logger.log_value("learning rate", optimizer[0].state_dict()['param_groups'][0]['lr'], epoch)

            if opt.classification:
                if acc > best_acc:
                    best_acc = acc
                    best_model = deepcopy(model).cpu()
                    best_epoch = epoch
                    _, _, test_acc = validation(test_dataset, model, opt, mu, std, 0, epoch)
                    logger.log_value("test auroc", test_acc, epoch)
                    print("test auroc:{}".format(test_acc))
                print("val auroc:{}".format(acc))
                
            else:
                if acc < best_acc:
                    best_acc = acc
                    best_model = deepcopy(model).cpu()
                    best_epoch = epoch
                    _, _, test_acc = validation(test_dataset, model, opt, mu, std, 0, epoch)
                    logger.log_value("test rmse", test_acc, epoch)
                    print("test rmse:{}".format(test_acc))
                print("val rmse:{}".format(acc))

        y_true, y_pred, test_acc = validation(test_dataset, best_model.cuda(), opt, mu, std, 1, epoch-1)
        _, _, val_acc = validation(val_dataset, best_model.cuda(), opt, mu, std, 0, epoch-1)

        if opt.save_model:
            torch.save(best_model.state_dict(), os.path.join(opt.save_folder, "model.pth"))

        df = pd.DataFrame(y_true.squeeze())
        df.to_csv(opt.save_folder+'//true_result.csv')
        df = pd.DataFrame(y_pred.squeeze())
        df.to_csv(opt.save_folder+'//pred_result.csv')

        save_file = os.path.join(opt.save_folder, "result_pre.txt")
        txtFile = open(save_file, "w")
        txtFile.write("validation:" + str(val_acc) + "\n")
        txtFile.write("test:" + str(test_acc) + "\n")
        txtFile.write("best epoch:" + str(best_epoch) + "\n")
        txtFile.close()
        print("Val Result:{}".format(val_acc))
        print("Test Result:{}".format(test_acc))

if __name__ == "__main__":
    main()
