from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor
import pdb

class SupConLossReg(nn.Module):
    # Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    # The proposed Weighted Supervised Contrastive Loss for the regression task
    def __init__(
        self,
        temperature: float = 0.07,
        base_temperature: float = 0.07,
        gamma1: int = 2,
        gamma2: int = 2,
        threshold: float = 0.8,
    ):
        """Weighted Supervised Contrastive Loss initialization.

        Args:
            temperature (float, optional): The hyperparameter of the weighted supervised
                contrastive loss. Defaults to 0.07.
            base_temperature (float, optional): The hyperparameter of the weighted supervised
                contrastive loss. Defaults to 0.07.
            gamma1 (int, optional): The hyperparameter of the weighted supervised contrastive
                loss. Defaults to 2.
            gamma2 (int, optional): The hyperparameter of the weighted supervised contrastive
                loss. Defaults to 2.
            threshold (float, optional): The hyperparameter of the weighted supervised
                contrastive loss. Defaults to 0.8.
        """
        super(SupConLossReg, self).__init__()
        self.temperature = temperature
        self.base_temperature = base_temperature
        self.gamma1 = gamma1
        self.gamma2 = gamma2
        self.threshold = threshold

    def forward(
        self, dynamic_t, max_dist, features: Tensor, labels: Optional[Tensor] = None, mask: Optional[Tensor] = None
    ):
        """Compute the supervised contrastive loss for model.

        Args:
            features (Tensor): hidden vector of shape [bsz, n_views, ...].
            labels (Optional[Tensor], optional): ground truth of shape [bsz].
            mask (Optional[Tensor], optional): contrastive mask of
                shape [bsz, bsz], mask_{i,j}=1 if sample j has the same
                class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = torch.device("cuda") if features.is_cuda else torch.device("cpu")

        contrast_count = 1
        contrast_feature_smiles = features[:, 1, :]
        contrast_feature_graph = features[:, 0, :]

        ############################anchor graph###############################
        anchor_feature = contrast_feature_graph
        anchor_count = 1

        pos_num = 64

        # anchor graph contrast SMILES
        batch_size = features.shape[0]

        labels = labels.contiguous().view(-1, 1)
        if labels.shape[0] != batch_size:
            raise ValueError("Num of labels does not match num of features")
        #mask = torch.eq(labels, labels.T).float().to(device)

        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature_smiles.T), self.temperature
        )

        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # calculate the distance between two samples.
        weight = torch.sqrt(
            (torch.pow(labels.repeat(1, batch_size) - labels.repeat(1, batch_size).T, 2))
        )

        #dynamic_t = torch.quantile(weight, 0.5, dim=1)
        # add a limitation to the largest threshold.
        #dynamic_t = torch.where(dynamic_t > self.threshold, self.threshold, dynamic_t.double())
        # samples with distance smaller than threshold will be considered as positive samples to the anchor.
        #mask = torch.le(weight, dynamic_t.repeat([batch_size, 1]).T).int()

        mask = torch.le(weight, dynamic_t).int()

        gamma1 = self.gamma1
        gamma2 = self.gamma2

        n_weight = (dynamic_t - weight) / (dynamic_t)
        n_weight = n_weight
        
        #calculate the weight for positive pairs.
        #d_weight = (
        #    (weight - dynamic_t.repeat([batch_size, 1]).T).T
        #    / (torch.max(weight, dim=1)[0] - dynamic_t)
        #).T + gamma2
        #d_weight = torch.exp(d_weight)
        
        d_weight = (weight - dynamic_t) / (max_dist - dynamic_t) #(max_dist - dynamic_t)
        d_weight = torch.exp(d_weight)

        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0,
        ) #* (1 - mask)

        mask = mask.fill_diagonal_(0)

        exp_logits = torch.exp(logits) * d_weight * logits_mask #+ torch.exp(logits) * mask
        #log_prob = torch.log(torch.exp(logits * n_weight * mask)) - torch.log(
        #    torch.div(exp_logits.sum(1, keepdim=True), (logits_mask).sum(1, keepdim=True))
        #)

        exp_logits = exp_logits.sum(1, keepdim=True)[logits_mask.sum(1) > 0]
        #exp_logits = torch.div(exp_logits.sum(1, keepdim=True)[logits_mask.sum(1) > 0], (logits_mask).sum(1, keepdim=True)[logits_mask.sum(1) > 0])
        #log_prob = torch.log(torch.exp(logits * n_weight))  - torch.log(exp_logits.sum(1, keepdim=True))

        log_prob = torch.exp(logits) / exp_logits #.sum(1, keepdim=True)
        log_prob = torch.log(log_prob)

        numerator = (mask * n_weight * log_prob).sum(1)
        denominator = (mask).sum(1)
        numerator = numerator[denominator > 0]
        denominator = denominator[denominator > 0]
        mean_log_prob_pos = numerator / denominator

        if torch.isnan(mean_log_prob_pos).any():
            pdb.set_trace()

        loss = -(self.temperature / self.base_temperature) * mean_log_prob_pos
        loss_graph_smiles = loss.view(anchor_count, -1).mean()

        loss = loss_graph_smiles
        return loss
