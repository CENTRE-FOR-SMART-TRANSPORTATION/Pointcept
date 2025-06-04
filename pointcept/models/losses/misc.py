"""
Misc Losses

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .builder import LOSSES


@LOSSES.register_module()
class CrossEntropyLoss(nn.Module):
    def __init__(
        self,
        weight=None,
        size_average=None,
        reduce=None,
        reduction="mean",
        label_smoothing=0.0,
        loss_weight=1.0,
        ignore_index=-1,
    ):
        super(CrossEntropyLoss, self).__init__()
        weight = torch.tensor(weight).cuda() if weight is not None else None
        self.loss_weight = loss_weight
        self.loss = nn.CrossEntropyLoss(
            weight=weight,
            size_average=size_average,
            ignore_index=ignore_index,
            reduce=reduce,
            reduction=reduction,
            label_smoothing=label_smoothing,
        )

    def forward(self, pred, target):
        return self.loss(pred, target) * self.loss_weight


@LOSSES.register_module()
class SmoothCELoss(nn.Module):
    def __init__(self, smoothing_ratio=0.1):
        super(SmoothCELoss, self).__init__()
        self.smoothing_ratio = smoothing_ratio

    def forward(self, pred, target):
        eps = self.smoothing_ratio
        n_class = pred.size(1)
        one_hot = torch.zeros_like(pred).scatter(1, target.view(-1, 1), 1)
        one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
        log_prb = F.log_softmax(pred, dim=1)
        loss = -(one_hot * log_prb).total(dim=1)
        loss = loss[torch.isfinite(loss)].mean()
        return loss


@LOSSES.register_module()
class BinaryFocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=0.5, logits=True, reduce=True, loss_weight=1.0):
        """Binary Focal Loss
        <https://arxiv.org/abs/1708.02002>`
        """
        super(BinaryFocalLoss, self).__init__()
        assert 0 < alpha < 1
        self.gamma = gamma
        self.alpha = alpha
        self.logits = logits
        self.reduce = reduce
        self.loss_weight = loss_weight

    def forward(self, pred, target, **kwargs):
        """Forward function.
        Args:
            pred (torch.Tensor): The prediction with shape (N)
            target (torch.Tensor): The ground truth. If containing class
                indices, shape (N) where each value is 0≤targets[i]≤1, If containing class probabilities,
                same shape as the input.
        Returns:
            torch.Tensor: The calculated loss
        """
        if self.logits:
            bce = F.binary_cross_entropy_with_logits(pred, target, reduction="none")
        else:
            bce = F.binary_cross_entropy(pred, target, reduction="none")
        pt = torch.exp(-bce)
        alpha = self.alpha * target + (1 - self.alpha) * (1 - target)
        focal_loss = alpha * (1 - pt) ** self.gamma * bce

        if self.reduce:
            focal_loss = torch.mean(focal_loss)
        return focal_loss * self.loss_weight


@LOSSES.register_module()
class FocalLoss(nn.Module):
    def __init__(
        self, gamma=2.0, alpha=0.5, reduction="mean", loss_weight=1.0, ignore_index=-1
    ):
        """Focal Loss
        <https://arxiv.org/abs/1708.02002>`
        """
        super(FocalLoss, self).__init__()
        assert reduction in (
            "mean",
            "sum",
        ), "AssertionError: reduction should be 'mean' or 'sum'"
        assert isinstance(
            alpha, (float, list)
        ), "AssertionError: alpha should be of type float"
        assert isinstance(gamma, float), "AssertionError: gamma should be of type float"
        assert isinstance(
            loss_weight, float
        ), "AssertionError: loss_weight should be of type float"
        assert isinstance(ignore_index, int), "ignore_index must be of type int"
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.ignore_index = ignore_index

    def forward(self, pred, target, **kwargs):
        """Forward function.
        Args:
            pred (torch.Tensor): The prediction with shape (N, C) where C = number of classes.
            target (torch.Tensor): The ground truth. If containing class
                indices, shape (N) where each value is 0≤targets[i]≤C−1, If containing class probabilities,
                same shape as the input.
        Returns:
            torch.Tensor: The calculated loss
        """
        # [B, C, d_1, d_2, ..., d_k] -> [C, B, d_1, d_2, ..., d_k]
        pred = pred.transpose(0, 1)
        # [C, B, d_1, d_2, ..., d_k] -> [C, N]
        pred = pred.reshape(pred.size(0), -1)
        # [C, N] -> [N, C]
        pred = pred.transpose(0, 1).contiguous()
        # (B, d_1, d_2, ..., d_k) --> (B * d_1 * d_2 * ... * d_k,)
        target = target.view(-1).contiguous()
        assert pred.size(0) == target.size(
            0
        ), "The shape of pred doesn't match the shape of target"
        valid_mask = target != self.ignore_index
        target = target[valid_mask]
        pred = pred[valid_mask]

        if len(target) == 0:
            return 0.0

        num_classes = pred.size(1)
        target = F.one_hot(target, num_classes=num_classes)

        alpha = self.alpha
        if isinstance(alpha, list):
            alpha = pred.new_tensor(alpha)
        pred_sigmoid = pred.sigmoid()
        target = target.type_as(pred)
        one_minus_pt = (1 - pred_sigmoid) * target + pred_sigmoid * (1 - target)
        focal_weight = (alpha * target + (1 - alpha) * (1 - target)) * one_minus_pt.pow(
            self.gamma
        )

        loss = (
            F.binary_cross_entropy_with_logits(pred, target, reduction="none")
            * focal_weight
        )
        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.total()
        return self.loss_weight * loss


@LOSSES.register_module()
class DiceLoss(nn.Module):
    def __init__(self, smooth=1, exponent=2, loss_weight=1.0, ignore_index=-1):
        """DiceLoss.
        This loss is proposed in `V-Net: Fully Convolutional Neural Networks for
        Volumetric Medical Image Segmentation <https://arxiv.org/abs/1606.04797>`_.
        """
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        self.exponent = exponent
        self.loss_weight = loss_weight
        self.ignore_index = ignore_index

    def forward(self, pred, target, **kwargs):
        # [B, C, d_1, d_2, ..., d_k] -> [C, B, d_1, d_2, ..., d_k]
        pred = pred.transpose(0, 1)
        # [C, B, d_1, d_2, ..., d_k] -> [C, N]
        pred = pred.reshape(pred.size(0), -1)
        # [C, N] -> [N, C]
        pred = pred.transpose(0, 1).contiguous()
        # (B, d_1, d_2, ..., d_k) --> (B * d_1 * d_2 * ... * d_k,)
        target = target.view(-1).contiguous()
        assert pred.size(0) == target.size(
            0
        ), "The shape of pred doesn't match the shape of target"
        valid_mask = target != self.ignore_index
        target = target[valid_mask]
        pred = pred[valid_mask]

        pred = F.softmax(pred, dim=1)
        num_classes = pred.shape[1]
        target = F.one_hot(
            torch.clamp(target.long(), 0, num_classes - 1), num_classes=num_classes
        )

        total_loss = 0
        for i in range(num_classes):
            if i != self.ignore_index:
                num = torch.sum(torch.mul(pred[:, i], target[:, i])) * 2 + self.smooth
                den = (
                    torch.sum(
                        pred[:, i].pow(self.exponent) + target[:, i].pow(self.exponent)
                    )
                    + self.smooth
                )
                dice_loss = 1 - num / den
                total_loss += dice_loss
        loss = total_loss / num_classes
        return self.loss_weight * loss

        
@LOSSES.register_module()
class JaccardLoss(nn.Module):
    def __init__(self, smooth=1, loss_weight=1.0, ignore_index=-1):
        super(JaccardLoss, self).__init__()
        self.smooth = smooth
        self.loss_weight = loss_weight
        self.ignore_index = ignore_index

    def forward(self, pred, target, **kwargs):
        pred = pred.transpose(0, 1)
        pred = pred.reshape(pred.size(0), -1)
        pred = pred.transpose(0, 1).contiguous()
        target = target.view(-1).contiguous()
        assert pred.size(0) == target.size(0), "Shape of pred doesn't match shape of target"
        valid_mask = target != self.ignore_index
        target = target[valid_mask]
        pred = pred[valid_mask]

        pred = F.softmax(pred, dim=1)
        num_classes = pred.shape[1]
        target = F.one_hot(target.long(), num_classes=num_classes)

        total_loss = 0
        for i in range(num_classes):
            if i != self.ignore_index:
                intersection = torch.sum(pred[:, i] * target[:, i])
                union = torch.sum(pred[:, i] + target[:, i]) - intersection
                jaccard_loss = 1 - (intersection + self.smooth) / (union + self.smooth)
                total_loss += jaccard_loss
        loss = total_loss / num_classes
        return self.loss_weight * loss
        
        

@LOSSES.register_module()
class CompositeLoss(nn.Module):
    def __init__(
        self,
        ce_weight=1.0,
        focal_weight=1.0,
        jaccard_weight=1.0,
        ce_params=None,
        focal_params=None,
        jaccard_params=None,
    ):
        super(CompositeLoss, self).__init__()
        self.ce_weight = ce_weight
        self.focal_weight = focal_weight
        self.jaccard_weight = jaccard_weight

        # Initialize individual losses with their respective parameters
        self.ce_loss = CrossEntropyLoss(**(ce_params or {}))
        self.focal_loss = FocalLoss(**(focal_params or {}))
        self.jaccard_loss = JaccardLoss(**(jaccard_params or {}))

    def forward(self, pred, target):
        ce_loss_value = self.ce_loss(pred, target) if self.ce_weight > 0 else 0
        focal_loss_value = self.focal_loss(pred, target) if self.focal_weight > 0 else 0
        jaccard_loss_value = self.jaccard_loss(pred, target) if self.jaccard_weight > 0 else 0
        
        total_loss = (
            self.ce_weight * ce_loss_value +
            self.focal_weight * focal_loss_value +
            self.jaccard_weight * jaccard_loss_value
        )
        
        return total_loss

        

@LOSSES.register_module()
class TverskyLoss(nn.Module):
    def __init__(self, alpha=0.5, beta=0.5, smooth=1, loss_weight=1.0, ignore_index=-1):
        super(TverskyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth
        self.loss_weight = loss_weight
        self.ignore_index = ignore_index

    def forward(self, pred, target, **kwargs):
        pred = pred.transpose(0, 1)
        pred = pred.reshape(pred.size(0), -1)
        pred = pred.transpose(0, 1).contiguous()
        target = target.view(-1).contiguous()
        assert pred.size(0) == target.size(0), "Shape of pred doesn't match shape of target"
        valid_mask = target != self.ignore_index
        target = target[valid_mask]
        pred = pred[valid_mask]

        pred = F.softmax(pred, dim=1)
        num_classes = pred.shape[1]
        target = F.one_hot(target.long(), num_classes=num_classes)

        total_loss = 0
        for i in range(num_classes):
            if i != self.ignore_index:
                TP = torch.sum(pred[:, i] * target[:, i])
                FP = torch.sum(pred[:, i] * (1 - target[:, i]))
                FN = torch.sum((1 - pred[:, i]) * target[:, i])
                tversky_loss = 1 - (TP + self.smooth) / (TP + self.alpha * FP + self.beta * FN + self.smooth)
                total_loss += tversky_loss
        loss = total_loss / num_classes
        return self.loss_weight * loss


@LOSSES.register_module()
class LovaszSoftmaxLoss(nn.Module):
    def __init__(self, loss_weight=1.0, ignore_index=-1):
        super(LovaszSoftmaxLoss, self).__init__()
        self.loss_weight = loss_weight
        self.ignore_index = ignore_index

    def forward(self, pred, target):
        pred = F.softmax(pred, dim=1)
        target = target.view(-1)
        pred = pred.view(-1, pred.size(1))

        valid_mask = target != self.ignore_index
        target = target[valid_mask]
        pred = pred[valid_mask]

        def lovasz_grad(gt_sorted):
            p = len(gt_sorted)
            gts = gt_sorted.sum()
            intersection = gts - gt_sorted.float().cumsum(0)
            union = gts + (1 - gt_sorted).float().cumsum(0)
            jaccard = 1.0 - intersection / union
            jaccard[1:] = jaccard[1:] - jaccard[:-1]
            return jaccard

        total_loss = 0
        num_classes = pred.shape[1]
        for c in range(num_classes):
            fg = (target == c).float()
            errors = (fg - pred[:, c]).abs()
            errors_sorted, perm = torch.sort(errors, 0, descending=True)
            fg_sorted = fg[perm]
            loss = torch.dot(errors_sorted, lovasz_grad(fg_sorted))
            total_loss += loss
        return self.loss_weight * (total_loss / num_classes)



@LOSSES.register_module()
class AsymmetricSimilarityLoss(nn.Module):
    def __init__(self, beta=1.0, smooth=1.0, loss_weight=1.0, ignore_index=-1):
        super(AsymmetricSimilarityLoss, self).__init__()
        self.beta = beta
        self.smooth = smooth
        self.loss_weight = loss_weight
        self.ignore_index = ignore_index

    def forward(self, pred, target, **kwargs):
        pred = pred.transpose(0, 1)
        pred = pred.reshape(pred.size(0), -1)
        pred = pred.transpose(0, 1).contiguous()
        target = target.view(-1).contiguous()
        assert pred.size(0) == target.size(0), "Shape of pred doesn't match shape of target"
        valid_mask = target != self.ignore_index
        target = target[valid_mask]
        pred = pred[valid_mask]

        pred = F.softmax(pred, dim=1)
        num_classes = pred.shape[1]
        target = F.one_hot(target.long(), num_classes=num_classes).float()

        total_loss = 0
        for c in range(num_classes):
            if c != self.ignore_index:
                pred_c = pred[:, c]
                target_c = target[:, c]

                tp = torch.sum(pred_c * target_c)
                fp = torch.sum(pred_c * (1 - target_c))
                fn = torch.sum((1 - pred_c) * target_c)

                precision = tp / (tp + fp + self.smooth)
                recall = tp / (tp + fn + self.smooth)

                asl = (1 + self.beta**2) * (precision * recall) / (self.beta**2 * precision + recall + self.smooth)
                asl_loss = 1 - asl
                total_loss += asl_loss

        loss = total_loss / num_classes
        return self.loss_weight * loss
    
            
        
@LOSSES.register_module()
class CombinedFCLoss(nn.Module):
    def __init__(self, focal_loss_weight=1.0, ce_loss_weight=1.0, gamma=2.0, alpha=0.5, reduction='mean', ignore_index=-1, **kwargs):
        """
        Combined Focal and Cross Entropy Loss.
        Args:
            focal_loss_weight (float): Weight for the focal loss.
            ce_loss_weight (float): Weight for the cross entropy loss.
            gamma (float): Gamma parameter for focal loss.
            alpha (float): Alpha parameter for focal loss.
            reduction (str): Reduction method for the losses.
            ignore_index (int): Index to ignore in target.
            kwargs: Additional arguments for the individual loss functions.
        """
        super(CombinedFCLoss, self).__init__()
        
        # Initialize FocalLoss with its specific arguments
        self.focal_loss = FocalLoss(gamma=gamma, alpha=alpha, reduction=reduction, **kwargs)
        
        # Initialize CrossEntropyLoss with its specific arguments
        self.ce_loss = CrossEntropyLoss(reduction=reduction, ignore_index=ignore_index, **kwargs)
        
        self.focal_loss_weight = focal_loss_weight
        self.ce_loss_weight = ce_loss_weight

    def forward(self, pred, target):
        focal_loss = self.focal_loss(pred, target)
        ce_loss = self.ce_loss(pred, target)
        combined_loss = self.focal_loss_weight * focal_loss + self.ce_loss_weight * ce_loss
        return combined_loss


