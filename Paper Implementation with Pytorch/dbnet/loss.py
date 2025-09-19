import torch
import torch.nn as nn

class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        pred = pred.contiguous().view(-1)
        target = target.contiguous().view(-1)
        intersection = (pred * target).sum()
        dice = (2. * intersection + self.smooth) / (pred.sum() + target.sum() + self.smooth)
        return 1 - dice

class MaskedL1Loss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, target, mask):
        mask = mask.float()
        loss = torch.abs(pred - target) * mask
        return loss.sum() / (mask.sum() + 1e-6)

class DBLoss(nn.Module):
    def __init__(self, alpha=1.0, beta=10.0, ohem_ratio=3.0):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.bce_loss = nn.BCELoss(reduction='none')
        self.dice_loss = DiceLoss()
        self.l1_loss = MaskedL1Loss()
        self.ohem_ratio = ohem_ratio

    def forward(self, pred, target):
        prob_map_pred, thresh_map_pred, binary_map_pred = pred
        prob_map_gt, thresh_map_gt, mask_gt = target

        # OHEM for BCE loss
        bce = self.bce_loss(prob_map_pred, prob_map_gt)
        positive_mask = (prob_map_gt > 0).float()
        negative_mask = 1 - positive_mask
        
        num_positive = int(positive_mask.sum())
        num_negative = min(int(negative_mask.sum()), int(num_positive * self.ohem_ratio))

        positive_loss = (bce * positive_mask).sum() / (num_positive + 1e-6)
        negative_loss_all = bce * negative_mask
        negative_loss, _ = torch.topk(negative_loss_all.view(-1), num_negative)
        
        loss_prob = positive_loss + negative_loss.mean()
        loss_binary = self.dice_loss(binary_map_pred, prob_map_gt)
        loss_thresh = self.l1_loss(thresh_map_pred, thresh_map_gt, mask_gt)

        return loss_prob + self.alpha * loss_binary + self.beta * loss_thresh
