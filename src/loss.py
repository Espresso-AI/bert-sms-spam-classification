import torch
import torch.nn as nn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Focal_Loss_Classification(nn.Module):

    __doc__ = """r
        The implementation of focal-loss for binary classification only.
        
        Softmax-based (num_classes=2) implementation has more stable learning graph than
        Sigmoid-based (num_classes=1) implementation.
    """

    def __init__(
            self,
            alpha: float = 0.25,
            gamma: float = 2.0,
            average: bool = True,
    ):
        super().__init__()

        self.alpha = alpha
        self.gamma = gamma
        self.average = average

    @classmethod
    def focal_loss(cls, pred, label, alpha, gamma):
        pred_t = torch.gather(pred, 1, label)
        alpha_t = torch.where(label == 1, alpha, 1 - alpha)
        weight = -1 * alpha_t * torch.pow(1 - pred_t, gamma)

        loss = weight * torch.log(pred_t)
        return loss


    def forward(self, logits, labels):
        probs = nn.Softmax(dim=-1)(logits)

        if labels.dim() != probs.dim():
            labels = labels.unsqueeze(axis=-1)

        losses = self.focal_loss(probs, labels, self.alpha, self.gamma)
        loss = losses.sum()

        if self.average:
            num_batch = losses.size(0)
            loss /= num_batch

        return loss
