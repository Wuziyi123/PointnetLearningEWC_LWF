from torch import nn


class KnowledgeDistillationLoss(nn.Module):

    def __init__(self, t):
        self._t = t
        super(KnowledgeDistillationLoss, self).__init__()

    def forward(self, pred, true):
        prob_true = nn.functional.softmax(true / self._t, dim=1)
        log_prob_pred = nn.functional.log_softmax(pred / self._t, dim=1)
        return - (prob_true * log_prob_pred).sum(dim=1).mean()
