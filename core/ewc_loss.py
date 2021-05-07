from torch import nn


class EWCLoss(nn.Module):

    def forward(self, fisher, optpar, params):
        loss = []
        for name, p in params:
            if name in fisher.keys():
                f = fisher[name]
                o = optpar[name]
                loss.append((f * (p - o) ** 2).sum())
        return sum(loss) / len(loss)
