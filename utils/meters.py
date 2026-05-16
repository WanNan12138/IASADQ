import torch


class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val, n=1):
        self.val = float(val)
        self.sum += float(val) * n
        self.count += n
        self.avg = self.sum / max(1, self.count)


def accuracy(output, target, topk=(1,)):
    if output.ndim != 2:
        return [torch.tensor(0.0, device=output.device) for _ in topk]
    maxk = min(max(topk), output.size(1))
    _, pred = output.topk(maxk, dim=1, largest=True, sorted=True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    results = []
    for k in topk:
        capped_k = min(k, output.size(1))
        correct_k = correct[:capped_k].reshape(-1).float().sum(0)
        results.append(correct_k.mul_(100.0 / target.size(0)))
    return results
