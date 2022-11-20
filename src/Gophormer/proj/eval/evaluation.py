import torch


def eval_classification(pred, target, n_class):
    '''
    Returns macro-f1 and micro-f1 score
    Args:
        pred:
        target:
        n_class:

    Returns:
        ma_f1,mi_f1: numpy values of macro-f1 and micro-f1 scores.
    '''

    def true_positive(pred, target, n_class):
        return torch.tensor([((pred == i) & (target == i)).sum()
                             for i in range(n_class)])

    def false_positive(pred, target, n_class):
        return torch.tensor([((pred == i) & (target != i)).sum()
                             for i in range(n_class)])

    def false_negative(pred, target, n_class):
        return torch.tensor([((pred != i) & (target == i)).sum()
                             for i in range(n_class)])

    def precision(tp, fp):
        res = tp / (tp + fp)
        res[torch.isnan(res)] = 0
        return res

    def recall(tp, fn):
        res = tp / (tp + fn)
        res[torch.isnan(res)] = 0
        return res

    def f1_score(prec, rec):
        f1_score = 2 * (prec * rec) / (prec + rec)
        f1_score[torch.isnan(f1_score)] = 0
        return f1_score

    def cal_maf1(tp, fp, fn):
        prec = precision(tp, fp)
        rec = recall(tp, fn)
        ma_f1 = f1_score(prec, rec)
        return torch.mean(ma_f1).cpu().numpy()

    def cal_mif1(tp, fp, fn):
        gl_tp, gl_fp, gl_fn = torch.sum(tp), torch.sum(fp), torch.sum(fn)
        gl_prec = precision(gl_tp, gl_fp)
        gl_rec = recall(gl_tp, gl_fn)
        mi_f1 = f1_score(gl_prec, gl_rec)
        return mi_f1.cpu().numpy()

    def accuracy(pred, target):
        return (pred == target).sum().item() / target.numel()

    tp = true_positive(pred, target, n_class).to(torch.float)
    fn = false_negative(pred, target, n_class).to(torch.float)
    fp = false_positive(pred, target, n_class).to(torch.float)

    ma_f1, mi_f1 = cal_maf1(tp, fp, fn), cal_mif1(tp, fp, fn)
    acc = accuracy(pred, target)
    return acc, ma_f1.item(), mi_f1.item()


def eval_logits(logits, target_x, target_y):
    pred_y = torch.argmax(logits[target_x], dim=1)
    return eval_classification(pred_y, target_y, n_class=logits.shape[1])
