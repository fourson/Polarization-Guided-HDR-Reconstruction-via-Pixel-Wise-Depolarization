import torch
import torch.nn.functional as F


# LOG1P5000 = torch.log1p(torch.tensor(5000.0)).cuda()


def l1_and_log_l2(H_pred, H, **kwargs):
    l1_loss_lambda = kwargs.get('l1_loss_lambda', 1)
    l1_loss = F.l1_loss(H_pred, H) * l1_loss_lambda

    l2_loss_lambda = kwargs.get('l2_loss_lambda', 1)
    # l2_loss = F.mse_loss(torch.log1p(5000 * H_pred) / LOG1P5000, torch.log1p(5000 * H) / LOG1P5000) * l2_loss_lambda
    l2_loss = F.mse_loss(torch.log1p(H_pred), torch.log1p(H)) * l2_loss_lambda  # this compression could be enough

    print('l1_loss:', l1_loss.item())
    print('log_l2_loss:', l2_loss.item())
    return l1_loss + l2_loss
