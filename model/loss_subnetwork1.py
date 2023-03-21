import torch
import torch.nn.functional as F


def polarization_constrain(I_cat_pred):
    # self-supervised
    th = 0.95
    split_size = I_cat_pred.shape[1] // 4
    I1_pred, I2_pred, I3_pred, I4_pred = torch.split(I_cat_pred, split_size, dim=1)
    mask1 = (I1_pred < th).float()
    mask2 = (I2_pred < th).float()
    mask3 = (I3_pred < th).float()
    mask4 = (I4_pred < th).float()
    not_overexposed_at_all_mask = ((mask1 + mask2 + mask3 + mask4) == 4).float()
    valid_mse = F.mse_loss(
        not_overexposed_at_all_mask * (I1_pred + I3_pred),
        not_overexposed_at_all_mask * (I2_pred + I4_pred),
        reduction='sum'
    ) / (not_overexposed_at_all_mask.sum() + 1e-7)

    return valid_mse


def l2_and_polarization_constrain(I_cat_pred, I_cat, **kwargs):
    l2_loss_lambda = kwargs.get('l2_loss_lambda', 1)
    l2_loss = F.mse_loss(I_cat_pred, I_cat) * l2_loss_lambda

    polarization_constrain_loss_lambda = kwargs.get('polarization_constrain_loss_lambda', 1)
    polarization_constrain_loss = polarization_constrain(I_cat_pred) * polarization_constrain_loss_lambda

    print('l2_loss:', l2_loss.item())
    print('polarization_constrain_loss:', polarization_constrain_loss.item())

    return l2_loss + polarization_constrain_loss
