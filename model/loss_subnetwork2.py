import torch
import torch.nn.functional as F


def polarization_constrain(I_cat, p_pred, theta_pred):
    # self-supervised
    p_cos_2_theta_pred = p_pred * torch.cos(2 * theta_pred)
    p_sin_2_theta_pred = p_pred * torch.sin(2 * theta_pred)

    split_size = I_cat.shape[1] // 4
    I1, I2, I3, I4 = torch.split(I_cat, split_size, dim=1)

    th = 0.95
    mask1 = (I1 < th).float()
    mask2 = (I2 < th).float()
    mask3 = (I3 < th).float()
    mask4 = (I4 < th).float()
    sum_mask = mask1 + mask2 + mask3 + mask4
    m = (sum_mask == 2).float()
    m12 = m * (mask1 * mask2)
    m13 = m * (mask1 * mask3)
    m14 = m * (mask1 * mask4)
    m23 = m * (mask2 * mask3)
    m24 = m * (mask2 * mask4)
    m34 = m * (mask3 * mask4)

    mse12 = F.mse_loss(m12 * I1 * (1 - p_sin_2_theta_pred), m12 * I2 * (1 - p_cos_2_theta_pred), reduction='sum')
    mse13 = F.mse_loss(m13 * I1 * (1 + p_cos_2_theta_pred), m13 * I3 * (1 - p_cos_2_theta_pred), reduction='sum')
    mse14 = F.mse_loss(m14 * I1 * (1 + p_sin_2_theta_pred), m14 * I4 * (1 - p_cos_2_theta_pred), reduction='sum')
    mse23 = F.mse_loss(m23 * I2 * (1 + p_cos_2_theta_pred), m23 * I3 * (1 - p_sin_2_theta_pred), reduction='sum')
    mse24 = F.mse_loss(m24 * I2 * (1 + p_sin_2_theta_pred), m24 * I4 * (1 - p_sin_2_theta_pred), reduction='sum')
    mse34 = F.mse_loss(m34 * I3 * (1 + p_sin_2_theta_pred), m34 * I4 * (1 + p_cos_2_theta_pred), reduction='sum')
    valid_mse = (mse12 + mse13 + mse14 + mse23 + mse24 + mse34) / ((m12 + m13 + m14 + m23 + m24 + m34).sum() + 1e-7)

    return valid_mse


def l1_and_l2_and_polarization_constrain(I_cat, p_pred, p, theta_pred, theta, **kwargs):
    p_l1_loss_lambda = kwargs.get('p_l1_loss_lambda', 1)
    p_l1_loss = F.l1_loss(p_pred, p) * p_l1_loss_lambda

    p_l2_loss_lambda = kwargs.get('p_l2_loss_lambda', 1)
    p_l2_loss = F.mse_loss(p_pred, p) * p_l2_loss_lambda

    theta_l1_loss_lambda = kwargs.get('theta_l1_loss_lambda', 1)
    theta_l1_loss = F.l1_loss(theta_pred, theta) * theta_l1_loss_lambda

    theta_l2_loss_lambda = kwargs.get('theta_l2_loss_lambda', 1)
    theta_l2_loss = F.mse_loss(theta_pred, theta) * theta_l2_loss_lambda

    polarization_constrain_loss_lambda = kwargs.get('polarization_constrain_loss_lambda', 1)
    polarization_constrain_loss = polarization_constrain(I_cat, p_pred, theta_pred) * polarization_constrain_loss_lambda

    print('p_l1_loss:', p_l1_loss.item())
    print('p_l2_loss:', p_l2_loss.item())
    print('theta_l1_loss:', theta_l1_loss.item())
    print('theta_l2_loss:', theta_l2_loss.item())
    print('polarization_constrain_loss:', polarization_constrain_loss.item())
    return p_l1_loss + p_l2_loss + theta_l1_loss + theta_l2_loss + polarization_constrain_loss
