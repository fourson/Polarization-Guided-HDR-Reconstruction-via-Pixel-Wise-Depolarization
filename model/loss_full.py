from .loss_subnetwork2 import l1_and_l2_and_polarization_constrain as loss_subnetwork2
from .loss_subnetwork3 import l1_and_log_l2 as loss_subnetwork3


def loss_full(I_cat, p_pred, p, theta_pred, theta, H_pred, H, **kwargs):
    loss2_lambda = kwargs.get('loss2_lambda', 1)
    loss2 = loss_subnetwork2(I_cat, p_pred, p, theta_pred, theta) * loss2_lambda
    print('loss2:', loss2.item())

    loss3_lambda = kwargs.get('loss3_lambda', 1)
    loss3 = loss_subnetwork3(H_pred, H) * loss3_lambda
    print('loss3:', loss3.item())

    return loss2 + loss3
