import numpy as np
import torch

PI = torch.tensor(np.pi)

COEFFICIENT_MATRIX1 = torch.tensor(
    [[1 / 2, -np.cos(2 * alpha_i) / 2, -np.sin(2 * alpha_i) / 2] for alpha_i in
     [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]], dtype=torch.float32
).cuda()

COEFFICIENT_MATRIX2 = torch.tensor(
    [[1 / 2, -np.cos(2 * alpha_i) / 2, -np.sin(2 * alpha_i) / 2] for alpha_i in [np.pi / 4, np.pi / 2, 3 * np.pi / 4]],
    dtype=torch.float32
).cuda()

COEFFICIENT_MATRIX3 = torch.tensor(
    [[1 / 2, -np.cos(2 * alpha_i) / 2, -np.sin(2 * alpha_i) / 2] for alpha_i in [0, np.pi / 2, 3 * np.pi / 4]],
    dtype=torch.float32
).cuda()

COEFFICIENT_MATRIX4 = torch.tensor(
    [[1 / 2, -np.cos(2 * alpha_i) / 2, -np.sin(2 * alpha_i) / 2] for alpha_i in [0, np.pi / 4, 3 * np.pi / 4]],
    dtype=torch.float32
).cuda()

COEFFICIENT_MATRIX5 = torch.tensor(
    [[1 / 2, -np.cos(2 * alpha_i) / 2, -np.sin(2 * alpha_i) / 2] for alpha_i in [0, np.pi / 4, np.pi / 2]],
    dtype=torch.float32
).cuda()


# assume that the shape of input the image is (N, C, H, W)
def compute_unpol_from_at_least_three_pol(pol_imgs, coefficient_matrix):
    # for more than three polarized images, use least-squares solution to solve unpol_img (with p and theta)
    A = len(pol_imgs)
    shape = pol_imgs[0].shape  # (N, C, H, W)
    N = shape[0]
    input_matrix = torch.stack(
        [pol_img.reshape(N, -1) for pol_img in pol_imgs], dim=1
    )  # (N, A, CHW)
    # shape of coefficient_matrix: (A, 3)
    # solution = torch.linalg.lstsq(coefficient_matrix.expand((N, A, 3)), input_matrix).solution  # (N, 3, CHW)
    solution = torch.linalg.pinv(coefficient_matrix.expand((N, A, 3))) @ input_matrix  # (N, 3, CHW)
    solution0 = solution[:, 0, :].reshape(shape)  # I
    solution1 = solution[:, 1, :].reshape(shape)  # I*p*cos(2*theta)
    solution2 = solution[:, 2, :].reshape(shape)  # I*p*sin(2*theta)
    unpol_img = solution0
    p = torch.clamp(torch.sqrt(solution1 ** 2 + solution2 ** 2) / (solution0 + 1e-7), min=0, max=1)  # in [0, 1]
    theta = torch.atan2(solution2, solution1) / 2  # in [-pi/2, pi/2]
    theta = (theta < 0) * np.pi + theta  # convert to [0, pi] by adding pi to negative values
    return unpol_img, p, theta


class DataItem:
    # Invoke and Execution Order:
    # get_net1_input --Subnetwork1-> set_net1_output
    # get_net2_input --Subnetwork2-> set_net2_output
    # get_net3_input --Subnetwork3-> set_net3_output
    def __init__(self, L_cat):
        self.L_cat = L_cat
        self.L1, self.L2, self.L3, self.L4 = torch.split(self.L_cat, self.L_cat.shape[1] // 4, dim=1)

        self.I_cat = None
        self.I1 = None
        self.I2 = None
        self.I3 = None
        self.I4 = None

        self.H_hat_1 = None
        self.p_hat = None
        self.theta_hat = None
        self.p_and_theta_weight = None

        self.p = None
        self.theta = None

        self.H_hat_2 = None
        self.H_weight = None

        self.H = None

        self.m_case1 = None
        self.m_case2 = None
        self.m_case2_1 = None
        self.m_case2_2 = None
        self.m_case2_3 = None
        self.m_case2_4 = None
        self.m_case3 = None
        self.m_case3_1 = None
        self.m_case3_2 = None
        self.m_case3_3 = None
        self.m_case3_4 = None
        self.m_case3_5 = None
        self.m_case3_6 = None
        self.m_case4 = None
        self.m_case4_1 = None
        self.m_case4_2 = None
        self.m_case4_3 = None
        self.m_case4_4 = None
        self.m_case5 = None

    def get_net1_input(self):
        return self.L_cat

    def set_net1_output(self, I_cat):
        self.I_cat = I_cat
        self.I1, self.I2, self.I3, self.I4 = torch.split(self.I_cat, self.I_cat.shape[1] // 4, dim=1)
        self._precompute_1()

    def get_net2_input(self):
        # note that self.theta_hat is in [0, pi], and we would like to output it in [0, 1]
        return self.I_cat, self.p_hat, self.theta_hat / PI, self.p_and_theta_weight

    def set_net2_output(self, p, theta):
        # note that theta is in [0, 1], and we would like to save it in [0, pi]
        self.p = p
        self.theta = theta * PI
        self._precompute_2()

    def get_net3_input(self):
        return self.H_hat_2, self.H_weight

    def set_net3_output(self, H):
        self.H = H

    def _precompute_1(self):
        # 1: not overexposed
        # 0: overexposed
        th = 0.95
        mask1 = (self.I1 < th).float()
        mask2 = (self.I2 < th).float()
        mask3 = (self.I3 < th).float()
        mask4 = (self.I4 < th).float()
        sum_mask = mask1 + mask2 + mask3 + mask4

        # for those pixels which are not overexposed at all of the polarization angles
        self.m_case1 = (sum_mask == 4).float()
        H_case1, p_case1, theta_case1 = compute_unpol_from_at_least_three_pol(
            [self.I1, self.I2, self.I3, self.I4],
            COEFFICIENT_MATRIX1
        )

        # for those pixels which are not overexposed at three of the polarization angles
        self.m_case2 = (sum_mask == 3).float()
        self.m_case2_1 = self.m_case2 * (1 - mask1)  # I1 overexposed, and I2, I3, I4 not overexposed
        H_case2_1, p_case2_1, theta_case2_1 = compute_unpol_from_at_least_three_pol(
            [self.I2, self.I3, self.I4],
            COEFFICIENT_MATRIX2
        )
        self.m_case2_2 = self.m_case2 * (1 - mask2)  # I2 overexposed, and I1, I3, I4 not overexposed
        H_case2_2, p_case2_2, theta_case2_2 = compute_unpol_from_at_least_three_pol(
            [self.I1, self.I3, self.I4],
            COEFFICIENT_MATRIX3
        )
        self.m_case2_3 = self.m_case2 * (1 - mask3)  # I3 overexposed, and I1, I2, I4 not overexposed
        H_case2_3, p_case2_3, theta_case2_3 = compute_unpol_from_at_least_three_pol(
            [self.I1, self.I2, self.I4],
            COEFFICIENT_MATRIX4
        )
        self.m_case2_4 = self.m_case2 * (1 - mask4)  # I4 overexposed, and I1, I2, I3 not overexposed
        H_case2_4, p_case2_4, theta_case2_4 = compute_unpol_from_at_least_three_pol(
            [self.I1, self.I2, self.I3],
            COEFFICIENT_MATRIX5
        )

        # for those pixels which are not overexposed at two of the polarization angles
        self.m_case3 = (sum_mask == 2).float()
        # in pair: I1 and I3
        self.m_case3_1 = self.m_case3 * (mask1 * mask3)
        H_case3_1 = self.I1 + self.I3
        # in pair: I2 and I4
        self.m_case3_2 = self.m_case3 * (mask2 * mask4)
        H_case3_2 = self.I2 + self.I4
        # not in pair
        self.m_case3_3 = self.m_case3 * (mask1 * mask2)  # not in pair: I1 and I2
        self.m_case3_4 = self.m_case3 * (mask1 * mask4)  # not in pair: I1 and I4
        self.m_case3_5 = self.m_case3 * (mask2 * mask3)  # not in pair: I2 and I3
        self.m_case3_6 = self.m_case3 * (mask3 * mask4)  # not in pair: I3 and I4

        # for those pixels which are not overexposed at one of the polarization angles
        self.m_case4 = (sum_mask == 1).float()
        self.m_case4_1 = self.m_case4 * mask1
        self.m_case4_2 = self.m_case4 * mask2
        self.m_case4_3 = self.m_case4 * mask3
        self.m_case4_4 = self.m_case4 * mask4

        # for those pixels which are overexposed at all of the polarization angles
        self.m_case5 = (sum_mask == 0).float()

        H_hat_1_good = self.m_case1 * H_case1 + self.m_case2_1 * H_case2_1 + self.m_case2_2 * H_case2_2 + self.m_case2_3 * H_case2_3 + self.m_case2_4 * H_case2_4 + self.m_case3_1 * H_case3_1 + self.m_case3_2 * H_case3_2
        H_hat_1_bad = (
                              self.m_case3_3 + self.m_case3_4 + self.m_case3_5 + self.m_case3_6 + self.m_case4 + self.m_case5) * H_case1  # use case1 to fill the bad regions
        self.H_hat_1 = H_hat_1_good + H_hat_1_bad

        p_hat_good = self.m_case1 * p_case1 + self.m_case2_1 * p_case2_1 + self.m_case2_2 * p_case2_2 + self.m_case2_3 * p_case2_3 + self.m_case2_4 * p_case2_4
        p_hat_bad = (self.m_case3 + self.m_case4 + self.m_case5) * p_case1  # use case1 to fill the bad regions
        self.p_hat = p_hat_good + p_hat_bad

        theta_hat_good = self.m_case1 * theta_case1 + self.m_case2_1 * theta_case2_1 + self.m_case2_2 * theta_case2_2 + self.m_case2_3 * theta_case2_3 + self.m_case2_4 * theta_case2_4
        theta_hat_bad = (
                                self.m_case3 + self.m_case4 + self.m_case5) * theta_case1  # use case1 to fill the bad regions
        self.theta_hat = theta_hat_good + theta_hat_bad
        self.p_and_theta_weight = self.m_case1 * 0.01 + self.m_case2 * 0.05 + self.m_case3 + self.m_case4 + self.m_case5  # 1: bad  0: good

    def _precompute_2(self):
        f1 = ((1 - self.p * torch.cos(2 * self.theta)) / 2).float()
        f2 = ((1 - self.p * torch.sin(2 * self.theta)) / 2).float()
        f3 = ((1 + self.p * torch.cos(2 * self.theta)) / 2).float()
        f4 = ((1 + self.p * torch.sin(2 * self.theta)) / 2).float()

        H_case3_3 = (self.I1 / (f1 + 1e-7) + self.I2 / (f2 + 1e-7)) / 2
        H_case3_4 = (self.I1 / (f1 + 1e-7) + self.I4 / (f4 + 1e-7)) / 2
        H_case3_5 = (self.I2 / (f2 + 1e-7) + self.I3 / (f3 + 1e-7)) / 2
        H_case3_6 = (self.I3 / (f3 + 1e-7) + self.I4 / (f4 + 1e-7)) / 2

        H_case4_1 = self.I1 / (f1 + 1e-7)
        H_case4_2 = self.I2 / (f2 + 1e-7)
        H_case4_3 = self.I3 / (f3 + 1e-7)
        H_case4_4 = self.I4 / (f4 + 1e-7)

        unchange_mask = 1 - self.m_case3_3 - self.m_case3_4 - self.m_case3_5 - self.m_case3_6 - self.m_case4

        self.H_hat_2 = self.m_case3_3 * H_case3_3 + self.m_case3_4 * H_case3_4 + self.m_case3_5 * H_case3_5 + self.m_case3_6 * H_case3_6 + self.m_case4_1 * H_case4_1 + self.m_case4_2 * H_case4_2 + self.m_case4_3 * H_case4_3 + self.m_case4_4 * H_case4_4 + unchange_mask * self.H_hat_1
        self.H_weight = self.m_case1 * 0.01 + self.m_case2 * 0.05 + (self.m_case3_1 + self.m_case3_2) * 0.1 + (
                self.m_case3_3 + self.m_case3_4 + self.m_case3_5 + self.m_case3_6) * 0.25 + self.m_case4 * 0.5 + self.m_case5  # 1: bad  0: good

