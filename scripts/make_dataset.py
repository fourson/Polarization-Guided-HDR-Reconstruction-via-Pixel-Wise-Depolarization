import os
import random
import fnmatch

import numpy as np
import cv2


def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def ensure_dirs(base_dir, dir_names):
    for dir_name in dir_names:
        ensure_dir(os.path.join(base_dir, dir_name))


def save_npy(f, save_dir, f_name):
    out_path = os.path.join(save_dir, f_name + '.npy')
    np.save(out_path, f)


def do_some_rotation(E, p, theta, img_name):
    td = [np.flip(E, axis=0), np.flip(p, axis=0), np.flip(theta, axis=0), img_name + '_td']
    lr = [np.flip(E, axis=1), np.flip(p, axis=1), np.flip(theta, axis=1), img_name + '_lr']
    a_90 = [np.rot90(E, 1), np.rot90(p, 1), np.rot90(theta, 1), img_name + '_a90']
    a_180 = [np.rot90(E, 2), np.rot90(p, 2), np.rot90(theta, 2), img_name + '_a180']
    a_270 = [np.rot90(E, 3), np.rot90(p, 3), np.rot90(theta, 3), img_name + '_a270']
    return td, lr, a_90, a_180, a_270


def data_augmentation_train(E, p, theta, img_name, no_rotation=False):
    # for data augmentation
    items = []
    # input size: 1024*1224
    # first we crop to 1024*1024 in the middle
    E = E[:, 100:-100, :]
    p = p[:, 100:-100, :]
    theta = theta[:, 100:-100, :]

    # do not scale, random crop to 256*256
    offset_list = zip(random.sample(range(0, 768), 6), random.sample(range(0, 768), 6))
    for index, offset in enumerate(offset_list, 1):
        H_offset, W_offset = offset
        E_crop = E[H_offset: H_offset + 256, W_offset: W_offset + 256, :]
        p_crop = p[H_offset: H_offset + 256, W_offset: W_offset + 256, :]
        theta_crop = theta[H_offset: H_offset + 256, W_offset: W_offset + 256, :]
        items.append([E_crop, p_crop, theta_crop, img_name + '_crop' + str(index)])
        if not no_rotation:
            for rot_item in do_some_rotation(E_crop, p_crop, theta_crop, img_name + '_crop' + str(index)):
                items.append(rot_item)

    # scale to 512*512, then crop to 256*256
    E_half = cv2.resize(E, (512, 512), interpolation=cv2.INTER_LINEAR)
    p_half = cv2.resize(p, (512, 512), interpolation=cv2.INTER_LINEAR)
    theta_half = cv2.resize(theta, (512, 512), interpolation=cv2.INTER_LINEAR)

    E_half_tl = E_half[0:256, 0:256, :]
    p_half_tl = p_half[0:256, 0:256, :]
    theta_half_tl = theta_half[0:256, 0:256, :]
    items.append([E_half_tl, p_half_tl, theta_half_tl, img_name + '_half_tl'])
    if not no_rotation:
        for rot_item in do_some_rotation(E_half_tl, p_half_tl, theta_half_tl, img_name + '_half_tl'):
            items.append(rot_item)

    E_half_tr = E_half[0:256, 256:512, :]
    p_half_tr = p_half[0:256, 256:512, :]
    theta_half_tr = theta_half[0:256, 256:512, :]
    items.append([E_half_tr, p_half_tr, theta_half_tr, img_name + '_half_tr'])
    if not no_rotation:
        for rot_item in do_some_rotation(E_half_tr, p_half_tr, theta_half_tr, img_name + '_half_tr'):
            items.append(rot_item)

    E_half_bl = E_half[256:512, 0:256, :]
    p_half_bl = p_half[256:512, 0:256, :]
    theta_half_bl = theta_half[256:512, 0:256, :]
    items.append([E_half_bl, p_half_bl, theta_half_bl, img_name + '_half_bl'])
    if not no_rotation:
        for rot_item in do_some_rotation(E_half_bl, p_half_bl, theta_half_bl, img_name + '_half_bl'):
            items.append(rot_item)

    E_half_br = E_half[256:512, 256:512, :]
    p_half_br = p_half[256:512, 256:512, :]
    theta_half_br = theta_half[256:512, 256:512, :]
    items.append([E_half_br, p_half_br, theta_half_br, img_name + '_half_br'])
    if not no_rotation:
        for rot_item in do_some_rotation(E_half_br, p_half_br, theta_half_br, img_name + '_half_br'):
            items.append(rot_item)

    E_half_mid = E_half[128:384, 128:384, :]
    p_half_mid = p_half[128:384, 128:384, :]
    theta_half_mid = theta_half[128:384, 128:384, :]
    items.append([E_half_mid, p_half_mid, theta_half_mid, img_name + '_half_mid'])
    if not no_rotation:
        for rot_item in do_some_rotation(E_half_mid, p_half_mid, theta_half_mid, img_name + '_half_mid'):
            items.append(rot_item)

    # scale to 256
    E_quarter = cv2.resize(E, (256, 256), interpolation=cv2.INTER_LINEAR)
    p_quarter = cv2.resize(p, (256, 256), interpolation=cv2.INTER_LINEAR)
    theta_quarter = cv2.resize(theta, (256, 256), interpolation=cv2.INTER_LINEAR)
    items.append([E_quarter, p_quarter, theta_quarter, img_name + '_quarter'])
    if not no_rotation:
        for rot_item in do_some_rotation(E_quarter, p_quarter, theta_quarter, img_name + '_quarter'):
            items.append(rot_item)

    return items


def data_augmentation_test(E, p, theta, img_name, no_rotation=True):
    # for data augmentation
    items = []
    # input size: 1024*1224
    # first we crop to 1024*1024 in the middle
    E = E[:, 100:-100, :]
    p = p[:, 100:-100, :]
    theta = theta[:, 100:-100, :]

    # crop to 512*512
    E_tl = E[0:512, 0:512, :]
    p_tl = p[0:512, 0:512, :]
    theta_tl = theta[0:512, 0:512, :]
    items.append([E_tl, p_tl, theta_tl, img_name + '_tl'])
    if not no_rotation:
        for rot_item in do_some_rotation(E_tl, p_tl, theta_tl, img_name + '_tl'):
            items.append(rot_item)

    E_tr = E[0:512, 512:1024, :]
    p_tr = p[0:512, 512:1024, :]
    theta_tr = theta[0:512, 512:1024, :]
    items.append([E_tr, p_tr, theta_tr, img_name + '_tr'])
    if not no_rotation:
        for rot_item in do_some_rotation(E_tr, p_tr, theta_tr, img_name + '_tr'):
            items.append(rot_item)

    E_bl = E[512:1024, 0:512, :]
    p_bl = p[512:1024, 0:512, :]
    theta_bl = theta[512:1024, 0:512, :]
    items.append([E_bl, p_bl, theta_bl, img_name + '_bl'])
    if not no_rotation:
        for rot_item in do_some_rotation(E_bl, p_bl, theta_bl, img_name + '_bl'):
            items.append(rot_item)

    E_br = E[512:1024, 512:1024, :]
    p_br = p[512:1024, 512:1024, :]
    theta_br = theta[512:1024, 512:1024, :]
    items.append([E_br, p_br, theta_br, img_name + '_br'])
    if not no_rotation:
        for rot_item in do_some_rotation(E_br, p_br, theta_br, img_name + '_br'):
            items.append(rot_item)

    E_mid = E[256:768, 256:768, :]
    p_mid = p[256:768, 256:768, :]
    theta_mid = theta[256:768, 256:768, :]
    items.append([E_mid, p_mid, theta_mid, img_name + '_mid'])
    if not no_rotation:
        for rot_item in do_some_rotation(E_mid, p_mid, theta_mid, img_name + '_mid'):
            items.append(rot_item)

    # scale to 512*512
    E_half = cv2.resize(E, (512, 512), interpolation=cv2.INTER_LINEAR)
    p_half = cv2.resize(p, (512, 512), interpolation=cv2.INTER_LINEAR)
    theta_half = cv2.resize(theta, (512, 512), interpolation=cv2.INTER_LINEAR)
    items.append([E_half, p_half, theta_half, img_name + '_half'])
    if not no_rotation:
        for rot_item in do_some_rotation(E_half, p_half, theta_half, img_name + '_half'):
            items.append(rot_item)

    return items


def compute_unpol_from_at_least_three_pol(pol_imgs, alpha):
    A = len(pol_imgs)
    assert A == len(alpha)
    assert A >= 3
    # for more than three polarized images, use least-squares solution to solve unpol_img (with p and theta)
    shape = pol_imgs[0].shape
    coefficient_matrix = np.array(
        [[1 / 2, -np.cos(2 * alpha_i) / 2, -np.sin(2 * alpha_i) / 2] for alpha_i in alpha], dtype=np.float32
    )  # (A, 3)
    input_matrix = np.array(
        [pol_img.reshape(-1) for pol_img in pol_imgs], dtype=np.float32
    )  # (A, pixel_num)
    solution = np.linalg.lstsq(coefficient_matrix, input_matrix)[0]  # (3, pixel_num)
    solution0 = solution[0].reshape(shape)  # I
    solution1 = solution[1].reshape(shape)  # I*p*cos(2*theta)
    solution2 = solution[2].reshape(shape)  # I*p*sin(2*theta)
    unpol_img = solution0
    p = np.clip(np.sqrt(solution1 ** 2 + solution2 ** 2) / (solution0 + 1e-7), a_min=0, a_max=1)  # in [0, 1]
    theta = np.arctan2(solution2, solution1) / 2  # in [-pi/2, pi/2]
    theta = (theta < 0) * np.pi + theta  # convert to [0, pi] by adding pi to negative values
    theta = theta.astype(np.float32)
    return unpol_img, p, theta


class DatasetMaker:
    """
        find an appropriate exposure time t
        to ensure that the bad_pixel rate is in [bad_pixel_rate_lbound, bad_pixel_rate_ubound]
    """

    def __init__(self, E, p, theta, bad_pixel_rate_ubound, bad_pixel_rate_lbound, iter_max=15):
        self.E = (E - np.min(E)) / (np.max(E) - np.min(E))
        self.p = p
        self.theta = theta

        self.f1 = np.float32((1 - self.p * np.cos(2 * self.theta)) / 2)
        self.f2 = np.float32((1 - self.p * np.sin(2 * self.theta)) / 2)
        self.f3 = np.float32((1 + self.p * np.cos(2 * self.theta)) / 2)
        self.f4 = np.float32((1 + self.p * np.sin(2 * self.theta)) / 2)
        self.fm = max([np.max(self.f1), np.max(self.f2), np.max(self.f3), np.max(self.f4)])

        self.E1 = self.f1 * self.E
        self.E2 = self.f2 * self.E
        self.E3 = self.f3 * self.E
        self.E4 = self.f4 * self.E

        self.mask1 = None
        self.mask2 = None
        self.mask3 = None
        self.mask4 = None

        # we hope the bad_pixel rate is between [bad_pixel_rate_lbound, bad_pixel_rate_ubound]
        self.bad_pixel_rate_ubound = bad_pixel_rate_ubound
        self.bad_pixel_rate_lbound = bad_pixel_rate_lbound

        # initialize the search space
        self.t_ubound = 100 / self.fm
        self.t_lbound = 1 / self.fm
        self.t = np.random.random() * (self.t_ubound - self.t_lbound) + self.t_lbound

        # set the maximum number of iterations
        self.iter_max = iter_max

        self.success = True

        # for log
        self.iter_cnt = 0
        self.final_bad_pixel_rate = 0
        self.exposure = 0
        self.log = "iter_cnt exceeds the limit({})\n".format(self.iter_max)

        self.H = None
        self.I = None

        self.I1 = None
        self.I2 = None
        self.I3 = None
        self.I4 = None
        self.I_cat = None

        self.L1 = None
        self.L2 = None
        self.L3 = None
        self.L4 = None
        self.L_cat = None

        self.H_hat_1 = None
        self.p_hat = None
        self.theta_hat = None
        self.p_and_theta_weight = None

        self.H_hat_2 = None
        self.H_weight = None

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

    def _bad_pixel_rate(self):
        # 1: not overexposed
        # 0: overexposed
        th = 0.95
        self.mask1 = np.float32((self.E1 * self.t) < th)
        self.mask2 = np.float32((self.E2 * self.t) < th)
        self.mask3 = np.float32((self.E3 * self.t) < th)
        self.mask4 = np.float32((self.E4 * self.t) < th)
        mask_bad_pixel = np.float32((self.mask1 + self.mask3) <= 1) * np.float32((self.mask2 + self.mask4) <= 1)
        mask_bad_pixel_to_single_channel = (np.sum(mask_bad_pixel, axis=2) >= 1)
        return np.sum(mask_bad_pixel_to_single_channel) / mask_bad_pixel_to_single_channel.size

    def _find_t(self):
        # use binary-search to find an appropriate t
        bad_pixel_rate = self._bad_pixel_rate()
        if bad_pixel_rate > self.bad_pixel_rate_ubound:
            t_lb, t_ub = self.t_lbound, self.t
        elif bad_pixel_rate < self.bad_pixel_rate_lbound:
            t_lb, t_ub = self.t, self.t_ubound
        else:
            t_lb, t_ub = 0, 0
        iter_cnt = 0
        while t_lb < t_ub:
            if iter_cnt > self.iter_max:
                # exceeds the iteration limit
                self.success = False
                break
            iter_cnt += 1
            self.t = (t_lb + t_ub) / 2
            bad_pixel_rate = self._bad_pixel_rate()
            if bad_pixel_rate > self.bad_pixel_rate_ubound:
                t_ub = self.t
            elif bad_pixel_rate < self.bad_pixel_rate_lbound:
                t_lb = self.t
            else:
                break
        self.iter_cnt = iter_cnt
        self.final_bad_pixel_rate = bad_pixel_rate

    def _precompute_1(self):
        # 1: not overexposed
        # 0: overexposed
        sum_mask = self.mask1 + self.mask2 + self.mask3 + self.mask4

        # for those pixels which are not overexposed at all of the polarization angles
        self.m_case1 = np.float32(sum_mask == 4)
        H_hat_case1, p_hat_case1, theta_hat_case1 = compute_unpol_from_at_least_three_pol(
            [self.I1, self.I2, self.I3, self.I4],
            [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]
        )

        # for those pixels which are not overexposed at three of the polarization angles
        self.m_case2 = np.float32(sum_mask == 3)
        self.m_case2_1 = self.m_case2 * (1 - self.mask1)  # I1 overexposed, and I2, I3, I4 not overexposed
        H_hat_case2_1, p_hat_case2_1, theta_hat_case2_1 = compute_unpol_from_at_least_three_pol(
            [self.I2, self.I3, self.I4],
            [np.pi / 4, np.pi / 2, 3 * np.pi / 4]
        )
        self.m_case2_2 = self.m_case2 * (1 - self.mask2)  # I2 overexposed, and I1, I3, I4 not overexposed
        H_hat_case2_2, p_hat_case2_2, theta_hat_case2_2 = compute_unpol_from_at_least_three_pol(
            [self.I1, self.I3, self.I4],
            [0, np.pi / 2, 3 * np.pi / 4]
        )
        self.m_case2_3 = self.m_case2 * (1 - self.mask3)  # I3 overexposed, and I1, I2, I4 not overexposed
        H_hat_case2_3, p_hat_case2_3, theta_hat_case2_3 = compute_unpol_from_at_least_three_pol(
            [self.I1, self.I2, self.I4],
            [0, np.pi / 4, 3 * np.pi / 4]
        )
        self.m_case2_4 = self.m_case2 * (1 - self.mask4)  # I4 overexposed, and I1, I2, I3 not overexposed
        H_hat_case2_4, p_hat_case2_4, theta_hat_case2_4 = compute_unpol_from_at_least_three_pol(
            [self.I1, self.I2, self.I3],
            [0, np.pi / 4, np.pi / 2]
        )

        # for those pixels which are not overexposed at two of the polarization angles
        self.m_case3 = np.float32(sum_mask == 2)
        # in pair: I1 and I3
        self.m_case3_1 = self.m_case3 * (self.mask1 * self.mask3)
        H_hat_case3_1 = self.I1 + self.I3
        # in pair: I2 and I4
        self.m_case3_2 = self.m_case3 * (self.mask2 * self.mask4)
        H_hat_case3_2 = self.I2 + self.I4
        # not in pair
        self.m_case3_3 = self.m_case3 * (self.mask1 * self.mask2)  # not in pair: I1 and I2
        self.m_case3_4 = self.m_case3 * (self.mask1 * self.mask4)  # not in pair: I1 and I4
        self.m_case3_5 = self.m_case3 * (self.mask2 * self.mask3)  # not in pair: I2 and I3
        self.m_case3_6 = self.m_case3 * (self.mask3 * self.mask4)  # not in pair: I3 and I4

        # for those pixels which are not overexposed at one of the polarization angles
        self.m_case4 = np.float32(sum_mask == 1)
        self.m_case4_1 = self.m_case4 * self.mask1
        self.m_case4_2 = self.m_case4 * self.mask2
        self.m_case4_3 = self.m_case4 * self.mask3
        self.m_case4_4 = self.m_case4 * self.mask4

        # for those pixels which are overexposed at all of the polarization angles
        self.m_case5 = np.float32(sum_mask == 0)

        H_hat_1_good = self.m_case1 * H_hat_case1 + self.m_case2_1 * H_hat_case2_1 + self.m_case2_2 * H_hat_case2_2 + self.m_case2_3 * H_hat_case2_3 + self.m_case2_4 * H_hat_case2_4 + self.m_case3_1 * H_hat_case3_1 + self.m_case3_2 * H_hat_case3_2
        H_hat_1_bad = (
                              self.m_case3_3 + self.m_case3_4 + self.m_case3_5 + self.m_case3_6 + self.m_case4 + self.m_case5) * H_hat_case1  # use case1 to fill the bad regions
        self.H_hat_1 = H_hat_1_good + H_hat_1_bad

        p_hat_good = self.m_case1 * p_hat_case1 + self.m_case2_1 * p_hat_case2_1 + self.m_case2_2 * p_hat_case2_2 + self.m_case2_3 * p_hat_case2_3 + self.m_case2_4 * p_hat_case2_4
        p_hat_bad = (self.m_case3 + self.m_case4 + self.m_case5) * p_hat_case1  # use case1 to fill the bad regions
        self.p_hat = p_hat_good + p_hat_bad

        theta_hat_good = self.m_case1 * theta_hat_case1 + self.m_case2_1 * theta_hat_case2_1 + self.m_case2_2 * theta_hat_case2_2 + self.m_case2_3 * theta_hat_case2_3 + self.m_case2_4 * theta_hat_case2_4
        theta_hat_bad = (
                                self.m_case3 + self.m_case4 + self.m_case5) * theta_hat_case1  # use case1 to fill the bad regions
        self.theta_hat = theta_hat_good + theta_hat_bad

        self.p_and_theta_weight = self.m_case1 * 0.01 + self.m_case2 * 0.05 + self.m_case3 + self.m_case4 + self.m_case5  # 1: bad  0: good

    def _precompute_2(self):
        H_hat_case3_3 = (self.I1 / (self.f1 + 1e-7) + self.I2 / (self.f2 + 1e-7)) / 2
        H_hat_case3_4 = (self.I1 / (self.f1 + 1e-7) + self.I4 / (self.f4 + 1e-7)) / 2
        H_hat_case3_5 = (self.I2 / (self.f2 + 1e-7) + self.I3 / (self.f3 + 1e-7)) / 2
        H_hat_case3_6 = (self.I3 / (self.f3 + 1e-7) + self.I4 / (self.f4 + 1e-7)) / 2

        H_hat_case4_1 = self.I1 / (self.f1 + 1e-7)
        H_hat_case4_2 = self.I2 / (self.f2 + 1e-7)
        H_hat_case4_3 = self.I3 / (self.f3 + 1e-7)
        H_hat_case4_4 = self.I4 / (self.f4 + 1e-7)

        unchange_mask = 1 - self.m_case3_3 - self.m_case3_4 - self.m_case3_5 - self.m_case3_6 - self.m_case4

        self.H_hat_2 = self.m_case3_3 * H_hat_case3_3 + self.m_case3_4 * H_hat_case3_4 + self.m_case3_5 * H_hat_case3_5 + self.m_case3_6 * H_hat_case3_6 + self.m_case4_1 * H_hat_case4_1 + self.m_case4_2 * H_hat_case4_2 + self.m_case4_3 * H_hat_case4_3 + self.m_case4_4 * H_hat_case4_4 + unchange_mask * self.H_hat_1
        self.H_weight = self.m_case1 * 0.01 + self.m_case2 * 0.05 + (self.m_case3_1 + self.m_case3_2) * 0.1 + (
                self.m_case3_3 + self.m_case3_4 + self.m_case3_5 + self.m_case3_6) * 0.25 + self.m_case4 * 0.5 + self.m_case5  # 1: bad  0: good

    def make(self):
        # net1:
        #   input: L_cat
        #   target: I_cat
        # net2:
        #   input: I_cat, p_hat, theta_hat, p_and_theta_weight
        #   target: theta, p
        # net3:
        #   input: H_hat_2, H_weight
        #   target: H
        self._find_t()
        if self.success:
            self.H = self.E * self.t
            self.I = np.clip(self.H, a_min=0, a_max=1)
            self.I1 = np.clip(self.E1 * self.t, a_min=0, a_max=1)
            self.I2 = np.clip(self.E2 * self.t, a_min=0, a_max=1)
            self.I3 = np.clip(self.E3 * self.t, a_min=0, a_max=1)
            self.I4 = np.clip(self.E4 * self.t, a_min=0, a_max=1)
            self.I_cat = np.concatenate([self.I1, self.I2, self.I3, self.I4], axis=2)

            # add 2% Gaussian noise and quantize
            shape = self.H.shape
            self.L1 = np.float32(
                np.floor(np.clip(np.random.normal(self.I1, self.I1 * 0.02, shape), a_min=0, a_max=1) * 255) / 255.)
            self.L2 = np.float32(
                np.floor(np.clip(np.random.normal(self.I2, self.I2 * 0.02, shape), a_min=0, a_max=1) * 255) / 255.)
            self.L3 = np.float32(
                np.floor(np.clip(np.random.normal(self.I3, self.I3 * 0.02, shape), a_min=0, a_max=1) * 255) / 255.)
            self.L4 = np.float32(
                np.floor(np.clip(np.random.normal(self.I4, self.I4 * 0.02, shape), a_min=0, a_max=1) * 255) / 255.)
            self.L_cat = np.concatenate([self.L1, self.L2, self.L3, self.L4], axis=2)

            self.log = "iter_cnt: {}\nfinal_bad_pixel_rate: {}\nt: {}\n".format(self.iter_cnt,
                                                                                self.final_bad_pixel_rate,
                                                                                self.t)
            self._precompute_1()
            self._precompute_2()
            return True
        else:
            return False

    def get_parameters_subnetwork1(self):
        # net1:
        #   input: L_cat
        #   target: I_cat
        input_parameters = [self.L_cat]
        target_parameters = [self.I_cat]
        return input_parameters, target_parameters

    def get_parameters_subnetwork2(self):
        # net2:
        #   input: I_cat, p_hat, theta_hat, p_and_theta_weight
        #   target: theta, p
        input_parameters = [self.I_cat, self.p_hat, self.theta_hat, self.p_and_theta_weight]
        target_parameters = [self.theta, self.p]
        return input_parameters, target_parameters

    def get_parameters_subnetwork3(self):
        # net3:
        #   input: H_hat_2, H_weight
        #   target: H
        input_parameters = [self.H_hat_2, self.H_weight]
        target_parameters = self.H
        return input_parameters, target_parameters


class DatasetSaver:
    """for saving the dataset"""

    def __init__(self, in_dir, number_for_train, out_base_dir, out_share_dir_names, out_subnetwork1_dir_names,
                 out_subnetwork2_dir_names, out_subnetwork3_dir_names, dataset_maker_args):
        self.E_dir = os.path.join(in_dir, 'HDR')
        self.p_dir = os.path.join(in_dir, 'p')
        self.theta_dir = os.path.join(in_dir, 'theta')

        self.file_names = fnmatch.filter(os.listdir(self.E_dir), '*.npy')
        random.shuffle(self.file_names)
        self.file_names_for_train = self.file_names[:number_for_train]
        self.file_names_for_test = self.file_names[number_for_train:]

        self.out_base_dir = out_base_dir
        self.out_share_dir_names = out_share_dir_names
        self.out_subnetwork1_dir_names = out_subnetwork1_dir_names
        self.out_subnetwork2_dir_names = out_subnetwork2_dir_names
        self.out_subnetwork3_dir_names = out_subnetwork3_dir_names

        self.dataset_maker_args = dataset_maker_args

        self.error_list_train = []
        self.error_list_test = []

    def _save(self, mode):
        out_share_dir = os.path.join(self.out_base_dir, mode, 'share')
        out_subnetwork1_dir = os.path.join(self.out_base_dir, mode, 'subnetwork1')
        out_subnetwork2_dir = os.path.join(self.out_base_dir, mode, 'subnetwork2')
        out_subnetwork3_dir = os.path.join(self.out_base_dir, mode, 'subnetwork3')
        ensure_dirs(out_share_dir, self.out_share_dir_names)
        ensure_dirs(out_subnetwork1_dir, self.out_subnetwork1_dir_names)
        ensure_dirs(out_subnetwork2_dir, self.out_subnetwork2_dir_names)
        ensure_dirs(out_subnetwork3_dir, self.out_subnetwork3_dir_names)

        assert mode in ('train', 'test')
        error_list = getattr(self, 'error_list_{}'.format(mode))

        for file_name in getattr(self, 'file_names_for_{}'.format(mode)):
            E = np.load(os.path.join(self.E_dir, file_name))
            p = np.load(os.path.join(self.p_dir, file_name))
            theta = np.load(os.path.join(self.theta_dir, file_name))

            img_name = file_name.split('.')[0]
            if mode == 'train':
                items = data_augmentation_train(E, p, theta, img_name, no_rotation=False)
            elif mode == 'test':
                items = data_augmentation_test(E, p, theta, img_name, no_rotation=True)
            else:
                raise Exception('mode error!')

            for item in items:
                E_, p_, theta_, img_name_ = item

                dataset_maker = DatasetMaker(E_, p_, theta_, **self.dataset_maker_args)
                success = dataset_maker.make()

                if success:
                    for out_share_dir_name in self.out_share_dir_names:
                        f = getattr(dataset_maker, out_share_dir_name)
                        save_dir = os.path.join(out_share_dir, out_share_dir_name)
                        save_npy(f, save_dir, img_name_)
                    for out_subnetwork1_dir_name in self.out_subnetwork1_dir_names:
                        f = getattr(dataset_maker, out_subnetwork1_dir_name)
                        save_dir = os.path.join(out_subnetwork1_dir, out_subnetwork1_dir_name)
                        save_npy(f, save_dir, img_name_)
                    for out_subnetwork2_dir_name in self.out_subnetwork2_dir_names:
                        f = getattr(dataset_maker, out_subnetwork2_dir_name)
                        save_dir = os.path.join(out_subnetwork2_dir, out_subnetwork2_dir_name)
                        save_npy(f, save_dir, img_name_)
                    for out_subnetwork3_dir_name in self.out_subnetwork3_dir_names:
                        f = getattr(dataset_maker, out_subnetwork3_dir_name)
                        save_dir = os.path.join(out_subnetwork3_dir, out_subnetwork3_dir_name)
                        save_npy(f, save_dir, img_name_)
                    print('img: {}, iter_cnt: {}, final_bad_pixel_rate: {}'.format(img_name_, dataset_maker.iter_cnt,
                                                                                   dataset_maker.final_bad_pixel_rate))
                else:
                    print('img {} fail to expose, skip...'.format(img_name_))
                    error_list.append(img_name_)

        print('{} failed images (for {}):'.format(len(error_list), mode))
        print('error list (for {}):'.format(mode), error_list)

    def save_train(self):
        print('start to generate training data...')
        self._save('train')
        print('done!\n')

    def save_test(self):
        print('start to generate test data...')
        self._save('test')
        print('done!\n')


if __name__ == '__main__':
    in_dir = '../EdPolCommunitySourceFiles'
    number_for_train = 80
    out_base_dir = '../data'
    out_share_dir_names = ['I_cat', 'p', 'theta']
    out_subnetwork1_dir_names = ['L_cat']
    out_subnetwork2_dir_names = ['p_hat', 'theta_hat', 'p_and_theta_weight']
    out_subnetwork3_dir_names = ['H_hat_2', 'H_weight', 'H']
    dataset_maker_args = {
        'bad_pixel_rate_ubound': 0.15,
        'bad_pixel_rate_lbound': 0.025,
        'iter_max': 15
    }

    dataset_saver = DatasetSaver(
        in_dir,
        number_for_train,
        out_base_dir,
        out_share_dir_names,
        out_subnetwork1_dir_names,
        out_subnetwork2_dir_names,
        out_subnetwork3_dir_names,
        dataset_maker_args
    )
    dataset_saver.save_train()
    dataset_saver.save_test()
