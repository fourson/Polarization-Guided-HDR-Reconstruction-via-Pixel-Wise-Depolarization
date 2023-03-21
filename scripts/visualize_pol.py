import os
import cv2
import numpy as np

for f in os.listdir(os.getcwd()):
    if f.endswith('.npy'):
        name = f.split('.')[0]
        pol_HDR_imgs = np.load(f)

        I_0, I_45, I_90, I_135 = np.split(np.clip(pol_HDR_imgs, a_min=0, a_max=1), 4, axis=2)

        cv2.imwrite(name + '_0.jpg', cv2.cvtColor(I_0, cv2.COLOR_RGB2BGR) * 255)
        cv2.imwrite(name + '_45.jpg', cv2.cvtColor(I_45, cv2.COLOR_RGB2BGR) * 255)
        cv2.imwrite(name + '_90.jpg', cv2.cvtColor(I_90, cv2.COLOR_RGB2BGR) * 255)
        cv2.imwrite(name + '_135.jpg', cv2.cvtColor(I_135, cv2.COLOR_RGB2BGR) * 255)
