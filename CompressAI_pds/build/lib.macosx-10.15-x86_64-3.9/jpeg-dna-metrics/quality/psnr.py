"""PSNR Metrics"""
import math
import argparse
from skimage import io
import cv2



class RGBYCbCr():
    """RGB YCbCr color converter"""
    def forward(self, inp):
        """Wrapper method to covert RGB to YCbCr

        :param inp: Input image
        :type inp: np.array
        :return: YCbCr image
        :rtype: np.array
        """
        res = cv2.cvtColor(inp, cv2.COLOR_RGB2YCrCb)
        res[:, :, 1], res[:, :, 2] = res[:, :, 2], res[:, :, 1].copy()
        return res

    def inverse(self, inp):
        """Wrapper method to covert YCbCr to RGB

        :param inp: Input YCbCr image
        :type inp: np.array
        :return: RGB image
        :rtype: np.array
        """
        inp[:, :, 1], inp[:, :, 2] = inp[:, :, 2], inp[:, :, 1].copy()
        return cv2.cvtColor(inp, cv2.COLOR_YCrCb2RGB)

class RGBYUV():
    """RGB YCbCr color converter"""
    def forward(self, inp):
        """Wrapper method to covert RGB to YCbCr

        :param inp: Input image
        :type inp: np.array
        :return: YCbCr image
        :rtype: np.array
        """
        res = cv2.cvtColor(inp, cv2.COLOR_RGB2YUV)
        return res

    def inverse(self, inp):
        """Wrapper method to covert YCbCr to RGB

        :param inp: Input YCbCr image
        :type inp: np.array
        :return: RGB image
        :rtype: np.array
        """
        return cv2.cvtColor(inp, cv2.COLOR_YUV2RGB)

def gray_psnr(img1_path, img2_path):
    """Gray level PSNR"""

    img1 = io.imread(img1_path)
    img2 = io.imread(img2_path)

    diff = img1.astype(int) - img2.astype(int)
    mean_squared_error = 0
    for i in range(len(diff)):
        for j in range(len(diff[0])):
            mean_squared_error += diff[i,j] ** 2
        mean_squared_error /= len(diff)
        mean_squared_error /= len(diff[0])
    PSNR = 10 * math.log10((255**2)/mean_squared_error)
    return PSNR

def psnr_y_cb_cr(img1_path, img2_path):
    """PSNR metric on the YCbCr color space"""
    y_cb_cr_transform = RGBYCbCr()

    img1 = io.imread(img1_path)
    img2 = io.imread(img2_path)
    img1_y_cb_cr, img2_y_cb_cr = y_cb_cr_transform.forward(img1), y_cb_cr_transform.forward(img2)

    try:
        assert img1.shape == img2.shape
    except AssertionError as exce:
        print("Images have different sizes")
        raise exce

    PSNR_YCBCR = 0

    for k in range(3):
        diff_y_cb_cr = img1_y_cb_cr[:,:,k].astype(int) - img2_y_cb_cr[:,:,k].astype(int)
        for i in range(len(diff_y_cb_cr)):
            for j in range(len(diff_y_cb_cr[0])):
                mean_squared_error_y_cb_cr += diff_y_cb_cr[i, j]**2
        mean_squared_error_y_cb_cr /= len(diff_y_cb_cr)
        mean_squared_error_y_cb_cr /= len(diff_y_cb_cr[0])
        if k == 0:
            PSNR_YCBCR += 60 * math.log10((255**2)/mean_squared_error_y_cb_cr)
        else:
            PSNR_YCBCR += 10 * math.log10((255**2)/mean_squared_error_y_cb_cr)
    PSNR_YCBCR /= 8
    return PSNR_YCBCR

def psnr_yuv(img1_path, img2_path):
    """PSNR metric on the YUV color space"""
    yuv_transform = RGBYUV()

    img1 = io.imread(img1_path)
    img2 = io.imread(img2_path)
    img1_yuv, img2_yuv = yuv_transform.forward(img1), yuv_transform.forward(img2)

    try:
        assert img1.shape == img2.shape
    except AssertionError as exce:
        print("Images have different sizes")
        raise exce

    PSNR_YUV = 0

    for k in range(3):
        diff_yuv = img1_yuv[:,:,k].astype(int) - img2_yuv[:,:,k].astype(int)
        mean_squared_error_yuv = 0
        for i in range(len(diff_yuv)):
            for j in range(len(diff_yuv[0])):
                mean_squared_error_yuv += diff_yuv[i, j]**2
        mean_squared_error_yuv /= len(diff_yuv)
        mean_squared_error_yuv /= len(diff_yuv[0])
        if k == 0:
            PSNR_YUV += 60 * math.log10((255**2)/mean_squared_error_yuv)
        else:
            PSNR_YUV += 10 * math.log10((255**2)/mean_squared_error_yuv)
    PSNR_YUV /= 8
    return PSNR_YUV

def main(img1_path, img2_path):
    """Main script"""
    im = io.imread(img1_path)
    if len(im.shape) in [3, 4]:
        print(f"PSNR YUV: {psnr_yuv(img1_path, img2_path)}")
        print(f"PSNR YCbCr: {psnr_y_cb_cr(img1_path, img2_path)}")
    else:
        print(f"PSNR: {gray_psnr(img1_path, img2_path)}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('IM1PATH',
                        type=str,
                        help='First Image path')
    parser.add_argument('IM2PATH',
                        type=str,
                        help='Second Image path')

    args = parser.parse_args()

    img1_path = args.IM1PATH
    img2_path = args.IM2PATH
    main(img1_path, img2_path)
