import os

import cv2
import matplotlib.pyplot as plt
import numpy as np

'''PROBLEM 1'''


def power_law(image, y):
    scaled_img = image / 255.0
    # Apply power-law transformation
    res_img = np.array(255.0 * (scaled_img ** y), dtype='uint8')
    return res_img


def problem_one():
    smile = cv2.imread('smile.jpg', cv2.IMREAD_GRAYSCALE)
    gamma_point_3 = power_law(smile, 0.3)
    gamma_3 = power_law(smile, 3)
    os.makedirs('problem_1_imgs', exist_ok=True)

    cv2.imwrite('problem_1_imgs/smile_bw.jpg', smile)
    cv2.imwrite('problem_1_imgs/smile_pl_0.3.jpg', gamma_point_3)
    cv2.imwrite('problem_1_imgs/smile_pl_3.jpg', gamma_3)


'''PROBLEM 2'''


def get_k_bit_plane_slice(image: np.ndarray, k, multiply_for_visuals=False):
    height, width = image.shape
    plane = np.full((height, width), 2 ** k, dtype='uint8')
    res = cv2.bitwise_and(plane, image)
    # Multiply by 255 for visuals
    res = res * 255 if multiply_for_visuals else res
    return res


def combine_bit_planes(image, start=0, stop=7):
    bit_plane_slices = [get_k_bit_plane_slice(image, i) for i in range(start, stop + 1)]
    res_image = bit_plane_slices[0]
    for i in range(1, len(bit_plane_slices)):
        res_image = cv2.bitwise_or(res_image, bit_plane_slices[i])
    return res_image


def split_bit_planes(image, output_prefix='smile_bit'):
    bit_plane_slices = [get_k_bit_plane_slice(image, i, multiply_for_visuals=True) for i in range(8)]
    for i, bit_slice in enumerate(bit_plane_slices):
        cv2.imwrite(f'problem_2_imgs/{output_prefix}_{i}.jpg', bit_slice)


def problem_two():
    smile = cv2.imread('smile.jpg', cv2.IMREAD_GRAYSCALE)
    smile = np.array(smile, dtype='uint8')
    os.makedirs('problem_2_imgs', exist_ok=True)
    # Split all planes and save
    split_bit_planes(smile)
    # Combine top 4 and 2 planes into an image
    res_img_highest_four = combine_bit_planes(smile, start=4)
    res_img_highest_two = combine_bit_planes(smile, start=6)

    cv2.imwrite('problem_2_imgs/smile_highest_4_bits.jpg', res_img_highest_four)
    cv2.imwrite('problem_2_imgs/smile_highest_2_bits.jpg', res_img_highest_two)


'''PROBLEM 3'''


def save_hist(hist, title, save=False, filename='plot.jpg'):
    plt.figure()
    plt.title(title)
    plt.xlabel('Bins')
    plt.ylabel('# of Pixels')
    plt.plot(hist)
    plt.xlim([0, hist.shape[0]])
    if save:
        plt.savefig(filename)
    else:
        plt.show()
    plt.clf()


def problem_three():
    smile_point_three = cv2.imread('problem_1_imgs/smile_pl_0.3.jpg', cv2.IMREAD_GRAYSCALE)
    smile_three = cv2.imread('problem_1_imgs/smile_pl_3.jpg', cv2.IMREAD_GRAYSCALE)
    os.makedirs('problem_3_imgs', exist_ok=True)

    # Get current histograms of both images
    hist_three = cv2.calcHist([smile_three], [0], None, [256], [0, 256])
    hist_point_three = cv2.calcHist([smile_point_three], [0], None, [256], [0, 256])
    save_hist(hist_three, 'PL y=3 Histogram', save=True, filename='problem_3_imgs/hist_no_normalization_3.jpg')
    save_hist(hist_point_three, 'PL y=0.3 Histogram', save=True,
              filename='problem_3_imgs/hist_no_normalization_0.3.jpg')

    # Equalize the images
    smile_point_three_eq = cv2.equalizeHist(smile_point_three)
    smile_three_eq = cv2.equalizeHist(smile_three)
    cv2.imwrite('problem_3_imgs/smile_eq_pl_0.3.jpg', smile_point_three_eq)
    cv2.imwrite('problem_3_imgs/smile_eq_pl_3.jpg', smile_three_eq)

    # Save the equalized image histograms
    hist_three_eq = cv2.calcHist([smile_three_eq], [0], None, [256], [0, 256])
    hist_point_three_eq = cv2.calcHist([smile_point_three_eq], [0], None, [256], [0, 256])
    save_hist(hist_three_eq, 'PL Equalized y=3 Histogram', save=True, filename='problem_3_imgs/hist_normalized_3.jpg')
    save_hist(hist_point_three_eq, 'PL Equalized y=0.3 Histogram', save=True,
              filename='problem_3_imgs/hist_normalized_0.3.jpg')


'''PROBLEM 5'''


def equalize_histogram(image: np.ndarray, bits=8):
    num_of_vals = 2 ** bits
    # Get normalized cumulative histogram
    hist = np.bincount(image.flatten(), minlength=num_of_vals)
    num_pix = np.sum(hist)
    hist = hist / num_pix
    cum_hist = np.cumsum(hist)

    # Pixel mapping lookup table
    transform = np.floor((num_of_vals-1) * cum_hist)
    transform = transform.astype('uint8')

    # Transformation
    img_flattened = image.flatten()
    eq_img = np.array([transform[i] for i in img_flattened])

    # Reshape to original dims
    return np.reshape(eq_img, image.shape)


def problem_five():
    image = np.array([
        [1, 2, 4, 7, 3],
        [2, 4, 7, 3, 1],
        [5, 6, 2, 1, 1],
        [4, 7, 1, 1, 1]], dtype='uint8')
    os.makedirs('problem_4_imgs', exist_ok=True)
    image_hist = cv2.calcHist([image], [0], None, [8], [0, 8])
    save_hist(image_hist, 'Histogram', save=True, filename='problem_4_imgs/image_histogram.png')

    image_eq = equalize_histogram(image, bits=3)
    image_hist_eq = cv2.calcHist([image_eq], [0], None, [8], [0, 8])

    save_hist(image_hist_eq, 'Histogram Equalized', save=True, filename='problem_4_imgs/image_equalized_histogram.png')


if __name__ == '__main__':
    # problem_one()
    # problem_two()
    # problem_three()
    problem_five()
