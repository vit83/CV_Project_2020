from sklearn.cluster import KMeans, MiniBatchKMeans
from collections import Counter
import cv2
import numpy as np
import json
import glob
import matplotlib.pyplot as plt
import time
from sklearn.metrics import calinski_harabaz_score
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import


def get_dominant_color(image, k, image_processing_size=(50, 50)):
    """
    takes an image as input
    returns the dominant color of the image as a list

    dominant color is found by running k means on the
    pixels & returning the centroid of the largest cluster

    processing time is sped up by working with a smaller image;
    this resizing can be done with the image_processing_size param
    which takes a tuple of image dims as input


    """
    if image_processing_size is not None:

        image = cv2.resize(image, image_processing_size,
                           interpolation=cv2.INTER_AREA)

    # reshape the image to be a list of pixels
    image = image.reshape((image.shape[0] * image.shape[1], 3))

    # cluster and assign labels to the pixels

    clt = KMeans(n_clusters=k).fit(image)

    clt_2 = MiniBatchKMeans(n_clusters=k - 2).fit(image)
    clt_3 = MiniBatchKMeans(n_clusters=k - 1).fit(image)
    clt_5 = MiniBatchKMeans(n_clusters=k + 1).fit(image)
    clt_6 = MiniBatchKMeans(n_clusters=k + 2).fit(image)
    clt_7 = MiniBatchKMeans(n_clusters=k + 3).fit(image)
    clt_8 = MiniBatchKMeans(n_clusters=k + 4).fit(image)
    clt_9 = MiniBatchKMeans(n_clusters=k + 5).fit(image)
    clt_10 = MiniBatchKMeans(n_clusters=k + 6).fit(image)
    clt_11 = MiniBatchKMeans(n_clusters=k + 7).fit(image)
    clt_12 = MiniBatchKMeans(n_clusters=k + 8).fit(image)

    clt_list = [clt, clt_2, clt_3, clt_5, clt_6, clt_7, clt_8, clt_9, clt_10,
                clt_11, clt_12]
    CH_score = []

    for model in clt_list:
        labels = model.labels_
        CH_score.append(calinski_harabaz_score(image, labels))

    #plt.plot([2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], CH_score)
    #plt.xticks([2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
    #plt.title("Calinski Harabaz Scores for Different Values of K")
    #plt.ylabel("Variance Ratio")
    #plt.xlabel("K=")
    #plt.show()

    #     clt = MiniBatchKMeans(init='k-means++', n_clusters=k, batch_size=45,
    #                       n_init=10, max_no_improvement=10, verbose=0).fit(image)

    labels = clt.predict(image)

    # plot KMeans clusters to determine distance accuracy

    #fig = plt.figure(figsize=(12, 8))
    #ax1 = fig.add_subplot(111, projection='3d')
    #centers = clt.cluster_centers_
    #ax1.scatter(centers[:, 0], centers[:, 1], centers[:, 2], c='black', s=70, alpha=1)
    #x = image[:, 0]
    #y = image[:, 1]
    #z = image[:, 2]
    #ax1.scatter(x, y, z, marker=".", c=labels, s=5, alpha=0.2)
    #plt.show()

    # count labels to find most popular
    label_counts = Counter(labels)

    # subset out most popular centroid
    dominant_color_1 = clt.cluster_centers_[label_counts.most_common(1)[0][0]]
    dominant_color_2 = clt.cluster_centers_[label_counts.most_common(2)[1][0]]
    dominant_color_3 = clt.cluster_centers_[label_counts.most_common(3)[2][0]]
    dominant_color_4 = clt.cluster_centers_[label_counts.most_common(4)[3][0]]
    dominant_color_5 = clt.cluster_centers_[label_counts.most_common(5)[4][0]]
    dominant_color_6 = clt.cluster_centers_[label_counts.most_common(6)[5][0]]

    return list(dominant_color_1), list(dominant_color_2), list(dominant_color_3), list(dominant_color_4), list(dominant_color_5), list(dominant_color_6)


# ap = argparse.ArgumentParser()
# ap.add_argument("-i", "--imagePath", required=True,
# 	help="Path to image to find dominant color of")
# ap.add_argument("-k", "--clusters", default=3, type=int,
# 	help="Number of clusters to use in kmeans when finding dominant color")
# args = vars(ap.parse_args())

for i in glob.glob('../BusesOnly/*'):
    # read in image of interest
    bgr_image = cv2.imread(i)


    h, w, _ = bgr_image.shape
    x_center = int(w / 2)
    y_center = int(h / 2)
    wBox = 100
    hBox = 50
    bgr_image = bgr_image[(y_center - hBox):(y_center + hBox), (x_center - wBox):(x_center + wBox)]



    # convert to HSV; this is a better representation of how we see color
    hsv_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2HSV)
    # extract 3 most dominant colors
    # (aka the centroid of the most popular k means cluster)
    start = time.time()
    dom_color_1, dom_color_2, dom_color_3, dom_color_4, dom_color_5, dom_color_6 = get_dominant_color(hsv_image, k=6)
    print('get_dominant_color function: {:.3f}s'.format(time.time() - start))

    # create a square showing dominant color of equal size to input image
    dom_color_1_hsv = np.full(bgr_image.shape, dom_color_1, dtype='uint8')
    # convert to bgr color space for display
    dom_color_1_rgb = cv2.cvtColor(dom_color_1_hsv, cv2.COLOR_HSV2RGB)

    # create a square showing dominant color of equal size to input image
    dom_color_2_hsv = np.full(bgr_image.shape, dom_color_2, dtype='uint8')
    # convert to bgr color space for display
    dom_color_2_rgb = cv2.cvtColor(dom_color_2_hsv, cv2.COLOR_HSV2RGB)

    # create a square showing dominant color of equal size to input image
    dom_color_3_hsv = np.full(bgr_image.shape, dom_color_3, dtype='uint8')
    # convert to bgr color space for display
    dom_color_3_rgb = cv2.cvtColor(dom_color_3_hsv, cv2.COLOR_HSV2RGB)

    # create a square showing dominant color of equal size to input image
    dom_color_4_hsv = np.full(bgr_image.shape, dom_color_4, dtype='uint8')
    # convert to bgr color space for display
    dom_color_4_rgb = cv2.cvtColor(dom_color_4_hsv, cv2.COLOR_HSV2RGB)

    # create a square showing dominant color of equal size to input image
    dom_color_5_hsv = np.full(bgr_image.shape, dom_color_5, dtype='uint8')
    # convert to bgr color space for display
    dom_color_5_rgb = cv2.cvtColor(dom_color_5_hsv, cv2.COLOR_HSV2RGB)

    # create a square showing dominant color of equal size to input image
    dom_color_6_hsv = np.full(bgr_image.shape, dom_color_6, dtype='uint8')
    # convert to bgr color space for display
    dom_color_6_rgb = cv2.cvtColor(dom_color_6_hsv, cv2.COLOR_HSV2RGB)

    # concat input image and dom color square side by side for display
    output_image = np.hstack((bgr_image[:, :, ::-1], dom_color_1_rgb, dom_color_2_rgb, dom_color_3_rgb, dom_color_4_rgb, dom_color_5_rgb, dom_color_6_rgb))

    # show results to screen
    print('\nMost prominent color:\nred:', dom_color_1_rgb[0][0][0], ' green: ', dom_color_1_rgb[0][0][1],
          ' blue: ', dom_color_1_rgb[0][0][2], '\nHex: ', '#%02x%02x%02x' % (dom_color_1_rgb[0][0][0],
                                                                             dom_color_1_rgb[0][0][1],
                                                                             dom_color_1_rgb[0][0][2]))

    print('\nSecond color:\nred:', dom_color_2_rgb[0][0][0], ' green: ', dom_color_1_rgb[0][0][1],
          ' blue: ', dom_color_2_rgb[0][0][2], '\nHex: ', '#%02x%02x%02x' % (dom_color_2_rgb[0][0][0],
                                                                             dom_color_2_rgb[0][0][1],
                                                                             dom_color_2_rgb[0][0][2]))

    print('\nThird color:\nred:', dom_color_3_rgb[0][0][0], ' green: ', dom_color_1_rgb[0][0][1],
          ' blue: ', dom_color_3_rgb[0][0][2], '\nHex: ', '#%02x%02x%02x' % (dom_color_3_rgb[0][0][0],
                                                                             dom_color_3_rgb[0][0][1],
                                                                             dom_color_3_rgb[0][0][2]))
    print('\nFourth color:\nred:', dom_color_4_rgb[0][0][0], ' green: ', dom_color_4_rgb[0][0][1],
          ' blue: ', dom_color_4_rgb[0][0][2], '\nHex: ', '#%02x%02x%02x' % (dom_color_4_rgb[0][0][0],
                                                                             dom_color_4_rgb[0][0][1],
                                                                             dom_color_4_rgb[0][0][2]))

    print('\nFifth color:\nred:', dom_color_5_rgb[0][0][0], ' green: ', dom_color_5_rgb[0][0][1],
          ' blue: ', dom_color_5_rgb[0][0][2], '\nHex: ', '#%02x%02x%02x' % (dom_color_5_rgb[0][0][0],
                                                                             dom_color_5_rgb[0][0][1],
                                                                             dom_color_5_rgb[0][0][2]))

    print('\nSixth color:\nred:', dom_color_6_rgb[0][0][0], ' green: ', dom_color_6_rgb[0][0][1],
          ' blue: ', dom_color_6_rgb[0][0][2], '\nHex: ', '#%02x%02x%02x' % (dom_color_6_rgb[0][0][0],
                                                                             dom_color_6_rgb[0][0][1],
                                                                             dom_color_6_rgb[0][0][2]))
    plt.figure(figsize=(12, 6))
    plt.imshow(output_image)
    plt.show()