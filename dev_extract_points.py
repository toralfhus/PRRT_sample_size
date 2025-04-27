import cv2
import numpy as np
import sys
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import r2_score
from dev_utils_mc import load_dose_volume_points


if __name__ == "__main__":
    # img_path = "tcp_g2_warfvinge_2024.png"
    # img_path = "tcp_hebert_2024.png"
    img_path = "../../PRRT_sample_size/figures/pub1_hebert24/dose_dtv_hebert_2024.png"

    # x, y = load_dose_volume_points(plot=True)
    x, y = load_dose_volume_points(img_path, plot=True)


    sys.exit()

    img_path = "../../PRRT_sample_size/figures/pub1_hebert24/dose_dtv_hebert_2024.png"
    px_to_xval = np.array([[0, 70], [50, 205], [100, 340], [150, 474], [200, 609]])         # total tumor absorbed dose, x-pixel value
    px_to_yval = np.array([[0, 221], [10, 189], [20, 156], [40, 92], [60, 27]])                 # delta tumor volume, y-pixel value

    px_to_xval = np.array([[y, x] for (x, y) in px_to_xval])
    px_to_yval = np.array([[y, x] for (x, y) in px_to_yval])

    lr_x = LinearRegression(fit_intercept=True)
    lr_x.fit(*[x.reshape(-1, 1) for x in px_to_xval.T])
    print(lr_x.coef_, lr_x.score(*[x.reshape(-1, 1) for x in px_to_xval.T]))

    lr_y = LinearRegression(fit_intercept=True)
    lr_y.fit(*[y.reshape(-1, 1) for y in px_to_yval.T])
    print(lr_y.coef_, lr_y.score(*[y.reshape(-1, 1) for y in px_to_yval.T]))


    print(img_path)
    img = cv2.imread(img_path)
    image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # gray_image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    gray_image = image_rgb[:, :, 0] # single channel
    # print(np.unique(gray_image, return_counts=True))

    # fig, ax = plt.subplots(ncols=3)
    # ax[0].imshow(image_rgb[:, :, 0])
    # ax[1].imshow(image_rgb[:, :, 1])
    # ax[2].imshow(image_rgb[:, :, 2])
    # fig.tight_layout()
    # plt.show()

    # plt.imshow(gray_image, cmap="gray")
    # plt.show()

    # _, thresh = cv2.threshold(gray_image, 237, 255, cv2.THRESH_BINARY_INV)
    # fig, ax = plt.subplots(figsize=(12, 12))
    # ax.imshow(thresh, cmap="gray_r")
    # plt.show()

    mask = cv2.inRange(gray_image, 0, 1)

    # plt.imshow(mask, cmap="gray_r")
    # plt.show()

    # Find contours of the points in the new image
    # contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # cimg = np.copy(img)
    # cv2.drawContours(cimg, contours, -1, (0, 255, 0), 2)  # Green color for contours
    # cimg_rgb = cv2.cvtColor(cimg, cv2.COLOR_BGR2RGB)
    # plt.imshow(cimg_rgb)
    # plt.show()

    # Extract the x, y coordinates of each detected point
    points = []
    for contour in contours:
        # Get the center of the contour
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            points.append((cx, cy))

    # Convert points to a numpy array
    points = np.array(points)
    # points_new = np.sort(points_new, axis=0)
    # points_new = np.array(sorted(points_new, key=lambda point: point[0]))
    print(points.shape)

    # print(points_new)
    # points_new = np.array([point for point in points_new if point[0] > 50])
    # print(points)

    points_x = lr_x.predict(points.T[0].reshape(-1, 1))
    points_y = lr_y.predict(points.T[1].reshape(-1, 1))

    # points_x = lr_x.predict(points.T[1].reshape(-1, 1))
    # points_y = lr_y.predict(points.T[0].reshape(-1, 1))

    # print(points_x.ravel())
    # print(points_y.ravel())

    print(points.shape)

    # plt.plot(points.T[0], -points.T[1], "o", ms=2)
    fig, ax = plt.subplots(nrows=2, sharex=True)
    ax[0].plot(points_x, points_y, "o", ms=2)
    ax[0].hlines([20], 0, 300, ls=":")
    ax[0].set_xlim(0, 300)
    ax[0].set_title(f"{len(points)} points found")
    ax[1].hist(points_x, bins=21)
    ax[1].set_xlabel("Gy")
    ax[0].set_ylabel("rel_change_volume")
    plt.show()

    x0 = points_x[points_y > 0]
    x1 = points_x[points_y <= 0]

    x = np.concatenate([x0, x1]).reshape(-1, 1)
    y = np.array([0] * len(x0) + [1] * len(x1)).reshape(-1, 1)

    lr0 = LogisticRegression()
    lr0.fit(x, y)
    xvals = np.linspace(0, np.max(x), 100).reshape(-1, 1)
    phat = lr0.predict_proba(xvals)[:, -1]

    fig, ax = plt.subplots()
    ax.plot(x, y, "o")
    ax.plot(xvals, phat, ":", c="black")
    ax.set_yticks([0, .25, .50, .75, 1])
    ax.grid(1)
    plt.show()
