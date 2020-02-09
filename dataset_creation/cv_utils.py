import cv2 as cv
import numpy as np


# Maybe better is to approximate: https://docs.opencv.org/3.4/dd/d49/tutorial_py_contour_features.html
def get_convex_hull(unit_bw, y_shift, x_shift):
    """Return the convex polygon of a unit."""
    contours, _ = cv.findContours(unit_bw, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    hull = cv.convexHull(contours[0])
    return hull + np.array([x_shift, y_shift])


def get_bounding_box(unit_bw, y_shift, x_shift):
    """Return the bounding box of a unit."""
    hull = get_convex_hull(unit_bw, y_shift, x_shift)
    bbox = cv.boundingRect(hull)
    return bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]  # XYWH -> XYXY


def get_polygon_from_bbox(bbox):
    """Return the polygon which strictly repeats the bounding box."""
    xmin, ymin, xmax, ymax = bbox
    return [xmin, ymin, xmin, ymax, xmax, ymax, xmax, ymin]


def paste_unit(arena, unit, y, x):
    """Paste a unit onto an arena. Return the resulting image."""
    y1, y2 = y, y + unit.shape[0]
    x1, x2 = x, x + unit.shape[1]

    alpha_u = unit[:, :, 3] / 255.0
    alpha_a = 1.0 - alpha_u

    for c in range(3):
        arena[y1:y2, x1:x2, c] = (alpha_u * unit[:, :, c] + alpha_a * arena[y1:y2, x1:x2, c])

    return arena


def scale_unit(img, img_bw, ratio):
    new_w = round(img.shape[1] * ratio)
    new_h = round(img.shape[0] * ratio)
    return cv.resize(img, (new_w, new_h)), cv.resize(img_bw, (new_w, new_h))
