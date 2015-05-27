import numpy as np, cv2


def order_points(points):
    """
    Sortira koordinate pravokutnika u poretku za transformacijsku matricu
    :param points: list of 4 2-tuples representing four vertices of rectangle
    :return: returns coordinates in order: top-left, top-right, bottom-right, bottom-left
    """

    rect = np.zeros((4, 2), dtype="float32")
    s = points.sum(axis=1)

    # rect = [top-left, top-right, bottom-right, bottom-left]

    rect[0] = points[np.argmin(s)]  # gornji lijevi vrh ima najmanju sumu x i y koordinata
    rect[2] = points[np.argmax(s)]  # donji desni vrh ima najvecu sumu x i y koordinata

    d = np.diff(points, axis=1)

    rect[1] = points[np.argmin(d)]  # gornji desni vrh ima najmanju razliku x i y koordinata
    rect[3] = points[np.argmax(d)]  # donji lijevi vrh ima najvecu razliku x i y koordinata

    return rect


def transform(image, points):
    """
    Transformira sliku koja je pod nekim kutem u top-down perspective sliku
    :param image: image we want to apply perspective transform to
    :param points: list of 4 2-tuples representing four vertices of rectangle (ROI of image)
    :return: returns warped image (top-down perspective)
    """

    rect = order_points(points)
    (top_left, top_right, bottom_right, bottom_left) = rect

    # racuna sirinu nove slike, max(duljina(top-right -> top-left), duljina(bottom-right -> bottom-left))
    # duljina = sqrt((x2 - x1)^2 + (y2 - y1)^2)
    width = max(int(np.sqrt((bottom_right[0] - bottom_left[0]) ** 2 + (bottom_right[1] - bottom_left[1]) ** 2)),
                int(np.sqrt((top_right[0] - top_left[0]) ** 2 + (top_right[1] - top_left[1]) ** 2)))

    # racuna sirinu nove slike, max(duljina(bottom-right -> top-right), duljina(bottom-left -> top-right))
    # duljina = sqrt((x2 - x1)^2 + (y2 - y1)^2)
    height = max(int(np.sqrt(((bottom_right[0] - top_right[0]) ** 2) + ((bottom_right[1] - top_right[1]) ** 2))),
                 int(np.sqrt((bottom_left[0] - top_left[0]) ** 2 + (bottom_left[1] - top_left[1]) ** 2)))

    # nova 4 vrha za transformiranu sliku ovisni o pronadjenoj sirini i visini slike
    destination = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype="float32")


    # transformacijska matrica orginalni vrhovi -> dest vrhovi
    transform_matrix = cv2.getPerspectiveTransform(rect, destination)

    # konacno digni sliku (top-down perspective)
    wraped_image = cv2.warpPerspective(image, transform_matrix, (width, height))

    return wraped_image
