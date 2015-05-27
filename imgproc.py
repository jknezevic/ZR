import numpy as np, cv2

width, height, area = 250, 350, 250 * 350


def black_background(input_image, white_level):
    """
    Uzima top-down sliku i stavlja umjesto bijele pozadine crnu
    :param input_image: image for processing
    :param white_level: white level of the card
    :return: image with black background instead of white
    """

    dims = input_image.shape
    output_image = input_image.copy()

    for y in xrange(dims[0]):
        for x in xrange(dims[1]):
            b, g, r = input_image[y][x]

            if b > white_level and g > white_level and r > white_level:
                output_image[y][x] = [0, 0, 0]

    return output_image


def filter_red(input_image, value):
    """
    Filtrira sliku i ostavlja samo crvenu boju
    :param input_image: image for processing
    :param value: 0 by default, set green and blue values to this value
    :return: image with only red channel
    """

    dims = input_image.shape
    output_image = input_image.copy()

    for y in xrange(dims[0]):
        for x in xrange(dims[1]):
            output_image[y][x] = [value, value, output_image[y][x][2]]

    return output_image


def is_red_suit(input_image, thresh, target_regions, percentage_red):
    """
    Provjerava je li karta ima crvenu boju (herc ili karo)
    :param input_image: image for processing
    :param thresh: minimum pixel value
    :param regions: minimum number of red regions to classify
    :param percentage_red: % of how much pixels must be red to count that region as red
    :return: boolean (True or False) depending of picture being red or not
    """
    # dims = input_image.shape
    # clone = input_image.copy()
    blue, green, red = 0, 0, 0
    red_regions = [False, False]
    region_width, region_height = 32, 93

    regions = [input_image[5:region_height, 5:region_width], input_image[247:(247 + 98), 208:(208 + 37)]]
    # print regions
    for i in xrange(2):
        blue, green, red = 0, 0, 0

        for y in xrange(region_height - 5):
            for x in xrange(region_width - 5):
                b, g, r = regions[i][y][x]
                # print x, b, g, r
                if max(r, g, b) == r and r > thresh:
                    red += 1
                elif max(r, g, b) == g and g > thresh:
                    green += 1
                elif max(r, g, b) == b and b > thresh:
                    blue += 1

        total = float(region_width * region_height)
        match_perc = red / total

        red_regions[i] = (red > blue and red > green) and match_perc > percentage_red

    count_red = len([i for i in red_regions if i is True])

    return count_red >= target_regions


def find_squares(input_image):
    """
    Pronalazi broj kvadrata na karti, sto pomaze kod odredjivanja je li karta slika (J, Q, K) ili samo numericka
    :param input_image: image for processing
    :return: returns 1 if image type is numeric, and 2 if image type is royal
    """

    gray = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)  # convert to grayscale
    gray = cv2.GaussianBlur(gray, (5, 5), 0)  # reduce noise
    edged = cv2.Canny(gray, 0, 100)
    cv2.imshow('canny', edged)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    (contours, _) = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:3]

    best_contour = []
    for cnt in contours:
        approx = cv2.approxPolyDP(cnt, cv2.arcLength(cnt, True) * 0.02, True)
        area = cv2.contourArea(approx)
        # print area
        if len(approx) == 4 and area > 40000:
            best_contour.append(approx)

    if len(best_contour) < 1:
        return 1  # karta je numericka
    elif len(best_contour) >= 1:
        return 2  # karta je J || Q || K


def xor(image1, image2, rects):
    """
    XOR nad svim elementima slike (bitwise xor) i pravi crop nad regijom gdje nadje da je znak
    :param image1:
    :param image2:
    :return: returns bounding rectangle for rank and suit
    """
    dims = image1.shape
    xord = image1 ^ image2
    cv2.imshow('xor', xord)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    (contours, _) = cv2.findContours(xord, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    # for x in contours:
        # print "AREA: ", cv2.boundingRect(x), cv2.contourArea(x)
    # get_r1 = [x[1] for x in rects if len(rects)]
    for cnt in contours:
        r = cv2.boundingRect(cnt)
        if r not in rects:
            return r


def hit_or_miss(image, template):
    """
    Uzima dvije 1-channel slike i vraca postotak slicnosti izmedju njih
    :param image: Image to compare vs template
    :param template: template of sign
    :return: percentage of similarity between compared images
    """

    total = image.shape[0] * image.shape[1]
    min_height = min(image.shape[0], template.shape[0])

    matched = 0

    for y in xrange(min_height):
        for x in xrange(image.shape[1]):
            if np.equal(image[y][x].all(), template[y][x].all()):
                matched += 1

    return matched / float(total)