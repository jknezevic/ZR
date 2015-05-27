import numpy as np, cv2
from imgproc import *
from skimage.filter import threshold_adaptive
from scipy.ndimage.morphology import binary_closing, binary_dilation, binary_erosion, binary_hit_or_miss, binary_opening
from imutils import *


def detect_colour(input_image):
    """
    Detektira boju karte (crvena ili crna)
    :param input_image: image for processing
    :return: returns transformed image and True or False, whether image is red (True) or black (False)
    """
    black_b = black_background(input_image, 100)
    input_image = filter_red(black_b, 0)
    is_red = is_red_suit(input_image, 100, 2, 0.10)

    # print is_red
    # cv2.imshow("Black", black_b)
    # cv2.imshow("Filter red", input_image)
    # cv2.imwrite('crvena.jpg', filter_r)

    return input_image, is_red


def detect_type(input_image):
    """
    Provjerava tip karte, vraca 1 ako je numericka, 2 ako je "kraljevska"
    :param input_image: image for processing
    :return: returns if there is another square in card
    """
    return find_squares(input_image)


def find_symbols(input_image):
    """
    Pronalazi na karti pozicije broja i boje te ih ekstraktira u nove matrice
    :param input_image: image for processing
    :return: dimensions of new image matrix for rank and suit
    """
    grey_warped_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)
    black_and_white = threshold_adaptive(grey_warped_image, 250, offset=10)  # napravi binarnu sliku, crno-bijelu
    black_and_white = black_and_white.astype("uint8") * 255

    kernel = np.ones((3, 2), 'uint8')
    # print black_and_white[20][20]
    black_and_white = cv2.erode(black_and_white, kernel, iterations=1)
    # cv2.imshow('Erodirana', black_and_white)
    blob_found = False
    region_width, region_height = 32, 93
    rect_top, rect_bot = input_image[5:region_height, 5:region_width], input_image[247:(247 + 98), 208:(208 + 37)]

    blob_found = False
    region_width, region_height = 32, 93
    # rect_top, rect_bot = input_image[5:region_height, 5:region_width], input_image[247:(247 + 98), 208:(208 + 37)]
    # print black_and_white.shape
    mask = np.zeros((black_and_white.shape[0] + 2, black_and_white.shape[1] + 2), 'uint8')
    bin_card = black_and_white.copy()
    rects = []
    for y in xrange(5, region_height):
        for x in xrange(5, region_width):
            bgr = black_and_white[y][x]
            if bgr == 0:
                cv2.floodFill(black_and_white, mask, (x, y), (255, 255, 255))
                cv2.imshow("flooded", black_and_white)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
                rects.append(xor(black_and_white, bin_card, rects))

    print "RECTS: ", rects
    if len(rects) < 3:
        rank_dim, suit_dim = rects[0], rects[1]
    else:
        x1, y1, w1, h1 = rects[0]
        x2, y2, w2, h2 = rects[1]
        rank_dim = (x1, y1, w1 + w2 + 2, h1)
        suit_dim = rects[2]

    return rank_dim, suit_dim


def detect_value(input_image):
    """
    Vraca vrijednost karte (2 - 10)
    :param input_image: image for processing
    :return: value of input image
    """
    grey_warped_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)
    black_and_white = threshold_adaptive(grey_warped_image, 250, offset=10)  # napravi binarnu sliku, crno-bijelu
    black_and_white = black_and_white.astype("uint8") * 255

    kernel = np.ones((3, 3), 'uint8')
    # ignoriraj kuteve
    region_width, region_height = 32, 93
    rect_top, rect_bot = black_and_white[5:region_height, 5:region_width], black_and_white[247:(247 + 98),
                                                                           208:(208 + 37)]

    cv2.rectangle(black_and_white, (2, 2), (2 + region_width, 2 + region_height), (255, 255, 255), -1)
    cv2.rectangle(black_and_white, (218, 263), (218 + 29, 263 + 82), (255, 255, 255), -1)

    # cv2.imshow("bez kuteva", black_and_white)

    black_and_white = cv2.dilate(black_and_white, kernel, iterations=1)

    # cv2.imshow("Dilate", black_and_white)

    black_and_white = cv2.morphologyEx(black_and_white, cv2.MORPH_CLOSE, kernel=np.ones((5, 5), 'uint8'), iterations=3)
    mask = np.zeros((black_and_white.shape[0] + 2, black_and_white.shape[1] + 2), 'uint8')
    # cv2.imshow("Closed", black_and_white)
    count_blobs = 0
    for y in xrange(black_and_white.shape[0]):
        for x in xrange(black_and_white.shape[1]):
            if black_and_white[y][x] == 0:
                count_blobs += 1
                cv2.floodFill(black_and_white, mask, (x, y), (255, 255, 255))

    return count_blobs


def detect_value_picture(input_image):
    """

    :param input_image: image for processing
    :return: returns whether a card is Jack, Queen or King
    """
    grey_warped_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)
    black_and_white = threshold_adaptive(grey_warped_image, 250, offset=10)  # napravi binarnu sliku, crno-bijelu
    black_and_white = black_and_white.astype("uint8") * 255

    black_and_white = resize(black_and_white, width=100, height=100)

    # print "BW shape", black_and_white.shape

    cv2.imshow("one channel", black_and_white)
    jack, queen, king = cv2.imread("simboli/jack.png"), cv2.imread("simboli/queen.png"), cv2.imread("simboli/king.png")

    # jack_cnt, queen_cnt, king_cnt = 0, 0, 0
    jack = resize(jack, width=100, height=100)
    queen = resize(queen, width=100, height=100)
    king = resize(king, width=100, height=100)

    grey_jack = cv2.cvtColor(jack, cv2.COLOR_BGR2GRAY)
    jack_bw = threshold_adaptive(grey_jack, 250, offset=10)  # napravi binarnu sliku, crno-bijelu
    jack_bw = jack_bw.astype("uint8") * 255

    # print "Jack shape", jack_bw.shape
    # cv2.imshow("Jack BW", jack_bw)

    grey_queen = cv2.cvtColor(queen, cv2.COLOR_BGR2GRAY)
    queen_bw = threshold_adaptive(grey_queen, 250, offset=10)  # napravi binarnu sliku, crno-bijelu
    queen_bw = queen_bw.astype("uint8") * 255

    # print "Queen shape", queen_bw.shape
    # cv2.imshow("Queen BW", queen_bw)

    grey_king = cv2.cvtColor(king, cv2.COLOR_BGR2GRAY)
    king_bw = threshold_adaptive(grey_king, 250, offset=10)  # napravi binarnu sliku, crno-bijelu
    king_bw = king_bw.astype("uint8") * 255

    # print "King shape", king_bw.shape
    # cv2.imshow("King BW", king_bw)

    # print "Black and white size" + str(black_and_white.shape) + "\n HOM size" + str(jack_bw.shape)
    jack_hom = hit_or_miss(black_and_white, jack_bw)
    king_hom = hit_or_miss(black_and_white, king_bw)
    queen_hom = hit_or_miss(black_and_white, queen_bw)

    # print "Black and white size" + str(black_and_white.shape) + "\n HOM size" + str(jack_hom.shape)


    print "VJEROJATNOSTI SLIKA: "
    print "DECKO: ", jack_hom
    print "DAMA: ", queen_hom
    print "KRALJ: ", king_hom

    if max(jack_hom, queen_hom, king_hom) == jack_hom:
        return "Decko"
    elif max(jack_hom, queen_hom, king_hom) == queen_hom:
        return "Dama"
    elif max(jack_hom, queen_hom, king_hom) == king_hom:
        return "Kralj"
    else:
        return "###FAIL###"


def detect_suit(input_image, is_red):
    """
    Detektira "boju" karte (pik, herc, karo, tref)
    :param input_image: image for processing
    :param is_red: flag, if 1 -> check just hearts and diamonds, else check clubs and spades
    :return: returns card suit
    """

    grey_warped_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)
    black_and_white = threshold_adaptive(grey_warped_image, 250, offset=10)  # napravi binarnu sliku, crno-bijelu
    black_and_white = black_and_white.astype("uint8") * 255
    black_and_white = resize(black_and_white, width=100, height=100)

    hearts, diamonds = cv2.imread("simboli/heart.png"), cv2.imread("simboli/diamond.png")  # crveni
    spades, clubs = cv2.imread("simboli/spade.png"), cv2.imread("simboli/club.png")  # crne

    hearts = resize(hearts, width=100, height=100)
    diamonds = resize(diamonds, width=100, height=100)
    spades = resize(spades, width=100, height=100)
    clubs = resize(clubs, width=100, height=100)

    grey_hearts_image = cv2.cvtColor(hearts, cv2.COLOR_BGR2GRAY)
    hearts_bw = threshold_adaptive(grey_hearts_image, 250, offset=10)  # napravi binarnu sliku, crno-bijelu
    hearts_bw = hearts_bw.astype("uint8") * 255

    grey_diamonds_image = cv2.cvtColor(diamonds, cv2.COLOR_BGR2GRAY)
    diamonds_bw = threshold_adaptive(grey_diamonds_image, 250, offset=10)  # napravi binarnu sliku, crno-bijelu
    diamonds_bw = diamonds_bw.astype("uint8") * 255

    grey_spades_image = cv2.cvtColor(spades, cv2.COLOR_BGR2GRAY)
    spades_bw = threshold_adaptive(grey_spades_image, 250, offset=10)  # napravi binarnu sliku, crno-bijelu
    spades_bw = spades_bw.astype("uint8") * 255

    grey_clubs_image = cv2.cvtColor(clubs, cv2.COLOR_BGR2GRAY)
    clubs_bw = threshold_adaptive(grey_clubs_image, 250, offset=10)  # napravi binarnu sliku, crno-bijelu
    clubs_bw = clubs_bw.astype("uint8") * 255

    if is_red is True:
        hearts_hom = hit_or_miss(black_and_white, hearts_bw)
        diamonds_hom = hit_or_miss(black_and_white, diamonds_bw)
        if max(hearts_hom, diamonds_hom) == hearts_hom:
            return "Herc"
            # is_hearts = True
        elif max(hearts_hom, diamonds_hom) == diamonds_hom:
            return "Karo"
            # is_diamonds = True
    elif is_red is False:
        spades_hom = hit_or_miss(black_and_white, spades_bw)
        clubs_hom = hit_or_miss(black_and_white, clubs_bw)
        if max(spades_hom, clubs_hom) == spades_hom:
            return "Pik"
            # is_spades = True
        elif max(spades_hom, clubs_hom) == clubs_hom:
            return "Tref"
            # is_clubs = True
    else:
        return "Nisam dobro prepoznao boju!"


