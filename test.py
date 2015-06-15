import sys
sys.dont_write_bytecode = True

from transformation import transform
import imutils, numpy as np, cv2, argparse
from skimage.filter import threshold_adaptive
from skimage import img_as_ubyte
from imgproc import *
from classification import *
import time
from glob import glob


# fensi uredjivanje za pozivanje programa
ap = argparse.ArgumentParser()
ap.add_argument('-i', '--image', required=True, help="Path to image")
# ap.add_argument('-t', '--templates', required=True, help="Path to templates image")
args = vars(ap.parse_args())


# for image_path in glob(args["image"] + "/*.jpg"):
start_time = time.time()
image = cv2.imread(args["image"])
ratio, orig = image.shape[0] / 350., image.copy()  # keep track of original image

image = imutils.resize(image, height=350)

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # convert to grayscale
# gray = img_as_ubyte(threshold_adaptive(gray, 200, offset=10))
# cv2.imshow("thresh", gray)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
gray = cv2.GaussianBlur(gray, (5, 5), 0)  # reduce noise
# cv2.imwrite('blurred.jpg', gray)
edged_img = cv2.Canny(gray, 75, 200)  # wide threshold za cannya
# cv2.imshow("Image", image)
# cv2.imshow("Edged image", edged_img)
# cv2.imwrite('edged.jpg', edged_img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# sada treba naci konture slike kojoj smo nasli bridove
# koristimo RETR_LIST -> dobit cemo sve konture van
#           CHAIN_APPROX_SIMPLE -> sejvanje kontura, aproksimacija povezanosti
(contours, _) = cv2.findContours(edged_img.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
# cv2.drawContours(image, contours, -1, (0, 255, 0), 2)
# cv2.imwrite('sve_konture.jpg', image.copy())
# od svih kontura, uzima najvece 4 te ih sortira silazno po povrsini konture
contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]
relevant_areas = [contours[0], contours[1]]
# print contours
# squares = []
best_contour = []
for cnt in contours:

    perimeter = cv2.arcLength(cnt, True)  # racuna opseg konture
    approx_curve = cv2.approxPolyDP(cnt, 0.02 * perimeter, True)  # aproksimacija krivulje

    # ako approx_curve da 4 vrha, mozemo biti sigurni da je to nas trazeni objekt
    if len(approx_curve) == 4:
        best_contour = approx_curve
        break

    # print len(best_contour)
    # nacrtaj objekt na temelju nadjene konture, zelenom bojom
cv2.drawContours(image, [best_contour], -1, (0, 255, 0), 2)
# cv2.imshow("Outline", image)
# cv2.imwrite("outline.jpg", image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# sada je potrebno podignuti sliku, koja je pod nekim kutem != 90 stupnjeva u top-down perspektivu

warped_image = transform(orig, best_contour.reshape(4, 2) * ratio)
# print "SHAPE: ", warped_image.shape
# grey_warped_image = cv2.cvtColor(warped_image, cv2.COLOR_BGR2GRAY)
# black_and_white = threshold_adaptive(grey_warped_image, 250, offset=10) # napravi binarnu sliku, crno-bijelu
# black_and_white = black_and_white.astype("uint8") * 255
# cv2.imshow("Original", imutils.resize(orig, height=350))
warped_image = imutils.resize(warped_image, width=250, height=350)
# cv2.imwrite("blobcount.jpg", warped_image)
# print "SHAPE 2: ", warped_image.shape
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# warped_image = warped_image.astype("uint8") * 255
# cv2.imshow("Warped perspective", imutils.resize(warped_image, height=350))
# cv2.waitKey(0)
# cv2.destroyAllWindows()
type = detect_type(warped_image)  # 1 -> numericka, 2 -> royal
# print "Tip je ", type
red_image, is_red = detect_colour(warped_image)
# cv2.imwrite('crna.jpg', red_image)
rank_dim, suit_dim = find_symbols(warped_image)
x_r, y_r, w_r, h_r = rank_dim
x_s, y_s, w_s, h_s = suit_dim
cv2.rectangle(warped_image, (x_r, y_r), (x_r + w_r, y_r + h_r), (255, 0, 0), 1)
cv2.rectangle(warped_image, (x_s, y_s), (x_s + w_s, y_s + h_s), (255, 0, 0), 1)
# cv2.imshow("KUTEVI", warped_image)
# print rank_dim, suit_dim
rank = warped_image[y_r + 1:y_r + h_r - 1, x_r + 1:x_r + w_r - 1]
suit = warped_image[y_s + 1:y_s + h_s - 1, x_s + 1:x_s + w_s - 1]
# cv2.imshow("Rank", rank)
# cv2.imwrite("rankjack.jpg", rank)
# cv2.imshow("Suit", suit)
# cv2.imwrite("suitsym2.jpg", suit)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
if type == 1:
    # detektiraj vrijednost karte, ukoliko je numericka
    rank_sym = detect_value(warped_image)
    # print count_blobs
    if rank_sym <= 1:   # onda je A, posebno raspoznavanje
        rank_sym = 'A'
else:  # tip je "royal"
    rank_sym = detect_value_picture(rank)

suit_sym = detect_suit(suit, is_red)
end_time = round(time.time() - start_time, 2)
# text = str(rank_sym) + " " + str(suit_sym)
for_text = np.ones(warped_image.shape)
# print "VRIJEME: ", str()
cv2.putText(for_text, "Color: " + "Red" if is_red else "Color: " + "Black", (30, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0))
cv2.putText(for_text, "Rank: " + str(rank_sym), (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0))
cv2.putText(for_text, "Suit: " + str(suit_sym), (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0))
cv2.putText(for_text, "Time: " + str(end_time) + "s", (30, 120), cv2.FONT_HERSHEY_SIMPLEX,
            1.0, (0, 0, 0))
cv2.imshow("Original", imutils.resize(orig, height=350))
cv2.imshow("Recognition", warped_image)
cv2.imshow("Card Info", for_text)
# cv2.imwrite("recognition.jpg", warped_image)
# cv2.imwrite("info.jpg", for_text)
cv2.waitKey(0)
cv2.destroyAllWindows()