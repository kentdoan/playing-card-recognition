import cv2
import numpy as np

CARD_W, CARD_H = 200, 300


def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    d1 = np.linalg.norm(rect[0] - rect[1])
    d2 = np.linalg.norm(rect[1] - rect[2])
    if d1 > d2:
        rect = np.roll(rect, -1, axis=0)
    return rect


def extract_cards(img_path, is_training=False):
    img = cv2.imread(img_path)
    if img is None:
        return [], None

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    kernel_close = np.ones((5, 5), dtype=np.uint8)
    kernel_open = np.ones((3, 3), dtype=np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel_close, iterations=1)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel_open, iterations=1)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    cards = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if not is_training and area < 20000:
            continue

        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)

        if len(approx) == 4:
            pts = approx.reshape(4, 2)
            rect = order_points(pts)

            dst = np.array([[0, 0], [CARD_W - 1, 0], [CARD_W - 1, CARD_H - 1], [0, CARD_H - 1]], dtype="float32")
            M = cv2.getPerspectiveTransform(rect, dst)

            warped_binary = cv2.warpPerspective(thresh, M, (CARD_W, CARD_H))

            cards.append({
                'contour': cnt,
                'warped': warped_binary,
                'center': np.mean(pts, axis=0)
            })

    if is_training and len(cards) > 0:
        cards = [max(cards, key=lambda c: cv2.contourArea(c['contour']))]

    return cards, img
