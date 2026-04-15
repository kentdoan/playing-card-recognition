import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from helpers import extract_cards

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)

TEST_DIR = os.path.join(_PROJECT_ROOT, 'test')
TEMPLATE_FILE = os.path.join(_SCRIPT_DIR, 'card_templates.npz')

CORNER_W = 70
CORNER_H = 95
RATIO_TEST_THRESHOLD = 0.75


def normalize_warp(warp):
    if warp.dtype != np.uint8:
        warp = warp.astype(np.uint8)
    if len(warp.shape) == 3:
        warp = cv2.cvtColor(warp, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(warp, (3, 3), 0)
    _, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary


def corner_strip(card_binary):
    top_left = card_binary[:CORNER_H, :CORNER_W]
    bottom_right = card_binary[-CORNER_H:, -CORNER_W:]
    bottom_right = np.rot90(bottom_right, 2)
    return np.hstack([top_left, bottom_right])


def template_score(test_binary, template_binary):
    test_strip = corner_strip(test_binary)
    template_strip = corner_strip(template_binary)
    corner_score = np.sum(cv2.absdiff(test_strip, template_strip))
    full_score = np.sum(cv2.absdiff(test_binary, template_binary))
    return 0.8 * corner_score + 0.2 * full_score


def good_match_count(matcher, des_query, des_train, ratio_threshold):
    if des_query is None or des_train is None:
        return 0
    if len(des_query) < 2 or len(des_train) < 2:
        return 0
    knn = matcher.knnMatch(des_query, des_train, k=2)
    good = 0
    for pair in knn:
        if len(pair) < 2:
            continue
        m, n = pair
        if m.distance < ratio_threshold * n.distance:
            good += 1
    return good

def test_image(image_filename, method='template'):
    if not os.path.exists(TEMPLATE_FILE):
        print("Error: template file not found. Run training first.")
        return

    img_path = os.path.join(TEST_DIR, image_filename)
    detected_cards, original_img = extract_cards(img_path)

    if original_img is None:
        print("Error: test image could not be loaded.")
        return

    img_disp = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)

    npz_file = np.load(TEMPLATE_FILE)
    templates = {name: npz_file[name] for name in npz_file.files}

    feature_extractor = None
    matcher = None
    template_features = {}

    if method == 'sift':
        feature_extractor = cv2.SIFT_create()
        matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    elif method == 'orb':
        feature_extractor = cv2.ORB_create()
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    if feature_extractor is not None:
        for name, temp_warp in templates.items():
            temp_norm = normalize_warp(temp_warp)
            _, des = feature_extractor.detectAndCompute(temp_norm, None)
            template_features[name] = des

    for test_card in detected_cards:
        test_warp = normalize_warp(test_card['warped'])
        test_warp_180 = np.rot90(test_warp, 2)

        best_match = "Unknown"

        if method == 'template':
            best_score = float('inf')
            for name, temp_warp in templates.items():
                temp_norm = normalize_warp(temp_warp)
                score1 = template_score(test_warp, temp_norm)
                score2 = template_score(test_warp_180, temp_norm)
                match_score = min(score1, score2)

                if match_score < best_score:
                    best_score = match_score
                    best_match = name

        else:
            best_score = -1
            _, des_test_0 = feature_extractor.detectAndCompute(test_warp, None)
            _, des_test_180 = feature_extractor.detectAndCompute(test_warp_180, None)

            for name, des_temp in template_features.items():
                if des_temp is None:
                    continue

                score_0 = good_match_count(matcher, des_test_0, des_temp, RATIO_TEST_THRESHOLD)
                score_180 = good_match_count(matcher, des_test_180, des_temp, RATIO_TEST_THRESHOLD)
                match_score = max(score_0, score_180)

                if match_score > best_score:
                    best_score = match_score
                    best_match = name

        cv2.drawContours(img_disp, [test_card['contour']], -1, (255, 0, 255), 3)

        cx, cy = int(test_card['center'][0]), int(test_card['center'][1])
        cv2.putText(img_disp, best_match, (cx - 50, cy), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 255), 3)

    plt.figure(figsize=(10, 10))
    plt.imshow(img_disp)
    plt.axis('off')
    plt.title(f"Method: {method.upper()} | Result: {image_filename}", fontsize=14)
    plt.show()