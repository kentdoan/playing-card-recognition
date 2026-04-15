# uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload 
import os
import time
from typing import Dict, List, Tuple

import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

CARD_W, CARD_H = 200, 300
CORNER_W = 70
CORNER_H = 95
RATIO_TEST_THRESHOLD = 0.75
SUPPORTED_METHODS = {"template", "sift", "orb"}

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BASE_DIR)
TEMPLATE_FILE = os.path.join(PROJECT_DIR, "src", "card_templates.npz")

app = FastAPI(title="Card Recognition API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def order_points(pts: np.ndarray) -> np.ndarray:
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


def extract_cards_from_image(img: np.ndarray) -> List[Dict[str, np.ndarray]]:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    kernel_close = np.ones((5, 5), dtype=np.uint8)
    kernel_open = np.ones((3, 3), dtype=np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel_close, iterations=1)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel_open, iterations=1)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    cards: List[Dict[str, np.ndarray]] = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 20000:
            continue

        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)

        if len(approx) != 4:
            continue

        pts = approx.reshape(4, 2)
        rect = order_points(pts)
        dst = np.array(
            [[0, 0], [CARD_W - 1, 0], [CARD_W - 1, CARD_H - 1], [0, CARD_H - 1]],
            dtype="float32",
        )

        transform = cv2.getPerspectiveTransform(rect, dst)
        warped_binary = cv2.warpPerspective(thresh, transform, (CARD_W, CARD_H))

        cards.append(
            {
                "contour": cnt,
                "warped": warped_binary,
                "center": np.mean(pts, axis=0),
                "area": area,
            }
        )

    cards.sort(key=lambda x: float(x["area"]), reverse=True)
    return cards


def normalize_warp(warp: np.ndarray) -> np.ndarray:
    if warp.dtype != np.uint8:
        warp = warp.astype(np.uint8)
    if len(warp.shape) == 3:
        warp = cv2.cvtColor(warp, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(warp, (3, 3), 0)
    _, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary


def corner_strip(card_binary: np.ndarray) -> np.ndarray:
    top_left = card_binary[:CORNER_H, :CORNER_W]
    bottom_right = card_binary[-CORNER_H:, -CORNER_W:]
    bottom_right = np.rot90(bottom_right, 2)
    return np.hstack([top_left, bottom_right])


def template_score(test_binary: np.ndarray, template_binary: np.ndarray) -> float:
    test_strip = corner_strip(test_binary)
    template_strip = corner_strip(template_binary)
    corner_score = np.sum(cv2.absdiff(test_strip, template_strip))
    full_score = np.sum(cv2.absdiff(test_binary, template_binary))
    return 0.8 * float(corner_score) + 0.2 * float(full_score)


def good_match_count(
    matcher: cv2.BFMatcher,
    des_query: np.ndarray,
    des_train: np.ndarray,
    ratio_threshold: float,
) -> int:
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


def _load_templates() -> Dict[str, np.ndarray]:
    if not os.path.exists(TEMPLATE_FILE):
        raise FileNotFoundError(
            "Template file not found at src/card_templates.npz. Run training first."
        )
    npz_file = np.load(TEMPLATE_FILE)
    return {name: npz_file[name] for name in npz_file.files}


def recognize_cards_from_cv2(image_bgr: np.ndarray, method: str = "template") -> List[str]:
    method = method.lower().strip()
    if method not in SUPPORTED_METHODS:
        raise ValueError(f"Unsupported method: {method}")

    templates = _load_templates()
    detected_cards = extract_cards_from_image(image_bgr)

    if not detected_cards:
        return []

    feature_extractor = None
    matcher = None
    template_features: Dict[str, np.ndarray] = {}

    if method == "sift":
        feature_extractor = cv2.SIFT_create()
        matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    elif method == "orb":
        feature_extractor = cv2.ORB_create()
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    if feature_extractor is not None:
        for name, temp_warp in templates.items():
            temp_norm = normalize_warp(temp_warp)
            _, des = feature_extractor.detectAndCompute(temp_norm, None)
            template_features[name] = des

    results: List[str] = []
    for test_card in detected_cards:
        test_warp = normalize_warp(test_card["warped"])
        test_warp_180 = np.rot90(test_warp, 2)

        best_match = "Unknown"

        if method == "template":
            best_score = float("inf")
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

                score_0 = good_match_count(
                    matcher, des_test_0, des_temp, RATIO_TEST_THRESHOLD
                )
                score_180 = good_match_count(
                    matcher, des_test_180, des_temp, RATIO_TEST_THRESHOLD
                )
                match_score = max(score_0, score_180)

                if match_score > best_score:
                    best_score = match_score
                    best_match = name

        results.append(best_match)

    return results


@app.get("/")
def health_check() -> Dict[str, str]:
    return {"status": "ok", "service": "card-recognition-api"}


@app.post("/api/recognize")
async def recognize(file: UploadFile = File(...), method: str = "template") -> JSONResponse:
    start_time = time.perf_counter()

    try:
        image_bytes = await file.read()
        if not image_bytes:
            return JSONResponse(
                status_code=400,
                content={
                    "success": False,
                    "result": None,
                    "processing_time": "0.000s",
                    "error": "Empty file uploaded",
                },
            )

        np_buffer = np.frombuffer(image_bytes, dtype=np.uint8)
        image_bgr = cv2.imdecode(np_buffer, cv2.IMREAD_COLOR)

        if image_bgr is None:
            return JSONResponse(
                status_code=400,
                content={
                    "success": False,
                    "result": None,
                    "processing_time": "0.000s",
                    "error": "Invalid image data",
                },
            )

        recognized_cards = recognize_cards_from_cv2(image_bgr, method=method)

        processing_time = f"{time.perf_counter() - start_time:.3f}s"
        if not recognized_cards:
            return JSONResponse(
                status_code=200,
                content={
                    "success": True,
                    "result": "Unknown",
                    "processing_time": processing_time,
                },
            )

        result_text = recognized_cards[0] if len(recognized_cards) == 1 else ", ".join(recognized_cards)
        return JSONResponse(
            status_code=200,
            content={
                "success": True,
                "result": result_text,
                "processing_time": processing_time,
            },
        )
    except Exception as exc:
        processing_time = f"{time.perf_counter() - start_time:.3f}s"
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "result": None,
                "processing_time": processing_time,
                "error": str(exc),
            },
        )
