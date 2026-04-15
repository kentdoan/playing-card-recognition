# Playing Card Recognition

This project detects and recognizes playing cards from a single input image containing multiple cards. The pipeline extracts each card, normalizes its perspective, and identifies the card using one of three matching methods: Template Matching, SIFT, or ORB.

## Overview

The system follows a classical computer vision approach:

1. Preprocess the input image.
2. Detect card contours.
3. Warp each card to a fixed size of `200x300`.
4. Match the card against the template database.
5. Draw the predicted label on the original image.

The project also handles card rotation by checking both `0°` and `180°` orientations.

## Features

- Detect multiple cards in one image.
- Normalize card perspective before recognition.
- Support three recognition methods:
  - Template Matching
  - SIFT
  - ORB
- Automatically build the template database from 52 training images if it does not already exist.
- Visualize the final result with bounding contours and predicted labels.

## Project Structure

```text
playing-card-recognition/
├── dataset/              # 52 training images, one per card
├── src/
│   ├── main.py           # Entry point
│   ├── train.py          # Builds the template database
│   ├── test.py           # Runs recognition on test images
│   ├── helpers.py        # Card extraction and perspective warp
│   └── card_templates.npz
├── test/                 # Test images
├── README.md
└── requirements.txt
```

## Requirements

- Python 3
- OpenCV
- NumPy
- Matplotlib

Install dependencies with:

```bash
pip install -r requirements.txt
```

## Dataset

The training set must contain 52 single-card images in `dataset/`, named with the following convention:

```text
<rank><suit>.jpg
```

Examples:

- `Ahearts.jpg`
- `10spades.jpg`
- `Qclubs.jpg`

Supported suits:

- `diamonds`
- `clubs`
- `hearts`
- `spades`

Supported ranks:

- `A`, `2`, `3`, `4`, `5`, `6`, `7`, `8`, `9`, `10`, `J`, `Q`, `K`

## How It Works

### 1. Training

`train.py` reads all 52 training images, extracts the largest clear card from each image, warps it to a standard binary template, and saves the results into `src/card_templates.npz`.

### 2. Recognition

`test.py` loads the template database and compares each detected card against the templates.

- **Template Matching** uses a weighted SAD score, with extra weight on the top-left and bottom-right corners.
- **SIFT** uses `BFMatcher` with `L2` distance and Lowe's ratio test.
- **ORB** uses `BFMatcher` with `Hamming` distance and Lowe's ratio test.

For each detected card, the system also checks the rotated `180°` version and keeps the better result.

## Usage

Run the project from the `src/` folder:

```bash
python main.py --method template
```

### Select a recognition method

```bash
python main.py --method template
python main.py --method sift
python main.py --method orb
```

### Choose a test image

```bash
python main.py --method sift --image Cards_2.jpg
```

If the template file does not exist, the program automatically runs training first.

## Output

The result is displayed as an annotated image showing (opencv window):

- detected card contours
- predicted card labels

## Notes

- The default test image is `Cards_2.jpg`.
- The template file is generated automatically as `src/card_templates.npz`.