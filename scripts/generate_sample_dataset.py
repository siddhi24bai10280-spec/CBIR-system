"""
generate_sample_dataset.py
-----------------------------
Generates a small synthetic image dataset so the CBIR pipeline can be
demoed and tested end-to-end without requiring an external dataset
download. Creates simple shape/color images across a few visual
"categories" (circles, squares, triangles, stripes) with random color
and size jitter, so genuine visual similarity clusters exist for the
retrieval system to find.

Run:
    python scripts/generate_sample_dataset.py
"""

import os
import random
import numpy as np
import cv2

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "images")
QUERY_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "query")
IMAGES_PER_CATEGORY = 8
IMAGE_SIZE = 300


def random_color():
    return tuple(int(c) for c in np.random.randint(50, 255, size=3))


def draw_circles(canvas, color):
    for _ in range(random.randint(3, 6)):
        center = (random.randint(30, IMAGE_SIZE - 30), random.randint(30, IMAGE_SIZE - 30))
        radius = random.randint(15, 40)
        cv2.circle(canvas, center, radius, color, -1)
    return canvas


def draw_squares(canvas, color):
    for _ in range(random.randint(3, 6)):
        x, y = random.randint(0, IMAGE_SIZE - 60), random.randint(0, IMAGE_SIZE - 60)
        size = random.randint(30, 60)
        cv2.rectangle(canvas, (x, y), (x + size, y + size), color, -1)
    return canvas


def draw_triangles(canvas, color):
    for _ in range(random.randint(3, 6)):
        x, y = random.randint(30, IMAGE_SIZE - 30), random.randint(30, IMAGE_SIZE - 30)
        s = random.randint(20, 45)
        pts = np.array([[x, y - s], [x - s, y + s], [x + s, y + s]], np.int32)
        cv2.fillPoly(canvas, [pts], color)
    return canvas


def draw_stripes(canvas, color):
    thickness = random.randint(10, 25)
    for offset in range(-IMAGE_SIZE, IMAGE_SIZE, thickness * 2):
        cv2.line(canvas, (offset, 0), (offset + IMAGE_SIZE, IMAGE_SIZE), color, thickness)
    return canvas


CATEGORIES = {
    "circles": draw_circles,
    "squares": draw_squares,
    "triangles": draw_triangles,
    "stripes": draw_stripes,
}


def generate_image(draw_fn) -> np.ndarray:
    background = tuple(int(c) for c in np.random.randint(200, 255, size=3))
    canvas = np.full((IMAGE_SIZE, IMAGE_SIZE, 3), background, dtype=np.uint8)
    color = random_color()
    return draw_fn(canvas, color)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(QUERY_DIR, exist_ok=True)
    random.seed(7)
    np.random.seed(7)

    count = 0
    for category, draw_fn in CATEGORIES.items():
        for i in range(IMAGES_PER_CATEGORY):
            image = generate_image(draw_fn)
            filename = f"{category}_{i:02d}.jpg"
            cv2.imwrite(os.path.join(OUTPUT_DIR, filename), image)
            count += 1

    # A few held-out query images, one per category, NOT included in the index folder
    for category, draw_fn in CATEGORIES.items():
        image = generate_image(draw_fn)
        cv2.imwrite(os.path.join(QUERY_DIR, f"query_{category}.jpg"), image)

    print(f"Generated {count} dataset images in {os.path.abspath(OUTPUT_DIR)}")
    print(f"Generated {len(CATEGORIES)} query images in {os.path.abspath(QUERY_DIR)}")


if __name__ == "__main__":
    main()
