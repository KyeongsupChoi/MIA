"""Display a chest X-ray image using OpenCV."""

import argparse
from pathlib import Path

import cv2 as cv


def show_image(image_path: str | Path, window_name: str = 'X-ray') -> None:
    """Display a single image in an OpenCV window.

    Raises:
        FileNotFoundError: If image_path does not exist.
        ValueError: If the image cannot be read.
    """
    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f'image not found: {image_path}')

    img = cv.imread(str(image_path))
    if img is None:
        raise ValueError(f'could not decode image: {image_path}')

    cv.imshow(window_name, img)
    cv.waitKey(0)
    cv.destroyAllWindows()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Display a chest X-ray')
    parser.add_argument('image_path', type=str, help='Path to image file')
    args = parser.parse_args()
    show_image(args.image_path)
