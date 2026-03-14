"""Count image files in a directory tree."""

import argparse
import os
from pathlib import Path

IMAGE_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff')


def count_images(folder_path: str | Path) -> int:
    """Recursively count image files by extension.

    Raises:
        FileNotFoundError: If folder_path does not exist.
    """
    folder_path = Path(folder_path)
    if not folder_path.is_dir():
        raise FileNotFoundError(f'directory not found: {folder_path}')

    count = 0
    for root, _, files in os.walk(folder_path):
        for f in files:
            if f.lower().endswith(IMAGE_EXTENSIONS):
                count += 1
    return count


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Count images in a folder')
    parser.add_argument(
        'folder', type=str, help='Path to folder containing images',
    )
    args = parser.parse_args()
    n = count_images(args.folder)
    print(f'There are {n} images in {args.folder}.')
