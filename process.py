import hashlib
import os
import cv2 as cv
import numpy as np

puzzle_dir_path = './days'
characters_dir_path = './characters'


def process_puzzle(puzzle):
    puzzle_id = puzzle.name.split('.')[0]
    print("processing", puzzle.path)
    puzzle_rgb = cv.imread(puzzle.path)
    assert puzzle_rgb is not None, "file could not be read, check with os.path.exists()"
    puzzle_gray = cv.cvtColor(puzzle_rgb, cv.COLOR_BGR2GRAY)

    characters_dir = os.scandir(characters_dir_path)
    for character in characters_dir:
        if character.is_file():
            template = cv.imread(character.path, cv.IMREAD_GRAYSCALE)
            assert template is not None, "file could not be read, check with os.path.exists()"
            w, h = template.shape[::-1]

            color_hash = hashlib.sha256()
            color_hash.update(template)
            hex_digest = color_hash.hexdigest()[:6]
            r = int(hex_digest[0:2], 16)
            g = int(hex_digest[2:4], 16)
            b = int(hex_digest[4:6], 16)

            res = cv.matchTemplate(puzzle_gray, template, cv.TM_CCOEFF_NORMED)
            threshold = 0.8
            loc = np.where(res >= threshold)
            for pt in zip(*loc[::-1]):
                cv.rectangle(puzzle_rgb, pt, (pt[0] + w,
                                              pt[1] + h), (r, g, b, 255), 2)

            cv.imwrite(os.path.join('output', puzzle_id + '.png'), puzzle_rgb)


puzzle_dir = os.scandir(puzzle_dir_path)
for puzzle in puzzle_dir:
    if puzzle.is_file():
        process_puzzle(puzzle)
