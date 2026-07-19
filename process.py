import csv
import hashlib
import os
from dataclasses import dataclass, replace
from pathlib import Path

import cv2 as cv
import numpy as np


PUZZLE_DIR = Path("days")
CHARACTER_DIR = Path("characters")
OUTPUT_DIR = Path("output")
MISSING_DIR = Path("missing_symbols")
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}
TEMPLATE_HEIGHT = 163
BASELINE_OFFSET = 77
MATCH_THRESHOLD = 0.76
L_MATCH_THRESHOLD = 0.55
MAX_TEMPLATE_OVERLAP = 24
L_WHITESPACE_WIDTH = 8
MISSING_CONTEXT = 24
MAX_REVIEW_CANDIDATES = 32


@dataclass(frozen=True)
class Template:
    name: str
    image: np.ndarray
    color: tuple

    @property
    def height(self):
        return self.image.shape[0]

    @property
    def width(self):
        return self.image.shape[1]


@dataclass(frozen=True)
class Candidate:
    template: Template
    x: int
    y: int
    score: float
    draw_x: int | None = None

    @property
    def start(self):
        return self.x if self.draw_x is None else self.draw_x

    @property
    def end(self):
        return self.x + self.template.width


def natural_key(path):
    try:
        return int(path.stem)
    except ValueError:
        return path.stem


def color_for(image):
    digest = hashlib.sha256(image.tobytes()).hexdigest()[:6]
    return tuple(int(digest[index : index + 2], 16) for index in (0, 2, 4))


def load_templates():
    templates = []
    for path in sorted(CHARACTER_DIR.iterdir()):
        if not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        image = cv.imread(os.fspath(path), cv.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError(f"could not read template: {path}")
        if image.shape[0] != TEMPLATE_HEIGHT:
            raise ValueError(
                f"template {path} has height {image.shape[0]}, expected {TEMPLATE_HEIGHT}"
            )
        templates.append(Template(path.stem, image, color_for(image)))
    if not templates:
        raise ValueError("no templates found")
    return templates


def neutral_black(image):
    channel_max = np.max(image, axis=2)
    channel_min = np.min(image, axis=2)
    return (channel_max < 170) & ((channel_max - channel_min) < 24)


def detect_baselines(ink):
    # Cipher rows contain a long horizontal spine. Chinese/English headings are
    # disconnected glyphs and do not survive this horizontal opening.
    opened = cv.morphologyEx(
        ink.astype(np.uint8), cv.MORPH_OPEN, np.ones((1, 81), np.uint8)
    )
    activity = np.count_nonzero(opened, axis=1) >= 45
    activity[: round(ink.shape[0] * 0.10)] = False
    count, _, stats, _ = cv.connectedComponentsWithStats(
        activity.astype(np.uint8).reshape(-1, 1)
    )
    projection = np.count_nonzero(ink, axis=1)
    baselines = []
    for label in range(1, count):
        y = int(stats[label, cv.CC_STAT_TOP])
        height = int(stats[label, cv.CC_STAT_HEIGHT])
        rows = np.arange(y, y + height)
        baseline = int(rows[np.argmax(projection[rows])])
        baselines.append(baseline)
    return sorted(baselines)


def candidates_for_row(gray, templates, row_top, content_start, content_end):
    y0 = max(0, row_top - 6)
    y1 = min(gray.shape[0], row_top + TEMPLATE_HEIGHT + 7)
    roi = gray[y0:y1]
    candidates = []
    for template in templates:
        if roi.shape[0] < template.height or gray.shape[1] < template.width:
            continue
        result = cv.matchTemplate(roi, template.image, cv.TM_CCOEFF_NORMED)
        scores = np.max(result, axis=0)
        offsets = np.argmax(result, axis=0)
        local_max = scores == cv.dilate(
            scores.reshape(1, -1), np.ones((1, 7), np.uint8)
        ).reshape(-1)
        threshold = L_MATCH_THRESHOLD if template.name == "l" else MATCH_THRESHOLD
        xs = np.where((scores >= threshold) & local_max)[0]
        for x in xs:
            if x + template.width < content_start - 10 or x > content_end + 10:
                continue
            candidates.append(
                Candidate(template, int(x), y0 + int(offsets[x]), float(scores[x]))
            )
    return candidates


def ink_area(ink, row_top, start, end):
    if end <= start:
        return 0
    y0 = max(0, row_top)
    y1 = min(ink.shape[0], row_top + TEMPLATE_HEIGHT)
    x0 = max(0, start)
    x1 = min(ink.shape[1], end)
    return int(np.count_nonzero(ink[y0:y1, x0:x1]))


def follows_whitespace(candidate, cursor, content_start):
    if candidate.template.name != "l":
        return True
    return cursor <= content_start + 2 or candidate.x >= cursor + L_WHITESPACE_WIDTH


def greedy_row(candidates, ink, row_top, content_start, content_end):
    matches = []
    missing = []
    cursor = content_start
    candidates = sorted(candidates, key=lambda item: (item.x, -item.score))

    def eligible(candidate):
        return follows_whitespace(candidate, cursor, content_start)

    while cursor < content_end:
        nearby = [
            candidate
            for candidate in candidates
            if cursor - MAX_TEMPLATE_OVERLAP <= candidate.x <= cursor + 8
            and candidate.end > cursor
            and eligible(candidate)
        ]
        if nearby:
            # Exact full symbols score higher than contained sub-shapes. Prefer
            # the wider candidate only when confidence is effectively tied.
            best_score = max(candidate.score for candidate in nearby)
            tied = [
                candidate for candidate in nearby if candidate.score >= best_score - 0.015
            ]
            best = max(tied, key=lambda item: (item.template.width, item.score))
            matches.append(replace(best, draw_x=max(cursor, best.x)))
            cursor = max(cursor + 1, best.end)
            continue

        future = [
            candidate
            for candidate in candidates
            if candidate.x > cursor + 8 and eligible(candidate)
        ]
        next_x = min((candidate.x for candidate in future), default=content_end)
        if next_x - cursor >= 14 and ink_area(ink, row_top, cursor, next_x) >= 60:
            missing.append((cursor, next_x))
        cursor = max(cursor + 1, next_x)

    return matches, missing


def crop_missing(image, ink, row_top, start, end):
    y0 = max(0, row_top)
    y1 = min(image.shape[0], row_top + TEMPLATE_HEIGHT)
    core_x0 = max(0, start)
    core_x1 = min(image.shape[1], end)
    mask = ink[y0:y1, core_x0:core_x1]
    if end - start < 14 or np.count_nonzero(mask) < 60:
        return None
    ys, xs = np.where(mask)
    core = mask[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1]
    if core.shape[0] < 24:
        return None

    review_x0 = max(0, core_x0 - MISSING_CONTEXT)
    review_x1 = min(image.shape[1], core_x1 + MISSING_CONTEXT)
    review = ink[y0:y1, review_x0:review_x1]
    review_ys, review_xs = np.where(review)
    review = review[
        review_ys.min() : review_ys.max() + 1,
        review_xs.min() : review_xs.max() + 1,
    ]
    return review, core


def normalize(mask, size=96):
    height, width = mask.shape
    scale = min((size - 12) / width, (size - 12) / height)
    resized = cv.resize(
        mask.astype(np.uint8) * 255,
        (max(1, round(width * scale)), max(1, round(height * scale))),
        interpolation=cv.INTER_NEAREST,
    )
    canvas = np.zeros((size, size), np.uint8)
    y = (size - resized.shape[0]) // 2
    x = (size - resized.shape[1]) // 2
    canvas[y : y + resized.shape[0], x : x + resized.shape[1]] = resized
    return canvas


def similarity(left, right):
    return float(cv.matchTemplate(left, right, cv.TM_CCOEFF_NORMED)[0, 0])


def process_puzzle(path, templates):
    image = cv.imread(os.fspath(path), cv.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"could not read puzzle: {path}")
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    ink = neutral_black(image)
    annotated = image.copy()
    baselines = detect_baselines(ink)
    all_matches = []
    all_missing = []

    for line_number, baseline in enumerate(baselines, 1):
        row_top = baseline - BASELINE_OFFSET
        y0 = max(0, row_top)
        y1 = min(image.shape[0], row_top + TEMPLATE_HEIGHT)
        columns = np.flatnonzero(np.any(ink[y0:y1], axis=0))
        if not len(columns):
            continue
        content_start = int(columns[0])
        content_end = int(columns[-1]) + 1
        candidates = candidates_for_row(
            gray, templates, row_top, content_start, content_end
        )
        matches, missing = greedy_row(
            candidates, ink, row_top, content_start, content_end
        )
        all_matches.extend(matches)
        for start, end in missing:
            crops = crop_missing(image, ink, row_top, start, end)
            if crops is not None:
                review_crop, core_crop = crops
                crop_norm = normalize(core_crop)
                all_missing.append(
                    {
                        "day": natural_key(path),
                        "line": line_number,
                        "x": start,
                        "y": row_top,
                        "mask": review_crop,
                        "norm": crop_norm,
                    }
                )
                cv.rectangle(
                    annotated,
                    (start, y0),
                    (end, y1),
                    (0, 0, 255),
                    3,
                )

    for match in all_matches:
        cv.rectangle(
            annotated,
            (match.start, match.y),
            (match.end, match.y + match.template.height),
            match.template.color,
            2,
        )
        cv.putText(
            annotated,
            match.template.name,
            (match.start + 2, match.y + 15),
            cv.FONT_HERSHEY_SIMPLEX,
            0.42,
            match.template.color,
            1,
            cv.LINE_AA,
        )

    OUTPUT_DIR.mkdir(exist_ok=True)
    if not cv.imwrite(os.fspath(OUTPUT_DIR / f"{path.stem}.png"), annotated):
        raise ValueError(f"could not write output for {path}")
    print(
        f"day {natural_key(path):>2}: lines={len(baselines):>2} "
        f"matches={len(all_matches):>3} missing={len(all_missing):>2}"
    )
    return all_missing


def write_missing(candidates):
    groups = []

    for candidate in candidates:
        group = next(
            (
                item
                for item in groups
                if similarity(candidate["norm"], item["norm"]) >= 0.82
            ),
            None,
        )
        source = (
            f"day {candidate['day']} line={candidate['line']} "
            f"x={candidate['x']} y={candidate['y']}"
        )
        if group is None:
            groups.append(
                {
                    "mask": candidate["mask"],
                    "norm": candidate["norm"],
                    "sources": [source],
                }
            )
        else:
            group["sources"].append(source)

    total_groups = len(groups)
    groups.sort(key=lambda item: len(item["sources"]), reverse=True)
    groups = groups[:MAX_REVIEW_CANDIDATES]

    MISSING_DIR.mkdir(exist_ok=True)
    for path in MISSING_DIR.glob("candidate_*.png"):
        path.unlink()
    for name in ("index.csv", "all-residuals-review.png"):
        path = MISSING_DIR / name
        if path.exists():
            path.unlink()

    rows = []
    for index, group in enumerate(groups, 1):
        name = f"candidate_{index:02d}.png"
        mask = group["mask"]
        output = np.zeros((mask.shape[0] + 16, mask.shape[1] + 16, 4), np.uint8)
        output[8 : 8 + mask.shape[0], 8 : 8 + mask.shape[1]][mask] = (0, 0, 0, 255)
        cv.imwrite(os.fspath(MISSING_DIR / name), output)
        rows.append((name, len(group["sources"]), "; ".join(group["sources"])))

    with (MISSING_DIR / "index.csv").open("w", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(("file", "occurrences", "sources"))
        writer.writerows(rows)
    print(
        f"missing symbols: {len(groups)} review candidates "
        f"from {total_groups} residual shape groups"
    )


def main():
    templates = load_templates()
    puzzles = [
        path
        for path in PUZZLE_DIR.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    ]
    puzzles.sort(key=natural_key)
    missing = []
    for path in puzzles:
        missing.extend(process_puzzle(path, templates))
    write_missing(missing)


if __name__ == "__main__":
    main()
