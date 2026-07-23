import csv
import hashlib
import json
import math
import os
from collections import Counter, defaultdict
from dataclasses import dataclass, replace
from pathlib import Path
from urllib.parse import quote

import cv2 as cv
import numpy as np


PUZZLE_DIR = Path("days")
CHARACTER_DIR = Path("characters")
OUTPUT_DIR = Path("output")
MISSING_DIR = Path("missing_symbols")
ENCODED_PATH = Path("encoded_symbols.txt")
METADATA_PATH = Path("passages.csv")
REPORT_PATH = Path("sequence_report.md")
REVIEW_PATH = Path("symbol_review.html")
REVIEW_EDITS_PATH = Path("symbol_review_edits.json")
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}
TEMPLATE_HEIGHT = 163
BASELINE_OFFSET = 77
MATCH_THRESHOLD = 0.76
L_MATCH_THRESHOLD = 0.55
MATCH_THRESHOLD_BY_TEMPLATE = {
    "q": 0.74,
    "p": 0.60,
    "w": 0.74,
    "u": 0.70,
    "l": 0.68,
    "e": 0.67,
    "[": 0.70,
    "f": 0.57,
    "m": 0.72,
    "n": 0.74,
    "y": 0.81,
}
SCORE_TIE_TOLERANCE = 0.015
NORMAL_S_MATCH_THRESHOLD = 0.70
FULL_GLYPH_SCORE_MARGIN = 0.08
MIN_NEW_SYMBOL_WIDTH = 12
MIN_NEW_WIDTH_BY_TEMPLATE = {"m": 11, "!": 11}
MIN_NEW_WIDTH_AFTER = {("[", "c"): 7}
OVERLAP_PREFIX_AFTER = {("[", "m"), ("[", "c")}
CONTAINED_STROKE_TEMPLATES = {"j", "d"}
FORWARD_TOLERANCE_BY_TEMPLATE = {"t": 10}
MAX_TEMPLATE_OVERLAP = 24
LEFT_SHARED_STROKE_BY_TEMPLATE = {"l": 8, "t": 20}
LEFT_OVERLAP_TOLERANCE_BY_TEMPLATE = {"l": 12, "t": 8}
L_WHITESPACE_WIDTH = 8
MISSING_CONTEXT = 24
MAX_REVIEW_CANDIDATES = 32

PASSAGE_METADATA = {
    1: ("書面語（粵語讀音）", "SWC"),
    2: ("新詩（粵語讀音）", "New Poetry"),
    3: ("書面語（粵語讀音）", "SWC"),
    4: ("白話文（含口語字）", "Cantonese"),
    5: ("書面語（粵語讀音）", "SWC"),
    6: ("白話文（含口語字）", "Cantonese"),
    7: ("書面語（粵語讀音）", "SWC"),
    8: ("古詩詞（1900年前寫作）", "Classical"),
    9: ("書面語（粵語讀音） 更正（以紅色標示）", "SWC"),
    10: ("新詩（粵語讀音）", "New Poetry"),
    11: ("白話文（含口語字）", "Cantonese"),
    12: ("書面語（粵語讀音）建議先解拆標點符號表達方式", "SWC"),
    13: ("書面語（粵語讀音）", "SWC"),
    14: ("古詩詞（1900年前寫作）", "Classical"),
    15: ("書面語（粵語讀音）", "SWC"),
    16: ("白話文", "Cantonese"),
    17: ("書面語（粵語讀音）", "SWC"),
    18: ("白話文", "Cantonese"),
    19: ("新詩（節錄）", "New Poetry"),
    20: ("專欄文章 | 書面語（粵語讀音）", "SWC"),
    21: ("百科全書文章 | 書面語（粵語讀音）", "SWC"),
    22: ("網路文章 | 白話文", "Cantonese"),
    23: ("網絡小說 | 書面語（粵語讀音）", "SWC"),
    24: ("五言絕句 | 古詩詞（1900年前寫作）", "Classical"),
    25: ("小說 | 書面語（粵語讀音）", "SWC"),
    26: ("小說 | 白話文", "Cantonese"),
    27: ("新詩（節錄）", "New Poetry"),
    28: ("教學指南 | 書面語（粵語讀音）", "SWC"),
    29: ("網頁內容 | 白話文", "Cantonese"),
    30: ("百科全書 | 白話文", "Cantonese"),
    31: ("古詩詞（1900年前寫作）", "Classical"),
    32: ("五言絕句全首 | 古詩詞（公元1900年前寫作）", "Classical"),
}


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
    review_name: str | None = None

    @property
    def name(self):
        return self.template.name if self.review_name is None else self.review_name

    @property
    def start(self):
        return self.x if self.draw_x is None else self.draw_x

    @property
    def sequence_start(self):
        return self.x + LEFT_SHARED_STROKE_BY_TEMPLATE.get(self.template.name, 0)

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
        threshold = MATCH_THRESHOLD_BY_TEMPLATE.get(template.name, MATCH_THRESHOLD)
        if template.name == "-":
            threshold = L_MATCH_THRESHOLD
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


def follows_whitespace(candidate, column_ink, content_start):
    if candidate.template.name != "-":
        return True
    if candidate.x <= content_start + 2:
        return True

    left = max(content_start, candidate.x - 40)
    preceding = ~column_ink[left : candidate.x]
    if preceding.size < L_WHITESPACE_WIDTH:
        return False
    runs = np.convolve(
        preceding.astype(np.uint8),
        np.ones(L_WHITESPACE_WIDTH, np.uint8),
        mode="valid",
    )
    return bool(np.any(runs == L_WHITESPACE_WIDTH))


def greedy_row(candidates, ink, row_top, content_start, content_end):
    matches = []
    missing = []
    cursor = content_start
    candidates = sorted(
        candidates, key=lambda item: (item.sequence_start, -item.score)
    )
    y0 = max(0, row_top)
    y1 = min(ink.shape[0], row_top + TEMPLATE_HEIGHT)
    column_ink = np.any(ink[y0:y1], axis=0)

    def eligible(candidate):
        if not follows_whitespace(candidate, column_ink, content_start):
            return False
        previous = matches[-1].template.name if matches else None
        if (
            candidate.template.name == "f"
            and candidate.score < NORMAL_S_MATCH_THRESHOLD
        ):
            return any(
                item.template.name in {"0", "y"}
                and candidate.end - 8 <= item.sequence_start <= candidate.end + 8
                for item in candidates
            )
        if (
            candidate.template.name == "p"
            and candidate.score < MATCH_THRESHOLD
        ):
            if previous == "f":
                return any(
                    item.template.name == "a"
                    and candidate.end - 8
                    <= item.sequence_start
                    <= candidate.end + 8
                    for item in candidates
                )
            if previous == "n":
                return content_end - candidate.end <= 8
            return False
        if (
            candidate.template.name == "y"
            and candidate.score < 0.95
        ):
            if previous == "f":
                return candidate.score >= 0.93
            return previous == "n"
        return True

    def begins_near_cursor(candidate):
        overlap = LEFT_OVERLAP_TOLERANCE_BY_TEMPLATE.get(
            candidate.template.name, MAX_TEMPLATE_OVERLAP
        )
        previous = matches[-1].template.name if matches else None
        forward = 8
        if candidate.template.name == "t" and previous == "+":
            forward = FORWARD_TOLERANCE_BY_TEMPLATE["t"]
        return cursor - overlap <= candidate.sequence_start <= cursor + forward

    def residual_core_after(candidate):
        future_starts = [
            item.sequence_start
            for item in candidates
            if item.sequence_start > candidate.end + 8 and eligible(item)
        ]
        next_x = min(future_starts, default=content_end)
        if next_x - candidate.end < 14:
            return None
        y0 = max(0, row_top)
        y1 = min(ink.shape[0], row_top + TEMPLATE_HEIGHT)
        mask = ink[y0:y1, candidate.end:next_x]
        if np.count_nonzero(mask) < 60:
            return None
        ys, xs = np.where(mask)
        return mask[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1]

    def leaves_symbol_residual(candidate):
        core = residual_core_after(candidate)
        return core is not None and core.shape[0] >= 24

    def leaves_bar_residual(candidate):
        core = residual_core_after(candidate)
        return core is not None and is_bar_only_residual(core)

    def compact_dot_count(dot_x0, dot_x1, dot_y0, dot_y1, min_area=15):
        dot_x0 = max(0, dot_x0)
        dot_x1 = min(ink.shape[1], dot_x1)
        dot_y0 = max(0, dot_y0)
        dot_y1 = min(ink.shape[0], dot_y1)
        region = ink[dot_y0:dot_y1, dot_x0:dot_x1]
        count, _, stats, _ = cv.connectedComponentsWithStats(
            region.astype(np.uint8), 8
        )
        return sum(
            3 <= int(stats[label, cv.CC_STAT_WIDTH]) <= 14
            and 3 <= int(stats[label, cv.CC_STAT_HEIGHT]) <= 14
            and int(stats[label, cv.CC_STAT_AREA]) >= min_area
            for label in range(1, count)
        )

    def dot_family_variant(top, nearby):
        if top.template.name in {"j", "d", "h"}:
            family = [
                item
                for item in nearby
                if item.template.name in {"j", "d", "h"}
                and abs(item.sequence_start - top.sequence_start) <= 8
            ]
            anchors = [item for item in family if item.template.name == "j"]
            if anchors:
                anchor_x = max(anchors, key=lambda item: item.score).x
            elif top.template.name == "h":
                anchor_x = top.x + 4
            else:
                anchor_x = top.x
            dot_count = compact_dot_count(
                anchor_x + 5,
                anchor_x + 38,
                row_top + 48,
                row_top + 74,
                min_area=10,
            )
            target = {0: "j", 1: "d"}.get(dot_count, "h")
        elif top.template.name in {"g", "i", "k"}:
            family = [
                item
                for item in nearby
                if item.template.name in {"g", "i", "k"}
                and abs(item.sequence_start - top.sequence_start) <= 5
            ]
            anchors = [item for item in family if item.template.name == "k"]
            if anchors:
                anchor_x = max(anchors, key=lambda item: item.score).x
            elif top.template.name == "i":
                anchor_x = top.x + 1
            else:
                anchor_x = top.x + 5
            # Keep to the arch's inner width. The wider/deeper area belongs to
            # adjacent glyphs and was the main source of q being read as #.
            dot_count = compact_dot_count(
                anchor_x + 6,
                anchor_x + 41,
                row_top + 84,
                row_top + 104,
            )
            target = {0: "g", 1: "i"}.get(dot_count, "k")
        else:
            return None

        matching = [item for item in family if item.template.name == target]
        if matching:
            return max(matching, key=lambda item: item.score)
        if family:
            # A dotted variant can fall below template-match threshold even
            # when its shared base is an excellent match. Dot geometry is the
            # deciding evidence, so retain the base box and assign its label.
            return replace(
                max(family, key=lambda item: item.score), review_name=target
            )
        return None

    while cursor < content_end:
        previous = matches[-1].template.name if matches else None
        nearby = [
            candidate
            for candidate in candidates
            if begins_near_cursor(candidate)
            and candidate.end - cursor
            >= MIN_NEW_WIDTH_AFTER.get(
                (previous, candidate.template.name),
                MIN_NEW_WIDTH_BY_TEMPLATE.get(
                    candidate.template.name, MIN_NEW_SYMBOL_WIDTH
                ),
            )
            and eligible(candidate)
        ]
        nearby.extend(
            candidate
            for candidate in candidates
            if candidate.template.name == "d"
            and cursor + 8 < candidate.sequence_start <= cursor + 12
            and candidate.end - cursor >= MIN_NEW_SYMBOL_WIDTH
            and eligible(candidate)
            and leaves_bar_residual(candidate)
        )
        nearby.extend(
            candidate
            for candidate in candidates
            if candidate.template.name == "g"
            and candidate.score >= 0.93
            and cursor + 8 < candidate.sequence_start <= cursor + 12
            and any(
                item.template.name == "p"
                and item.score >= MATCH_THRESHOLD
                and item.sequence_start <= cursor + 8
                and item.x <= candidate.x <= item.end
                and item.end - candidate.x > MAX_TEMPLATE_OVERLAP
                for item in candidates
            )
            and max(
                (
                    item.score
                    for item in candidates
                    if item.template.name == "-"
                    and begins_near_cursor(item)
                    and item.end - cursor >= MIN_NEW_SYMBOL_WIDTH
                    and eligible(item)
                ),
                default=-1,
            )
            <= max(
                (
                    item.score
                    for item in candidates
                    if item.template.name == "p"
                    and item.score >= MATCH_THRESHOLD
                    and item.sequence_start <= cursor + 8
                    and item.x <= candidate.x <= item.end
                    and item.end - candidate.x > MAX_TEMPLATE_OVERLAP
                ),
                default=-1,
            ) + SCORE_TIE_TOLERANCE
            and max(
                (
                    item.score
                    for item in candidates
                    if item.template.name == "k"
                    and candidate.x + 3 <= item.x <= candidate.x + 7
                ),
                default=-1,
            )
            > max(
                (
                    item.score
                    for item in candidates
                    if item.template.name == "i"
                    and candidate.x + 2 <= item.x <= candidate.x + 7
                ),
                default=-1,
            )
        )
        if nearby:
            # Exact full symbols score higher than contained sub-shapes. Prefer
            # the wider candidate only when confidence is effectively tied.
            best_score = max(candidate.score for candidate in nearby)
            top = max(nearby, key=lambda item: item.score)
            complete = []
            if top.template.name in CONTAINED_STROKE_TEMPLATES:
                complete = [
                    candidate
                    for candidate in nearby
                    if candidate.template.name not in CONTAINED_STROKE_TEMPLATES
                    and candidate.template.width > top.template.width
                    and abs(candidate.sequence_start - top.sequence_start) <= 12
                    and candidate.score >= best_score - FULL_GLYPH_SCORE_MARGIN
                ]
            dotted_variant = dot_family_variant(top, nearby)
            complete_without_residual = [
                candidate
                for candidate in complete
                if not leaves_symbol_residual(candidate)
            ]
            aligned_at_line_start = [
                candidate
                for candidate in nearby
                if not matches
                and abs(candidate.sequence_start - content_start) <= 8
                and candidate.score >= best_score - 0.02
            ]
            overlap_prefix = [
                candidate
                for candidate in nearby
                if (previous, candidate.template.name) in OVERLAP_PREFIX_AFTER
                and candidate.sequence_start < cursor
                and candidate.end <= top.sequence_start + 8
            ]
            if overlap_prefix:
                best = max(overlap_prefix, key=lambda item: item.score)
            elif (
                aligned_at_line_start
                and top.sequence_start < content_start - 8
            ):
                best = max(aligned_at_line_start, key=lambda item: item.score)
            elif dotted_variant is not None:
                best = dotted_variant
            elif leaves_symbol_residual(top) and complete_without_residual:
                best = max(
                    complete_without_residual,
                    key=lambda item: (item.template.width, item.score),
                )
            else:
                tied = [
                    candidate
                    for candidate in nearby
                    if candidate.score >= best_score - SCORE_TIE_TOLERANCE
                ]
                bar_z = [
                    candidate
                    for candidate in tied
                    if candidate.template.name == "d"
                    and leaves_bar_residual(candidate)
                ]
                if top.template.name in CONTAINED_STROKE_TEMPLATES and bar_z:
                    best = max(bar_z, key=lambda item: item.score)
                else:
                    best = max(
                        tied, key=lambda item: (item.template.width, item.score)
                    )
            matches.append(replace(best, draw_x=max(cursor, best.sequence_start)))
            cursor = max(cursor + 1, best.end)
            continue

        future = [
            candidate
            for candidate in candidates
            if candidate.sequence_start > cursor + 8 and eligible(candidate)
        ]
        next_x = min(
            (candidate.sequence_start for candidate in future), default=content_end
        )
        if next_x - cursor >= 14 and ink_area(ink, row_top, cursor, next_x) >= 60:
            missing.append((cursor, next_x))
        cursor = max(cursor + 1, next_x)

    return matches, missing


def encode_row(matches, missing, content_start, column_ink):
    items = [
        (match.start, match.end, match.name) for match in matches
    ] + [(start, end, "?") for start, end in missing]
    items.sort(key=lambda item: (item[0], item[1]))

    encoded = []
    cursor = content_start
    for start, end, symbol in items:
        if start - cursor >= L_WHITESPACE_WIDTH:
            blank = ~column_ink[cursor:start]
            runs = np.convolve(
                blank.astype(np.uint8),
                np.ones(L_WHITESPACE_WIDTH, np.uint8),
                mode="valid",
            )
            if np.any(runs == L_WHITESPACE_WIDTH):
                encoded.append(" ")
        encoded.append(symbol)
        cursor = max(cursor, end)
    return "".join(encoded)


def is_bar_only_residual(mask):
    count, _, stats, _ = cv.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    components = sorted(
        (
            (
                int(stats[label, cv.CC_STAT_WIDTH]),
                int(stats[label, cv.CC_STAT_HEIGHT]),
                int(stats[label, cv.CC_STAT_AREA]),
            )
            for label in range(1, count)
        ),
        key=lambda item: item[2],
        reverse=True,
    )
    if not components:
        return False
    width, height, area = components[0]
    other_area = sum(component[2] for component in components[1:])
    return (
        width >= 14
        and height <= 10
        and area >= width * 5
        and other_area <= area * 0.50
    )


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
    if core.shape[0] < 24 or is_bar_only_residual(core):
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


def load_review_edits(templates):
    if not REVIEW_EDITS_PATH.exists():
        return {}
    rows = json.loads(REVIEW_EDITS_PATH.read_text(encoding="utf-8"))
    template_names = {template.name for template in templates}
    edits = {}
    for row in rows:
        target = row["to"]
        if target not in template_names | {"__missing__", "__segmentation__"}:
            raise ValueError(f"unknown review target: {target}")
        key = (int(row["day"]), int(row["line"]), int(row["position"]))
        if key in edits:
            raise ValueError(f"duplicate review edit: {key}")
        edits[key] = row
    print(f"review edits loaded: {len(edits)} from {REVIEW_EDITS_PATH}")
    return edits


def process_puzzle(path, templates, review_edits, edit_stats):
    image = cv.imread(os.fspath(path), cv.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"could not read puzzle: {path}")
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    ink = neutral_black(image)
    annotated = image.copy()
    baselines = detect_baselines(ink)
    all_matches = []
    all_missing = []
    occurrences = []
    encoded_lines = []

    line_number = 0
    for baseline in baselines:
        row_top = baseline - BASELINE_OFFSET
        y0 = max(0, row_top)
        y1 = min(image.shape[0], row_top + TEMPLATE_HEIGHT)
        column_ink = np.any(ink[y0:y1], axis=0)
        columns = np.flatnonzero(column_ink)
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
        if not matches:
            continue
        line_number += 1
        reviewed_matches = []
        reviewed_missing = []
        day = natural_key(path)
        for position, match in enumerate(matches, 1):
            edit = review_edits.get((day, line_number, position))
            if edit is None:
                reviewed_matches.append(match)
                continue
            if edit["from"] != match.name:
                edit_stats["skipped"] += 1
                reviewed_matches.append(match)
                continue
            edit_stats["applied"] += 1
            if edit["to"] == "__segmentation__":
                continue
            if edit["to"] == "__missing__":
                reviewed_missing.append((match.start, match.end))
                continue
            reviewed_matches.append(replace(match, review_name=edit["to"]))
        matches = reviewed_matches
        missing.extend(reviewed_missing)
        all_matches.extend(matches)
        line_symbols = [match.name for match in matches]
        for index, match in enumerate(matches):
            occurrences.append(
                {
                    "day": natural_key(path),
                    "line": line_number,
                    "symbol": match.name,
                    "x": match.x,
                    "y": match.y,
                    "width": match.template.width,
                    "height": match.template.height,
                    "score": round(match.score, 4),
                    "position": index + 1,
                    "context": "".join(
                        line_symbols[max(0, index - 3) : index + 4]
                    ),
                }
            )
        valid_missing = []
        for start, end in missing:
            crops = crop_missing(image, ink, row_top, start, end)
            if crops is not None:
                valid_missing.append((start, end))
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
        encoded_lines.append(
            encode_row(matches, valid_missing, content_start, column_ink)
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
            match.name,
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
        f"day {natural_key(path):>2}: lines={len(encoded_lines):>2} "
        f"matches={len(all_matches):>3} missing={len(all_missing):>2}"
    )
    return all_missing, "/".join(encoded_lines), occurrences


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


def reduce_encoded_sequences(encoded):
    return (
        encoded.replace("rf", "r")
        .replace("gf", "f")
        .replace("gx", "x")
    )


def write_encoded(days):
    with ENCODED_PATH.open("w") as handle:
        handle.write("# day<TAB>encoded string\n")
        handle.write("# / = source line break; ? = unmatched region\n")
        for day, encoded in days:
            encoded = reduce_encoded_sequences(encoded)
            handle.write(f"{day}\t{encoded}\n")
    print(f"encoded strings: {ENCODED_PATH}")


def write_metadata():
    with METADATA_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(("day", "description", "lang"))
        for day in sorted(PASSAGE_METADATA):
            description, lang = PASSAGE_METADATA[day]
            writer.writerow((day, description, lang))
    print(f"passage metadata: {METADATA_PATH}")


def ngram_counts(encoded):
    counts = {length: Counter() for length in range(1, 7)}
    symbols = "".join(encoded.replace("/", "").split())
    for length in range(1, 7):
        counts[length].update(
            symbols[index : index + length]
            for index in range(len(symbols) - length + 1)
        )
    return counts


def write_sequence_report(days):
    encoded_by_day = dict(days)
    grams_by_day = {
        day: ngram_counts(encoded) for day, encoded in encoded_by_day.items()
    }
    genre_order = ("SWC", "Cantonese", "New Poetry", "Classical")
    genre_days = defaultdict(list)
    for day, encoded in encoded_by_day.items():
        if encoded:
            genre_days[PASSAGE_METADATA[day][1]].append(day)

    lines = [
        "# Distinctive symbol-sequence report",
        "",
        "This report removes encoded whitespace and `/` line markers, then "
        "counts every contiguous sequence from one through six symbols within "
        "each passage. Sequences may cross source-line boundaries but never day "
        "boundaries. Day 16 is retained in the metadata but excluded from "
        "sequence statistics because it has no cipher text.",
        "",
        "Genre distinctiveness uses passage prevalence rather than raw length: "
        "a sequence must occur in at least two passages of the target genre and "
        "be at least 10 percentage points more prevalent there. `log2 lift` is "
        "a smoothed target-versus-other-genre prevalence ratio.",
        "",
        "## Corpus",
        "",
        "| lang | passages | passages with symbols |",
        "| --- | ---: | ---: |",
    ]
    for genre in genre_order:
        total = sum(
            PASSAGE_METADATA[day][1] == genre for day in PASSAGE_METADATA
        )
        lines.append(f"| {genre} | {total} | {len(genre_days[genre])} |")

    symbol_inventory = sorted(
        set().union(
            *(grams_by_day[day][1] for day in encoded_by_day)
        )
    )
    symbol_frequencies = Counter()
    for day in encoded_by_day:
        symbol_frequencies.update(grams_by_day[day][1])
    total_symbols = sum(symbol_frequencies.values())
    frequency_rows = []
    for symbol, occurrences in symbol_frequencies.most_common():
        day_count = sum(
            symbol in grams_by_day[day][1] for day in encoded_by_day
        )
        frequency_rows.append((symbol, occurrences, day_count))

    lines.extend(
        [
            "",
            "## Symbol frequency inventory",
            "",
            "Frequencies exclude whitespace and line separators. Images are "
            "the current reference templates in `characters/`.",
            "",
            "| rank | image | code | occurrences | corpus share | days |",
            "| ---: | --- | --- | ---: | ---: | ---: |",
        ]
    )
    for rank, (symbol, occurrences, day_count) in enumerate(frequency_rows, 1):
        template_path = CHARACTER_DIR / f"{symbol}.png"
        if template_path.exists():
            image_path = f"characters/{quote(symbol + '.png')}"
            image_cell = f'<img src="{image_path}" alt="{symbol}" height="52">'
        else:
            image_cell = "manual correction (no template)"
        share = 100 * occurrences / total_symbols
        lines.append(
            f"| {rank} | {image_cell} "
            f"| `{symbol}` | {occurrences} | {share:.2f}% | {day_count} |"
        )

    lines.extend(
        [
            "",
            "## N-gram token frequencies by category",
            "",
            "These complete inventories count contiguous sequences after `/` "
            "line markers are removed, so n-grams may cross source-line "
            "boundaries. Corpus cells show "
            "`count (share of all corpus n-gram tokens)`. Category cells show "
            "`count (share of that category's n-gram tokens; log₂ lift)`. The "
            "lift compares the category's token rate with the combined rate in "
            "the other categories and uses add-0.5 rate smoothing; positive "
            "values indicate category enrichment.",
            "",
            "| n | corpus tokens | distinct types | SWC tokens | Cantonese tokens | New Poetry tokens | Classical tokens |",
            "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    ngram_frequency_data = {}
    for length in range(2, 6):
        corpus_counts = Counter()
        category_counts = {genre: Counter() for genre in genre_order}
        for day in encoded_by_day:
            corpus_counts.update(grams_by_day[day][length])
            genre = PASSAGE_METADATA[day][1]
            category_counts[genre].update(grams_by_day[day][length])
        corpus_tokens = sum(corpus_counts.values())
        category_tokens = {
            genre: sum(category_counts[genre].values())
            for genre in genre_order
        }
        ngram_frequency_data[length] = (
            corpus_counts,
            corpus_tokens,
            category_counts,
            category_tokens,
        )
        lines.append(
            f"| {length} | {corpus_tokens} | {len(corpus_counts)} | "
            + " | ".join(str(category_tokens[genre]) for genre in genre_order)
            + " |"
        )

    for length in range(2, 6):
        (
            corpus_counts,
            corpus_tokens,
            category_counts,
            category_tokens,
        ) = ngram_frequency_data[length]
        lines.extend(
            [
                "",
                f"### {length}-grams",
                "",
                "| rank | sequence | corpus | SWC | Cantonese | New Poetry | Classical |",
                "| ---: | --- | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        ranked = sorted(
            corpus_counts.items(), key=lambda item: (-item[1], item[0])
        )
        for rank, (sequence, corpus_count) in enumerate(ranked, 1):
            corpus_pct = 100 * corpus_count / corpus_tokens
            category_cells = []
            for genre in genre_order:
                category_count = category_counts[genre][sequence]
                target_tokens = category_tokens[genre]
                other_count = corpus_count - category_count
                other_tokens = corpus_tokens - target_tokens
                target_rate = (category_count + 0.5) / (
                    target_tokens + 1
                )
                other_rate = (other_count + 0.5) / (
                    other_tokens + 1
                )
                lift = math.log2(target_rate / other_rate)
                category_pct = 100 * category_count / target_tokens
                category_cells.append(
                    f"{category_count} ({category_pct:.3f}%; {lift:+.2f})"
                )
            lines.append(
                f"| {rank} | `{sequence}` | "
                f"{corpus_count} ({corpus_pct:.3f}%) | "
                + " | ".join(category_cells)
                + " |"
            )

    repeating = []
    non_repeating = []
    for symbol in symbol_inventory:
        pair = symbol * 2
        occurrences = sum(
            grams_by_day[day][2][pair] for day in encoded_by_day
        )
        days_with_pair = [
            day
            for day in sorted(encoded_by_day)
            if grams_by_day[day][2][pair]
        ]
        if occurrences:
            repeating.append((occurrences, symbol, days_with_pair))
        else:
            non_repeating.append(symbol)
    repeating.sort(key=lambda item: (-item[0], item[1]))

    lines.extend(
        [
            "",
            "## Immediate symbol repetition",
            "",
            "A symbol is listed as repeating when its doubled sequence occurs "
            "within a passage after `/` line markers are removed. ‘Not observed "
            "doubled’ is corpus evidence, not proof that the writing system "
            "forbids the repetition.",
            "",
            "### Observed doubled",
            "",
            "| symbol | sequence | occurrences | days |",
            "| --- | --- | ---: | --- |",
        ]
    )
    for occurrences, symbol, days_with_pair in repeating:
        day_list = ", ".join(str(day) for day in days_with_pair)
        lines.append(
            f"| `{symbol}` | `{symbol * 2}` | {occurrences} | {day_list} |"
        )
    lines.extend(
        [
            "",
            "### Not observed doubled",
            "",
            ", ".join(f"`{symbol}`" for symbol in non_repeating) or "-",
        ]
    )

    active_genres = [genre for genre, genre_list in genre_days.items() if genre_list]
    lines.extend(["", "## Genre-distinctive sequences", ""])
    if len(active_genres) < 2:
        lines.append("At least two non-empty genres are required for comparison.")
    else:
        for genre in genre_order:
            target_days = genre_days[genre]
            if not target_days:
                lines.extend(
                    [
                        f"### {genre}",
                        "",
                        "No passages carry this label in the current PDF, so no "
                        "genre profile can be estimated.",
                        "",
                    ]
                )
                continue
            other_days = [
                day
                for other_genre in active_genres
                if other_genre != genre
                for day in genre_days[other_genre]
            ]
            lines.extend(
                [
                    f"### {genre}",
                    "",
                    "| n | sequence | target passages | other passages | log2 lift | occurrences |",
                    "| ---: | --- | ---: | ---: | ---: | ---: |",
                ]
            )
            any_rows = False
            for length in range(1, 7):
                vocabulary = set().union(
                    *(grams_by_day[day][length] for day in target_days)
                )
                ranked = []
                for sequence in vocabulary:
                    target_support = sum(
                        sequence in grams_by_day[day][length]
                        for day in target_days
                    )
                    other_support = sum(
                        sequence in grams_by_day[day][length]
                        for day in other_days
                    )
                    target_rate = target_support / len(target_days)
                    other_rate = other_support / len(other_days)
                    prevalence_gap = target_rate - other_rate
                    if target_support < 2 or prevalence_gap < 0.10:
                        continue
                    smoothed_target = (target_support + 0.5) / (
                        len(target_days) + 1
                    )
                    smoothed_other = (other_support + 0.5) / (
                        len(other_days) + 1
                    )
                    lift = math.log2(smoothed_target / smoothed_other)
                    if lift <= 0:
                        continue
                    occurrences = sum(
                        grams_by_day[day][length][sequence]
                        for day in target_days
                    )
                    score = prevalence_gap * max(lift, 0) * math.log2(
                        occurrences + 1
                    )
                    ranked.append(
                        (
                            score,
                            prevalence_gap,
                            occurrences,
                            sequence,
                            target_support,
                            other_support,
                            lift,
                        )
                    )
                ranked.sort(reverse=True)
                for (
                    _,
                    _,
                    occurrences,
                    sequence,
                    target_support,
                    other_support,
                    lift,
                ) in ranked[:6]:
                    any_rows = True
                    target_pct = 100 * target_support / len(target_days)
                    other_pct = 100 * other_support / len(other_days)
                    lines.append(
                        f"| {length} | `{sequence}` | "
                        f"{target_support}/{len(target_days)} ({target_pct:.0f}%) | "
                        f"{other_support}/{len(other_days)} ({other_pct:.0f}%) | "
                        f"{lift:.2f} | {occurrences} |"
                    )
            if not any_rows:
                lines.append("| - | - | - | - | - | - |")
            lines.append("")

    document_frequency = {
        length: Counter(
            sequence
            for day in encoded_by_day
            for sequence in grams_by_day[day][length]
        )
        for length in range(1, 7)
    }
    lines.extend(
        [
            "## Passage-unique signatures",
            "",
            "A signature appears in exactly one passage in the corpus. The list "
            "prefers the shortest unique sequences and suppresses longer entries "
            "that merely contain an already-listed shorter signature.",
            "",
            "| day | lang | key unique sequences (occurrences in that passage) |",
            "| ---: | --- | --- |",
        ]
    )
    for day in sorted(encoded_by_day):
        selected = []
        for length in range(1, 7):
            unique = [
                (count, sequence)
                for sequence, count in grams_by_day[day][length].items()
                if document_frequency[length][sequence] == 1
            ]
            unique.sort(key=lambda item: (-item[0], item[1]))
            for count, sequence in unique:
                if any(existing in sequence for existing, _ in selected):
                    continue
                selected.append((sequence, count))
                if len(selected) == 8:
                    break
            if len(selected) == 8:
                break
        signatures = ", ".join(
            f"`{sequence}` ({count})" for sequence, count in selected
        )
        if not signatures:
            signatures = "-"
        lines.append(
            f"| {day} | {PASSAGE_METADATA[day][1]} | {signatures} |"
        )

    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"sequence report: {REPORT_PATH}")


def write_symbol_review(occurrences):
    payload = json.dumps(
        {
            "metadata": [
                [day, *PASSAGE_METADATA[day]] for day in sorted(PASSAGE_METADATA)
            ],
            "occurrences": [
                [
                    item["day"],
                    item["line"],
                    item["symbol"],
                    item["x"],
                    item["y"],
                    item["width"],
                    item["height"],
                    item["score"],
                    item["position"],
                    item["context"],
                ]
                for item in occurrences
            ],
        },
        ensure_ascii=False,
        separators=(",", ":"),
    )
    template = r'''<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Canto Cryptic symbol review</title>
<style>
:root{color-scheme:light;--ink:#17221d;--muted:#68736d;--paper:#f4f1e9;--card:#fffdf8;--line:#d9d4c8;--accent:#176b55;--accent-soft:#dcece5;--warn:#a94c2b}
*{box-sizing:border-box}body{margin:0;background:var(--paper);color:var(--ink);font:14px/1.45 ui-monospace,SFMono-Regular,Menlo,Consolas,monospace}
header{position:sticky;top:0;z-index:4;padding:18px clamp(18px,4vw,54px) 14px;background:rgba(244,241,233,.96);border-bottom:1px solid var(--line);backdrop-filter:blur(10px)}
.topline{display:flex;align-items:baseline;justify-content:space-between;gap:18px;flex-wrap:wrap}h1{margin:0;font:700 clamp(22px,3vw,34px)/1.1 ui-serif,Georgia,serif}.links{display:flex;gap:14px}.links a{color:var(--accent)}
.summary{margin:7px 0 0;color:var(--muted)}main{padding:22px clamp(18px,4vw,54px) 42px}.controls{display:grid;grid-template-columns:minmax(150px,1fr) repeat(3,minmax(130px,.55fr)) auto auto;gap:10px;align-items:end}
label{display:grid;gap:5px;color:var(--muted);font-size:12px}select,input,button{font:inherit;color:inherit;background:var(--card);border:1px solid var(--line);border-radius:8px;padding:9px 10px}button{cursor:pointer}button:hover,button:focus-visible{border-color:var(--accent)}
.symbols{display:flex;gap:7px;overflow:auto;padding:14px 0 8px;scrollbar-width:thin}.symbol{display:flex;gap:7px;align-items:center;min-width:max-content;padding:7px 10px}.symbol strong{font-size:18px}.symbol small{color:var(--muted)}.symbol.active{color:#fff;background:var(--accent);border-color:var(--accent)}.symbol.active small{color:#dcece5}
.resultbar{display:flex;align-items:center;justify-content:space-between;gap:12px;margin:12px 0}.pager{display:flex;gap:7px;align-items:center}.pager button:disabled{opacity:.4;cursor:default}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(250px,1fr));gap:12px}.card{position:relative;min-width:0;background:var(--card);border:1px solid var(--line);border-radius:12px;overflow:hidden;cursor:pointer}.card:hover,.card:focus-visible{border-color:var(--accent);outline:none}.card.edited{box-shadow:inset 0 0 0 2px var(--accent)}.comparison{display:grid;grid-template-columns:minmax(0,1fr) 92px;background:#fff}.crop{display:block;width:100%;height:130px;background:#fff}.reference{display:grid;grid-template-rows:auto 1fr;place-items:center;padding:7px;border-left:1px solid var(--line);color:var(--muted);font-size:10px;text-transform:uppercase;letter-spacing:.06em}.reference img{display:block;max-width:68px;max-height:104px;object-fit:contain}.decision{position:absolute;top:8px;left:8px;max-width:calc(100% - 16px);padding:3px 7px;border-radius:999px;background:var(--accent);color:#fff;font-size:11px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}.meta{padding:10px 12px 12px}.identity{display:flex;align-items:baseline;justify-content:space-between;gap:10px}.identity strong{font-size:22px}.score{font-size:12px;color:var(--muted)}.score.weak{color:var(--warn);font-weight:700}.where,.context,.description{white-space:nowrap;overflow:hidden;text-overflow:ellipsis}.where{margin-top:4px}.context{font-size:16px;color:var(--accent)}.description{margin-top:4px;color:var(--muted);font-size:12px}.empty{padding:40px;border:1px dashed var(--line);border-radius:12px;text-align:center;color:var(--muted)}
dialog{width:min(760px,calc(100vw - 28px));max-height:88vh;padding:0;border:1px solid var(--line);border-radius:14px;background:var(--card);color:var(--ink);box-shadow:0 24px 80px #17221d44}dialog::backdrop{background:#17221d99}.dialog-head{display:flex;justify-content:space-between;gap:16px;align-items:start;padding:18px 20px;border-bottom:1px solid var(--line)}.dialog-head h2{margin:0;font:700 22px/1.2 ui-serif,Georgia,serif}.dialog-head p{margin:5px 0 0;color:var(--muted)}.dialog-body{padding:18px 20px;overflow:auto}.special-actions{display:flex;flex-wrap:wrap;gap:8px;margin-bottom:16px}.special-actions .missing{color:#fff;background:var(--warn);border-color:var(--warn)}.choice-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(78px,1fr));gap:8px}.choice{display:grid;place-items:center;gap:5px;min-height:112px;padding:8px}.choice img{max-width:58px;max-height:72px;object-fit:contain}.choice strong{font-size:17px}.dialog-close{font-size:20px;padding:3px 9px}
@media(max-width:820px){.controls{grid-template-columns:1fr 1fr}.controls label:first-child{grid-column:1/-1}}@media(max-width:480px){.controls{grid-template-columns:1fr}.controls label:first-child{grid-column:auto}.grid{grid-template-columns:1fr}}
</style>
</head>
<body>
<header><div class="topline"><h1>Symbol classification review</h1><nav class="links"><a href="passages.csv">passages.csv</a><a href="sequence_report.md">sequence report</a></nav></div><p class="summary" id="summary"></p></header>
<main>
<section class="controls" aria-label="Review filters">
<label>Search symbol or context<input id="search" type="search" placeholder="e.g. #, k, ptij"></label>
<label>Day<select id="day"><option value="">All days</option></select></label>
<label>Language<select id="lang"><option value="">All languages</option><option>SWC</option><option>Cantonese</option><option>New Poetry</option><option>Classical</option></select></label>
<label>Order<select id="sort"><option value="score">Weakest score first</option><option value="source">Source order</option><option value="symbol">Symbol then source</option></select></label>
<button id="reset" type="button">Reset</button>
<button id="exportEdits" type="button">Export edits (0)</button>
</section>
<div class="symbols" id="symbols" aria-label="Filter by classified symbol"></div>
<div class="resultbar"><span id="resultCount"></span><div class="pager"><button id="prev" type="button">Previous</button><span id="page"></span><button id="next" type="button">Next</button></div></div>
<section class="grid" id="grid" aria-live="polite"></section>
</main>
<dialog id="editor"><div class="dialog-head"><div><h2 id="editorTitle">Edit classification</h2><p id="editorMeta"></p></div><button class="dialog-close" id="closeEditor" type="button" aria-label="Close">×</button></div><div class="dialog-body"><div class="special-actions"><button class="missing" id="markMissing" type="button">Missing symbol</button><button id="markSegmentation" type="button">Segmentation / no symbol</button><button id="clearDecision" type="button">Clear saved edit</button></div><div class="choice-grid" id="choices" aria-label="Replacement symbols"></div></div></dialog>
<script>
const raw=__DATA__;
const metadata=new Map(raw.metadata.map(([day,description,lang])=>[day,{description,lang}]));
const occurrences=raw.occurrences.map(([day,line,symbol,x,y,width,height,score,position,context])=>({day,line,symbol,x,y,width,height,score,position,context}));
const state={symbol:"",page:1,pageSize:120};
const imageCache=new Map();
const storageKey="canto-cryptic-review-edits-v3";
const $=id=>document.getElementById(id);
let activeItem=null;
let edits={};
try{edits=JSON.parse(localStorage.getItem(storageKey)||"{}")||{}}catch{edits={}}
const itemKey=item=>`${item.day}:${item.line}:${item.position}`;
const templateUrl=symbol=>`characters/${encodeURIComponent(symbol)}.png`;
const decisionLabel=value=>value==="__missing__"?"missing symbol":value==="__segmentation__"?"segmentation / no symbol":`replace with ${value}`;
const symbolCounts=new Map();for(const item of occurrences)symbolCounts.set(item.symbol,(symbolCounts.get(item.symbol)||0)+1);
const symbolOrder=[...symbolCounts.keys()].sort((a,b)=>a.localeCompare(b,"en"));
$("summary").textContent=`${occurrences.length.toLocaleString()} classified occurrences across ${metadata.size} days. Each source crop is shown beside the expected template; click a mismatch to record an edit.`;
for(const day of [...metadata.keys()].sort((a,b)=>a-b)){const option=document.createElement("option");option.value=day;option.textContent=`Day ${day}`;$("day").append(option)}
function makeSymbolButton(symbol,label,count){const button=document.createElement("button");button.type="button";button.className="symbol";button.dataset.symbol=symbol;button.innerHTML=`<strong></strong><small></small>`;button.querySelector("strong").textContent=label;button.querySelector("small").textContent=count.toLocaleString();button.addEventListener("click",()=>{state.symbol=symbol;state.page=1;render()});return button}
$("symbols").append(makeSymbolButton("","All",occurrences.length));for(const symbol of symbolOrder)$("symbols").append(makeSymbolButton(symbol,symbol,symbolCounts.get(symbol)));
for(const symbol of symbolOrder){const button=document.createElement("button");button.type="button";button.className="choice";const image=document.createElement("img");image.src=templateUrl(symbol);image.alt=`Template ${symbol}`;const label=document.createElement("strong");label.textContent=symbol;button.append(image,label);button.addEventListener("click",()=>saveDecision(symbol));$("choices").append(button)}
function updateEditCount(){const count=Object.keys(edits).length;$("exportEdits").textContent=`Export edits (${count})`;$("exportEdits").disabled=!count}
function saveDecision(value){if(!activeItem)return;edits[itemKey(activeItem)]={day:activeItem.day,line:activeItem.line,position:activeItem.position,context:activeItem.context,from:activeItem.symbol,to:value};localStorage.setItem(storageKey,JSON.stringify(edits));$("editor").close();updateEditCount();render()}
function openEditor(item){activeItem=item;const saved=edits[itemKey(item)];$("editorTitle").textContent=`Day ${item.day}, line ${item.line}, position ${item.position}`;$("editorMeta").textContent=`Current classification: ${item.symbol} · context ${item.context}${saved?` · saved: ${decisionLabel(saved.to)}`:""}`;$("clearDecision").disabled=!saved;$("editor").showModal()}
function filtered(){const query=$("search").value.trim();const day=Number($("day").value)||0;const lang=$("lang").value;const result=occurrences.filter(item=>(!state.symbol||item.symbol===state.symbol)&&(!query||item.symbol.includes(query)||item.context.includes(query))&&(!day||item.day===day)&&(!lang||metadata.get(item.day).lang===lang));const order=$("sort").value;if(order==="score")result.sort((a,b)=>a.score-b.score||a.day-b.day||a.line-b.line||a.position-b.position);else if(order==="symbol")result.sort((a,b)=>a.symbol.localeCompare(b.symbol,"en")||a.day-b.day||a.line-b.line||a.position-b.position);else result.sort((a,b)=>a.day-b.day||a.line-b.line||a.position-b.position);return result}
function loadDay(day){if(!imageCache.has(day)){const image=new Image();image.src=`days/${day}.png`;imageCache.set(day,image)}return imageCache.get(day)}
function drawCrop(canvas,item){const image=loadDay(item.day);const paint=()=>{const contextX=26,contextY=14;const sx=Math.max(0,item.x-contextX),sy=Math.max(0,item.y-contextY);const sw=Math.min(image.naturalWidth-sx,item.width+contextX*2),sh=Math.min(image.naturalHeight-sy,item.height+contextY*2);const scale=Math.min(canvas.width/sw,canvas.height/sh);const dw=sw*scale,dh=sh*scale,dx=(canvas.width-dw)/2,dy=(canvas.height-dh)/2;const ctx=canvas.getContext("2d");ctx.fillStyle="#fff";ctx.fillRect(0,0,canvas.width,canvas.height);ctx.drawImage(image,sx,sy,sw,sh,dx,dy,dw,dh);ctx.strokeStyle="#176b55";ctx.lineWidth=2;ctx.strokeRect(dx+(item.x-sx)*scale,dy+(item.y-sy)*scale,item.width*scale,item.height*scale)};if(image.complete&&image.naturalWidth)paint();else image.addEventListener("load",paint,{once:true})}
function cardFor(item){const card=document.createElement("article");const saved=edits[itemKey(item)];card.className=`card${saved?" edited":""}`;card.tabIndex=0;card.setAttribute("role","button");card.setAttribute("aria-label",`Edit day ${item.day}, line ${item.line}, symbol ${item.symbol}`);const comparison=document.createElement("div");comparison.className="comparison";const canvas=document.createElement("canvas");canvas.className="crop";canvas.width=360;canvas.height=260;canvas.setAttribute("aria-label",`Source crop for ${item.symbol}`);const reference=document.createElement("div");reference.className="reference";const referenceLabel=document.createElement("span");referenceLabel.textContent="correct form";const referenceImage=document.createElement("img");referenceImage.src=templateUrl(item.symbol);referenceImage.alt=`Expected template ${item.symbol}`;reference.append(referenceLabel,referenceImage);comparison.append(canvas,reference);card.append(comparison);if(saved){const badge=document.createElement("span");badge.className="decision";badge.textContent=decisionLabel(saved.to);card.append(badge)}const meta=document.createElement("div");meta.className="meta";const weak=item.score<.8?" weak":"";meta.innerHTML=`<div class="identity"><strong></strong><span class="score${weak}"></span></div><div class="where"></div><div class="context"></div><div class="description"></div>`;meta.querySelector("strong").textContent=item.symbol;meta.querySelector(".score").textContent=`score ${item.score.toFixed(3)}`;meta.querySelector(".where").textContent=`day ${item.day} · line ${item.line} · position ${item.position}`;meta.querySelector(".context").textContent=item.context;const info=metadata.get(item.day);meta.querySelector(".description").textContent=`${info.lang} · ${info.description}`;card.append(meta);drawCrop(canvas,item);card.addEventListener("click",()=>openEditor(item));card.addEventListener("keydown",event=>{if(event.key==="Enter"||event.key===" "){event.preventDefault();openEditor(item)}});return card}
function render(){for(const button of $("symbols").children)button.classList.toggle("active",button.dataset.symbol===state.symbol);const items=filtered(),pages=Math.max(1,Math.ceil(items.length/state.pageSize));state.page=Math.min(state.page,pages);const start=(state.page-1)*state.pageSize,pageItems=items.slice(start,start+state.pageSize);$("resultCount").textContent=`${items.length.toLocaleString()} occurrence${items.length===1?"":"s"}`;$("page").textContent=`${state.page} / ${pages}`;$("prev").disabled=state.page===1;$("next").disabled=state.page===pages;$("grid").replaceChildren(...(pageItems.length?pageItems.map(cardFor):[Object.assign(document.createElement("div"),{className:"empty",textContent:"No occurrences match these filters."})]))}
for(const id of ["search","day","lang","sort"]){$(id).addEventListener(id==="search"?"input":"change",()=>{state.page=1;render()})}
$("prev").addEventListener("click",()=>{state.page--;render();scrollTo({top:0,behavior:"smooth"})});
$("next").addEventListener("click",()=>{state.page++;render();scrollTo({top:0,behavior:"smooth"})});
$("reset").addEventListener("click",()=>{$("search").value="";$("day").value="";$("lang").value="";$("sort").value="score";state.symbol="";state.page=1;render()});
$("closeEditor").addEventListener("click",()=>$("editor").close());
$("markMissing").addEventListener("click",()=>saveDecision("__missing__"));
$("markSegmentation").addEventListener("click",()=>saveDecision("__segmentation__"));
$("clearDecision").addEventListener("click",()=>{if(!activeItem)return;delete edits[itemKey(activeItem)];localStorage.setItem(storageKey,JSON.stringify(edits));$("editor").close();updateEditCount();render()});
$("exportEdits").addEventListener("click",()=>{const blob=new Blob([JSON.stringify(Object.values(edits),null,2)+"\n"],{type:"application/json"});const link=document.createElement("a");link.href=URL.createObjectURL(blob);link.download="symbol_review_edits.json";link.click();setTimeout(()=>URL.revokeObjectURL(link.href),1000)});
updateEditCount();render();
</script>
</body>
</html>
'''
    REVIEW_PATH.write_text(template.replace("__DATA__", payload), encoding="utf-8")
    print(f"symbol review: {REVIEW_PATH}")


def main():
    templates = load_templates()
    review_edits = load_review_edits(templates)
    edit_stats = {"applied": 0, "skipped": 0}
    puzzles = [
        path
        for path in PUZZLE_DIR.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    ]
    puzzles.sort(key=natural_key)
    missing = []
    encoded_days = []
    occurrences = []
    for path in puzzles:
        day_missing, encoded, day_occurrences = process_puzzle(
            path, templates, review_edits, edit_stats
        )
        missing.extend(day_missing)
        occurrences.extend(day_occurrences)
        encoded_days.append(
            (natural_key(path), reduce_encoded_sequences(encoded))
        )
    write_missing(missing)
    write_encoded(encoded_days)
    write_metadata()
    write_sequence_report(encoded_days)
    write_symbol_review(occurrences)
    if review_edits:
        print(
            f"review edits: applied={edit_stats['applied']} "
            f"skipped_after_rematch={edit_stats['skipped']}"
        )


if __name__ == "__main__":
    main()
