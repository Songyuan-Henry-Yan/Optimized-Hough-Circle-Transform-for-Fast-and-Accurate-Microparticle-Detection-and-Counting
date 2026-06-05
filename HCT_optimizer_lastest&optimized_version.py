"""
HCT_optimizer_improved_with_explanations.py

Improved Hough Circle Transform (HCT) parameter optimizer for bead counting.

This script is intended for the training step:
    manually counted training images
        -> optimize HoughCircles parameters
        -> save ranked parameter sets and annotated preview images

Main improvements compared with the older optimizer:
    1. Correct parameter order:
       [DP, Minimum Distance, Median Blur, Param 1, Param 2, Minimum Radius, Maximum Radius]

    2. Duplicate / overlap removal:
       The optimizer can remove duplicate circles using the same general idea as the GUI preview.
       This helps make the training count more consistent with the visual GUI count.

    3. Better scoring metric:
       The old score was based mainly on the average "found/manual" ratio.
       This version uses absolute percent error for each image:
           percent_error = abs(found_count - manual_count) / manual_count
       Then it ranks parameter sets by:
           Metric Score = Mean Percent Error + SD Percent Error
       Lower is better.

    4. Faster image handling:
       Each worker loads the training images once and precomputes the grayscale/blurred images.
       This avoids repeatedly calling cv.imread(), cv.cvtColor(), and cv.medianBlur()
       inside every parameter combination.

    5. Cleaner multiprocessing:
       The script uses one analysis function with multiprocessing.Pool instead of eight nearly
       identical analysis_1(), analysis_2(), ... functions.

    6. Range-edge warnings:
       If the best parameter is at the minimum or maximum edge of your test range, the script
       prints a warning. That usually means you should expand that range and rerun.

    7. Easy best-result files:
       The script creates BEST_PARAMETER_SET_FOR_COPY_PASTE.txt and BEST_PARAMETER_SET.csv,
       so you can directly open the best parameters without searching the large CSV.

How to use:
    1. Put this .py file in the same folder as your training bead images.
    2. Edit the USER SETTINGS section below.
    3. Replace every manual count in test_images_dict with your real counted bead number.
    4. Adjust the Hough parameter ranges using your GUI estimates.
    5. Run:
           python HCT_optimizer_improved_with_explanations.py

Important:
    Use raw training images only.
    Do not use GUI screenshot images that already have circles drawn on them.
"""

import csv
import datetime as dt
import itertools
import math
import multiprocessing as mp
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

try:
    import cv2 as cv
except ImportError as exc:
    raise SystemExit(
        "OpenCV is not installed in this Python environment.\n"
        "Install it in the PyCharm terminal with:\n"
        "    pip install opencv-python numpy"
    ) from exc

try:
    import numpy as np
except ImportError as exc:
    raise SystemExit(
        "NumPy is not installed in this Python environment.\n"
        "Install it in the PyCharm terminal with:\n"
        "    pip install numpy"
    ) from exc


# =============================================================================
# USER SETTINGS
# =============================================================================

# -----------------------------------------------------------------------------
# 1) Training images and manual counts
# -----------------------------------------------------------------------------
# Replace the 0 values below with your manual bead counts.
#
# The filename must match exactly:
#     - same spelling
#     - same capitalization
#     - same extension, for example .jpg vs .png
#
# Example:
# test_images_dict = {
#     "cropImage1.jpg": 128,
#     "cropImage2.jpg": 134,
#     ...
# }
#
# This script will stop with a clear error message if any count is still 0.
test_images_dict = {
    "cropImage1.jpg": 391,
    "cropImage2.jpg": 470,
    "cropImage3.jpg": 564,
    "cropImage4.jpg": 681,
    "cropImage5.jpg": 760,
    "cropImage6.jpg": 841,
    "cropImage7.jpg": 915,
    "cropImage8.jpg": 979,
    "cropImage9.jpg": 1053,
    "cropImage10.jpg": 1094,
}

# -----------------------------------------------------------------------------
# 2) Image and output folders
# -----------------------------------------------------------------------------
# IMAGE_FOLDER:
#     None means "use the same folder as this .py script".
#     If your images are in a subfolder, use something like:
#         IMAGE_FOLDER = "training_images"
IMAGE_FOLDER = None

# OUTPUT_FOLDER_NAME:
#     Results will be written into this folder.
#     A timestamp is added automatically so older runs are not overwritten.
OUTPUT_FOLDER_NAME = "optimizer_output_improved"

# -----------------------------------------------------------------------------
# 3) Hough parameter ranges
# -----------------------------------------------------------------------------
# These ranges should be guided by your GUI preview.
#
# For Python range(start, stop, step), the stop value is NOT included.
# Example:
#     [*range(1, 6, 2)] gives [1, 3, 5]
#
# Median blur must be odd. A value of 1 means "no real blur" in this script.
median_blur_list = [*range(1, 8, 2)]        # [1, 3, 5]

# Param 1 is the upper Canny edge detector threshold used by HoughCircles.
# If the best value is at the edge, expand this range.
param_1_list = [*range(8, 18, 1)]           # example: 8, 12, 16, 20, 24, 28

# Param 2 is the center detection threshold.
# Lower values usually detect more circles; higher values are stricter.
param_2_list = [*range(10, 18, 1)]           # example: 8, 10, ..., 24

# Radius ranges are in pixels.
# For 3 um beads, use the GUI preview to estimate the radius in pixels.
min_radius_list = [*range(3, 5, 1)]       # example: 10, 11, 12, 13, 14
max_radius_list = [*range(6, 8, 1)]       # example: 13, 14, 15, 16, 17, 18

# Minimum distance between circle centers, in pixels.
# This is often close to the bead diameter or slightly smaller/larger,
# depending on crowding and overlap.
min_dist_list = [*range(1, 4, 1)]         # example: 30, 34, 38, 42, 46

# DP is the inverse accumulator resolution ratio in cv.HoughCircles.
# dp=1 is usually a good starting point and is kept fixed here.
dp_default = 1.0

# -----------------------------------------------------------------------------
# 4) Accuracy options
# -----------------------------------------------------------------------------
# If True, the script removes likely duplicate detections after HoughCircles.
# This is useful when the same bead is detected more than once.
REMOVE_DUPLICATE_CIRCLES = True

# Duplicate rule:
#     if distance_between_centers < DUPLICATE_DISTANCE_FACTOR * min(radius_a, radius_b),
#     then the later circle is treated as a duplicate.
#
# 1.0 is a reasonable default.
# Smaller values are less aggressive.
# Larger values are more aggressive.
DUPLICATE_DISTANCE_FACTOR = 1.0

# Optional region of interest.
# Leave as None to analyze the whole image.
#
# If you want to analyze only part of the image, set:
#     ROI = (x1, y1, x2, y2)
# where (x1, y1) is the top-left corner and (x2, y2) is the bottom-right corner.
#
# Example:
#     ROI = (100, 50, 900, 700)
ROI = None

# -----------------------------------------------------------------------------
# 5) Efficiency and output options
# -----------------------------------------------------------------------------
# Number of CPU workers.
# On Windows/PyCharm, 8 is usually okay if the computer has enough cores.
# To be safer or reduce heat/fan load, set N_WORKERS = 4.
N_WORKERS = min(8, os.cpu_count() or 1)

# How many parameter sets each worker receives at a time.
# Larger chunk size can reduce multiprocessing overhead.
CHUNK_SIZE = 25

# Print progress every N completed parameter sets.
PROGRESS_EVERY = 500

# Save the top N ranked parameter sets to top_n_params.csv and save preview images for them.
TOP_N_TO_SAVE = 10

# Save images with circles drawn for the top parameter sets.
SAVE_ANNOTATED_IMAGES = True

# If True, also save a small CSV with per-image counts for the best parameter set.
SAVE_BEST_COUNTS_CSV = True


# =============================================================================
# INTERNAL GLOBALS USED BY MULTIPROCESSING WORKERS
# =============================================================================
# These globals are initialized separately inside each worker process.
G_TEST_IMAGES_DICT: Dict[str, int] = {}
G_BLURRED_IMAGE_CACHE: Dict[Tuple[str, int], np.ndarray] = {}
G_REMOVE_DUPLICATES: bool = True
G_DUPLICATE_DISTANCE_FACTOR: float = 1.0


# =============================================================================
# PARAMETER AND IMAGE UTILITY FUNCTIONS
# =============================================================================

def get_script_folder() -> Path:
    """Return the folder containing this script."""
    try:
        return Path(__file__).resolve().parent
    except NameError:
        return Path.cwd()


def resolve_image_folder() -> Path:
    """Resolve IMAGE_FOLDER into an absolute Path."""
    script_folder = get_script_folder()

    if IMAGE_FOLDER is None:
        return script_folder

    image_folder = Path(IMAGE_FOLDER)
    if not image_folder.is_absolute():
        image_folder = script_folder / image_folder

    return image_folder.resolve()


def make_output_folder() -> Path:
    """
    Create a timestamped output folder.

    Timestamped folders prevent results from different runs from mixing together.
    """
    script_folder = get_script_folder()
    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_folder = script_folder / f"{OUTPUT_FOLDER_NAME}_{timestamp}"
    output_folder.mkdir(parents=True, exist_ok=False)
    return output_folder


def validate_user_settings(image_folder: Path) -> None:
    """
    Check for common setup mistakes before the long optimizer starts.
    """
    if not test_images_dict:
        raise ValueError("test_images_dict is empty. Add your training image filenames and manual counts.")

    zero_or_negative_counts = [
        name for name, count in test_images_dict.items()
        if not isinstance(count, int) or count <= 0
    ]
    if zero_or_negative_counts:
        message = (
            "The following images have manual counts that are 0, negative, or not integers:\n"
            + "\n".join(f"    {name}: {test_images_dict[name]}" for name in zero_or_negative_counts)
            + "\n\nEdit test_images_dict near the top of the script and replace each 0 with your manual bead count."
        )
        raise ValueError(message)

    missing_files = [
        name for name in test_images_dict
        if not (image_folder / name).is_file()
    ]
    if missing_files:
        message = (
            f"Could not find the following training image files in:\n    {image_folder}\n\n"
            + "\n".join(f"    {name}" for name in missing_files)
            + "\n\nCheck the filenames, extensions, and folder location."
        )
        raise FileNotFoundError(message)

    if not median_blur_list:
        raise ValueError("median_blur_list is empty.")

    for blur in median_blur_list:
        if not isinstance(blur, int) or blur < 1 or blur % 2 == 0:
            raise ValueError(
                f"Invalid median blur value: {blur}. "
                "Median blur values must be positive odd integers, for example 1, 3, 5."
            )

    positive_integer_lists = {
        "param_1_list": param_1_list,
        "param_2_list": param_2_list,
        "min_radius_list": min_radius_list,
        "max_radius_list": max_radius_list,
        "min_dist_list": min_dist_list,
    }

    for list_name, values in positive_integer_lists.items():
        if not values:
            raise ValueError(f"{list_name} is empty.")
        for value in values:
            if not isinstance(value, int) or value <= 0:
                raise ValueError(f"{list_name} contains invalid value {value}. Values must be positive integers.")

    if dp_default <= 0:
        raise ValueError("dp_default must be positive.")

    if ROI is not None:
        if (
            not isinstance(ROI, tuple)
            or len(ROI) != 4
            or not all(isinstance(v, int) for v in ROI)
        ):
            raise ValueError("ROI must be None or a tuple of four integers: (x1, y1, x2, y2).")

        x1, y1, x2, y2 = ROI
        if x1 < 0 or y1 < 0 or x2 <= x1 or y2 <= y1:
            raise ValueError("Invalid ROI. It should be (x1, y1, x2, y2) with x2 > x1 and y2 > y1.")

    if N_WORKERS < 1:
        raise ValueError("N_WORKERS must be at least 1.")


def build_parameter_grid() -> List[Tuple[float, int, int, int, int, int, int]]:
    """
    Build all valid parameter combinations.

    Parameter order is intentionally:
        DP, Minimum Distance, Median Blur, Param 1, Param 2, Minimum Radius, Maximum Radius

    This order matches the output CSV columns and the analysis code.
    """
    parameter_grid: List[Tuple[float, int, int, int, int, int, int]] = []

    for median_blur, param_1, param_2, min_radius, max_radius, min_dist in itertools.product(
        median_blur_list,
        param_1_list,
        param_2_list,
        min_radius_list,
        max_radius_list,
        min_dist_list,
    ):
        # Invalid Hough setup: minimum radius should not be greater than maximum radius.
        if min_radius > max_radius:
            continue

        parameter_grid.append(
            (
                float(dp_default),
                int(min_dist),
                int(median_blur),
                int(param_1),
                int(param_2),
                int(min_radius),
                int(max_radius),
            )
        )

    if not parameter_grid:
        raise ValueError(
            "No valid parameter sets were built. "
            "Check that min_radius_list values are not all greater than max_radius_list values."
        )

    return parameter_grid


def crop_if_needed(image: np.ndarray) -> np.ndarray:
    """Apply ROI crop if ROI is set; otherwise return the full image."""
    if ROI is None:
        return image

    x1, y1, x2, y2 = ROI

    height, width = image.shape[:2]
    if x2 > width or y2 > height:
        raise ValueError(
            f"ROI {ROI} is outside the image dimensions width={width}, height={height}."
        )

    return image[y1:y2, x1:x2]


def apply_median_blur(gray: np.ndarray, blur_size: int) -> np.ndarray:
    """
    Apply median blur safely.

    OpenCV medianBlur normally expects an odd kernel size.
    We treat blur_size = 1 as "no blur" because that is useful for testing.
    """
    if blur_size <= 1:
        return gray.copy()

    return cv.medianBlur(gray, blur_size)


def build_blurred_image_cache(
    image_folder: Path,
    image_names: Sequence[str],
    blur_values: Sequence[int],
) -> Dict[Tuple[str, int], np.ndarray]:
    """
    Read each image once and build grayscale/blurred versions for each blur value.

    This is faster than reading and blurring inside every parameter-set test.
    """
    cache: Dict[Tuple[str, int], np.ndarray] = {}

    for image_name in image_names:
        image_path = image_folder / image_name
        color_image = cv.imread(str(image_path), cv.IMREAD_COLOR)

        if color_image is None:
            raise FileNotFoundError(f"OpenCV could not read image: {image_path}")

        color_image = crop_if_needed(color_image)
        gray = cv.cvtColor(color_image, cv.COLOR_BGR2GRAY)

        for blur in blur_values:
            cache[(image_name, int(blur))] = apply_median_blur(gray, int(blur))

    return cache


# =============================================================================
# CIRCLE DETECTION FUNCTIONS
# =============================================================================

def remove_duplicate_circles(
    circles: Optional[np.ndarray],
    duplicate_distance_factor: float = 1.0,
) -> Optional[np.ndarray]:
    """
    Remove likely duplicate circle detections.

    HoughCircles may detect the same bead more than once with slightly different centers/radii.
    This function keeps the first circle and removes later circles whose center is too close
    to an already-kept circle.

    Duplicate criterion:
        distance_between_centers < duplicate_distance_factor * min(radius_a, radius_b)

    Notes:
        - This is a practical duplicate-removal heuristic.
        - It cannot perfectly decide whether two very close detections are duplicates or truly
          overlapping beads.
        - Always inspect annotated output images for the top parameter set.
    """
    if circles is None:
        return None

    if len(circles) == 0 or circles.shape[1] == 0:
        return None

    circle_array = np.asarray(circles[0], dtype=float)
    kept_circles: List[np.ndarray] = []

    for circle in circle_array:
        x, y, radius = float(circle[0]), float(circle[1]), float(circle[2])
        is_duplicate = False

        for kept in kept_circles:
            kept_x, kept_y, kept_radius = float(kept[0]), float(kept[1]), float(kept[2])
            distance = math.hypot(x - kept_x, y - kept_y)
            threshold = duplicate_distance_factor * min(radius, kept_radius)

            if distance < threshold:
                is_duplicate = True
                break

        if not is_duplicate:
            kept_circles.append(circle)

    if not kept_circles:
        return None

    return np.asarray([kept_circles], dtype=float)


def run_hough_circles(
    gray_image: np.ndarray,
    params: Tuple[float, int, int, int, int, int, int],
    remove_duplicates: bool,
    duplicate_distance_factor: float,
) -> Optional[np.ndarray]:
    """
    Run HoughCircles with one parameter set and optionally remove duplicates.
    """
    dp, min_dist, median_blur, param_1, param_2, min_radius, max_radius = params

    # median_blur is already applied in the cached gray_image.
    circles = cv.HoughCircles(
        image=gray_image,
        method=cv.HOUGH_GRADIENT,
        dp=float(dp),
        minDist=int(min_dist),
        param1=int(param_1),
        param2=int(param_2),
        minRadius=int(min_radius),
        maxRadius=int(max_radius),
    )

    if remove_duplicates:
        circles = remove_duplicate_circles(circles, duplicate_distance_factor)

    return circles


def count_circles(circles: Optional[np.ndarray]) -> int:
    """Return the number of circles, or 0 if no circles were found."""
    if circles is None:
        return 0
    if len(circles) == 0:
        return 0
    return int(circles.shape[1])


# =============================================================================
# PARAMETER SCORING
# =============================================================================

def worker_initializer(
    image_folder_string: str,
    manual_counts: Dict[str, int],
    blur_values: Sequence[int],
    remove_duplicates: bool,
    duplicate_distance_factor: float,
) -> None:
    """
    Initialize global image cache inside each worker process.

    On Windows, each worker is a separate Python process. Therefore each worker needs its
    own cache of preprocessed images.
    """
    global G_TEST_IMAGES_DICT
    global G_BLURRED_IMAGE_CACHE
    global G_REMOVE_DUPLICATES
    global G_DUPLICATE_DISTANCE_FACTOR

    image_folder = Path(image_folder_string)

    G_TEST_IMAGES_DICT = dict(manual_counts)
    G_BLURRED_IMAGE_CACHE = build_blurred_image_cache(
        image_folder=image_folder,
        image_names=list(manual_counts.keys()),
        blur_values=blur_values,
    )
    G_REMOVE_DUPLICATES = bool(remove_duplicates)
    G_DUPLICATE_DISTANCE_FACTOR = float(duplicate_distance_factor)


def analyze_parameter_set(
    params: Tuple[float, int, int, int, int, int, int]
) -> Dict[str, object]:
    """
    Test one Hough parameter set on all training images.

    The returned dictionary becomes one row in all_params_ranked.csv.
    """
    dp, min_dist, median_blur, param_1, param_2, min_radius, max_radius = params

    row: Dict[str, object] = {
        "DP": dp,
        "Minimum Distance": min_dist,
        "Median Blur": median_blur,
        "Param 1": param_1,
        "Param 2": param_2,
        "Minimum Radius": min_radius,
        "Maximum Radius": max_radius,
    }

    ratios: List[float] = []
    absolute_errors: List[float] = []
    percent_errors: List[float] = []

    for image_index, (image_name, manual_count) in enumerate(G_TEST_IMAGES_DICT.items(), start=1):
        gray_image = G_BLURRED_IMAGE_CACHE[(image_name, median_blur)]

        circles = run_hough_circles(
            gray_image=gray_image,
            params=params,
            remove_duplicates=G_REMOVE_DUPLICATES,
            duplicate_distance_factor=G_DUPLICATE_DISTANCE_FACTOR,
        )

        found_count = count_circles(circles)
        ratio = found_count / manual_count
        absolute_error = abs(found_count - manual_count)
        percent_error = absolute_error / manual_count

        ratios.append(ratio)
        absolute_errors.append(float(absolute_error))
        percent_errors.append(float(percent_error))

        row[f"Image {image_index} Name"] = image_name
        row[f"Image {image_index} Manual Count"] = manual_count
        row[f"Image {image_index} Beads Found"] = found_count
        row[f"Image {image_index} Found/Manual"] = ratio
        row[f"Image {image_index} Absolute Error"] = absolute_error
        row[f"Image {image_index} Percent Error"] = percent_error

    mean_ratio = float(np.mean(ratios))
    mean_absolute_error = float(np.mean(absolute_errors))
    mean_percent_error = float(np.mean(percent_errors))
    sd_percent_error = float(np.std(percent_errors))
    max_percent_error = float(np.max(percent_errors))

    # Main optimizer score.
    # Lower is better.
    #
    # Mean Percent Error rewards accurate average counts.
    # SD Percent Error penalizes parameter sets that work for some images but fail on others.
    metric_score = mean_percent_error + sd_percent_error

    row["Mean Found/Manual"] = mean_ratio
    row["Mean Absolute Error"] = mean_absolute_error
    row["Mean Percent Error"] = mean_percent_error
    row["SD Percent Error"] = sd_percent_error
    row["Max Percent Error"] = max_percent_error
    row["Metric Score"] = metric_score

    return row


def build_csv_columns(image_names: Sequence[str]) -> List[str]:
    """Create consistent CSV column order."""
    columns = [
        "DP",
        "Minimum Distance",
        "Median Blur",
        "Param 1",
        "Param 2",
        "Minimum Radius",
        "Maximum Radius",
        "Mean Found/Manual",
        "Mean Absolute Error",
        "Mean Percent Error",
        "SD Percent Error",
        "Max Percent Error",
        "Metric Score",
    ]

    for image_index, _ in enumerate(image_names, start=1):
        columns.extend(
            [
                f"Image {image_index} Name",
                f"Image {image_index} Manual Count",
                f"Image {image_index} Beads Found",
                f"Image {image_index} Found/Manual",
                f"Image {image_index} Absolute Error",
                f"Image {image_index} Percent Error",
            ]
        )

    return columns


def write_csv(path: Path, rows: Sequence[Dict[str, object]], columns: Sequence[str]) -> None:
    """Write rows to a CSV file."""
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(columns), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


# =============================================================================
# OUTPUT AND REVIEW FUNCTIONS
# =============================================================================

def get_params_from_row(row: Dict[str, object]) -> Tuple[float, int, int, int, int, int, int]:
    """Extract parameter tuple from a CSV/output row."""
    return (
        float(row["DP"]),
        int(row["Minimum Distance"]),
        int(row["Median Blur"]),
        int(row["Param 1"]),
        int(row["Param 2"]),
        int(row["Minimum Radius"]),
        int(row["Maximum Radius"]),
    )


def sanitize_filename_part(text: str) -> str:
    """Make a safe filename component."""
    allowed = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-"
    return "".join(ch if ch in allowed else "_" for ch in text)


def save_annotated_images_for_top_rows(
    top_rows: Sequence[Dict[str, object]],
    image_folder: Path,
    output_folder: Path,
    parent_blurred_cache: Dict[Tuple[str, int], np.ndarray],
) -> None:
    """
    Save images with detected circles drawn on top.

    These images are important because the numerical best parameter set should still be
    visually checked. A low error score is helpful, but it does not replace inspection.
    """
    annotated_folder = output_folder / "annotated_top_parameter_sets"
    annotated_folder.mkdir(parents=True, exist_ok=True)

    for param_index, row in enumerate(top_rows, start=1):
        params = get_params_from_row(row)
        dp, min_dist, median_blur, param_1, param_2, min_radius, max_radius = params

        param_folder = annotated_folder / f"top_param_set_{param_index:03d}"
        param_folder.mkdir(parents=True, exist_ok=True)

        for image_name, manual_count in test_images_dict.items():
            image_path = image_folder / image_name
            color_image = cv.imread(str(image_path), cv.IMREAD_COLOR)

            if color_image is None:
                raise FileNotFoundError(f"OpenCV could not read image: {image_path}")

            color_image = crop_if_needed(color_image)
            gray_image = parent_blurred_cache[(image_name, int(median_blur))]

            circles = run_hough_circles(
                gray_image=gray_image,
                params=params,
                remove_duplicates=REMOVE_DUPLICATE_CIRCLES,
                duplicate_distance_factor=DUPLICATE_DISTANCE_FACTOR,
            )

            found_count = count_circles(circles)

            if circles is not None:
                rounded_circles = np.uint16(np.around(circles))
                for circle in rounded_circles[0, :]:
                    center = (int(circle[0]), int(circle[1]))
                    radius = int(circle[2])

                    # Small dot at center.
                    cv.circle(color_image, center, 1, (0, 100, 100), 2)

                    # Circle outline.
                    cv.circle(color_image, center, radius, (0, 0, 255), 2)

            # Add a small label to the image.
            label = f"found={found_count}, manual={manual_count}, error={abs(found_count - manual_count)}"
            cv.putText(
                color_image,
                label,
                (10, 30),
                cv.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2,
                cv.LINE_AA,
            )

            safe_stem = sanitize_filename_part(Path(image_name).stem)
            output_name = (
                f"top_{param_index:03d}_{safe_stem}_"
                f"found_{found_count}_manual_{manual_count}.png"
            )
            cv.imwrite(str(param_folder / output_name), color_image)


def save_best_counts_csv(
    best_row: Dict[str, object],
    output_folder: Path,
) -> None:
    """
    Save a simple per-image count/error table for the best parameter set.
    """
    rows = []
    for image_index, image_name in enumerate(test_images_dict.keys(), start=1):
        manual = best_row[f"Image {image_index} Manual Count"]
        found = best_row[f"Image {image_index} Beads Found"]
        abs_error = best_row[f"Image {image_index} Absolute Error"]
        pct_error = best_row[f"Image {image_index} Percent Error"]

        rows.append(
            {
                "Image": image_name,
                "Manual Count": manual,
                "Detected Count": found,
                "Absolute Error": abs_error,
                "Percent Error": pct_error,
            }
        )

    write_csv(
        output_folder / "best_parameter_set_counts.csv",
        rows,
        ["Image", "Manual Count", "Detected Count", "Absolute Error", "Percent Error"],
    )


def save_best_parameter_files(
    best_row: Dict[str, object],
    tied_best_rows: Sequence[Dict[str, object]],
    output_folder: Path,
    columns: Sequence[str],
) -> None:
    """
    Save easy-to-find files containing only the best parameter result.

    This is intentionally redundant with top_scoring_params.csv, because during real
    experiments it is easy to lose the best row inside a large output folder.

    Files created:
        1. BEST_PARAMETER_SET.csv
           One-row CSV containing the single best parameter set.

        2. BEST_PARAMETER_SET_TIES.csv
           If multiple parameter sets have the same best score, this file stores all ties.
           If there is only one best row, it will contain the same single row.

        3. BEST_PARAMETER_SET_FOR_COPY_PASTE.txt
           Human-readable text with the exact values to copy into the final 50-frame
           analysis script.

        4. BEST_PARAMETER_SET_AS_PYTHON_DICT.txt
           Python dictionary format, useful if you want to paste the parameters into
           another Python program.
    """
    # 1) Single best row only.
    write_csv(output_folder / "BEST_PARAMETER_SET.csv", [best_row], columns)

    # 2) All exact ties for the best score.
    write_csv(output_folder / "BEST_PARAMETER_SET_TIES.csv", tied_best_rows, columns)

    best_params = {
        "dp": float(best_row["DP"]),
        "minDist": int(best_row["Minimum Distance"]),
        "medianBlur": int(best_row["Median Blur"]),
        "param1": int(best_row["Param 1"]),
        "param2": int(best_row["Param 2"]),
        "minRadius": int(best_row["Minimum Radius"]),
        "maxRadius": int(best_row["Maximum Radius"]),
    }

    copy_paste_path = output_folder / "BEST_PARAMETER_SET_FOR_COPY_PASTE.txt"
    with copy_paste_path.open("w", encoding="utf-8") as f:
        f.write("BEST Hough Circle Transform parameter set\n")
        f.write("=" * 60 + "\n\n")
        f.write("Use these values for the final 50-frame analysis.\n")
        f.write("Do not change these values when applying them to the 50 validation frames.\n\n")

        f.write("Copy/paste values:\n")
        f.write("-" * 60 + "\n")
        f.write(f"dp = {best_params['dp']}\n")
        f.write(f"minDist = {best_params['minDist']}\n")
        f.write(f"medianBlur = {best_params['medianBlur']}\n")
        f.write(f"param1 = {best_params['param1']}\n")
        f.write(f"param2 = {best_params['param2']}\n")
        f.write(f"minRadius = {best_params['minRadius']}\n")
        f.write(f"maxRadius = {best_params['maxRadius']}\n\n")

        f.write("Training performance of this parameter set:\n")
        f.write("-" * 60 + "\n")
        f.write(f"Mean Found/Manual = {best_row['Mean Found/Manual']}\n")
        f.write(f"Mean Absolute Error = {best_row['Mean Absolute Error']} beads/image\n")
        f.write(f"Mean Percent Error = {100 * float(best_row['Mean Percent Error']):.4f}%\n")
        f.write(f"SD Percent Error = {100 * float(best_row['SD Percent Error']):.4f}%\n")
        f.write(f"Max Percent Error = {100 * float(best_row['Max Percent Error']):.4f}%\n")
        f.write(f"Metric Score = {float(best_row['Metric Score']):.8f}\n\n")

        f.write("Notes:\n")
        f.write("-" * 60 + "\n")
        f.write("BEST_PARAMETER_SET.csv contains the same best result as a one-row CSV.\n")
        f.write("BEST_PARAMETER_SET_TIES.csv contains all parameter sets tied for the best score.\n")
        f.write("best_parameter_set_counts.csv contains the per-image detected counts and errors.\n")
        f.write("Always inspect the annotated images before using the parameter set for final analysis.\n")

    python_dict_path = output_folder / "BEST_PARAMETER_SET_AS_PYTHON_DICT.txt"
    with python_dict_path.open("w", encoding="utf-8") as f:
        f.write("# Best Hough parameters from optimizer\n")
        f.write("# Paste this dictionary into your 50-frame analysis code if useful.\n\n")
        f.write("best_hough_params = {\n")
        f.write(f"    'dp': {best_params['dp']},\n")
        f.write(f"    'minDist': {best_params['minDist']},\n")
        f.write(f"    'medianBlur': {best_params['medianBlur']},\n")
        f.write(f"    'param1': {best_params['param1']},\n")
        f.write(f"    'param2': {best_params['param2']},\n")
        f.write(f"    'minRadius': {best_params['minRadius']},\n")
        f.write(f"    'maxRadius': {best_params['maxRadius']},\n")
        f.write("}\n")


def find_range_edge_warnings(best_row: Dict[str, object]) -> List[str]:
    """
    Warn if a best parameter value is at the min or max tested value.

    If the optimum is at the edge, the true optimum may be outside your tested range.
    """
    warnings: List[str] = []

    checks = [
        ("Median Blur", int(best_row["Median Blur"]), median_blur_list),
        ("Param 1", int(best_row["Param 1"]), param_1_list),
        ("Param 2", int(best_row["Param 2"]), param_2_list),
        ("Minimum Radius", int(best_row["Minimum Radius"]), min_radius_list),
        ("Maximum Radius", int(best_row["Maximum Radius"]), max_radius_list),
        ("Minimum Distance", int(best_row["Minimum Distance"]), min_dist_list),
    ]

    for name, best_value, tested_values in checks:
        min_tested = min(tested_values)
        max_tested = max(tested_values)

        if best_value == min_tested:
            warnings.append(
                f"{name} best value {best_value} is at the LOWER edge of the tested range. "
                f"Consider testing lower values."
            )
        elif best_value == max_tested:
            warnings.append(
                f"{name} best value {best_value} is at the UPPER edge of the tested range. "
                f"Consider testing higher values."
            )

    return warnings


def write_run_summary(
    output_folder: Path,
    image_folder: Path,
    parameter_count: int,
    best_row: Dict[str, object],
    edge_warnings: Sequence[str],
    elapsed_seconds: float,
) -> None:
    """
    Write a human-readable summary of the run.
    """
    summary_path = output_folder / "run_summary.txt"

    with summary_path.open("w", encoding="utf-8") as f:
        f.write("Hough Circle Transform optimizer run summary\n")
        f.write("=" * 60 + "\n\n")

        f.write(f"Run time: {dt.datetime.now()}\n")
        f.write(f"Image folder: {image_folder}\n")
        f.write(f"Output folder: {output_folder}\n")
        f.write(f"Training images: {len(test_images_dict)}\n")
        f.write(f"Parameter sets tested: {parameter_count}\n")
        f.write(f"Workers used: {N_WORKERS}\n")
        f.write(f"Elapsed seconds: {elapsed_seconds:.2f}\n\n")

        f.write("Accuracy settings\n")
        f.write("-" * 60 + "\n")
        f.write(f"Remove duplicate circles: {REMOVE_DUPLICATE_CIRCLES}\n")
        f.write(f"Duplicate distance factor: {DUPLICATE_DISTANCE_FACTOR}\n")
        f.write(f"ROI: {ROI}\n\n")

        f.write("Parameter ranges tested\n")
        f.write("-" * 60 + "\n")
        f.write(f"median_blur_list = {median_blur_list}\n")
        f.write(f"param_1_list = {param_1_list}\n")
        f.write(f"param_2_list = {param_2_list}\n")
        f.write(f"min_radius_list = {min_radius_list}\n")
        f.write(f"max_radius_list = {max_radius_list}\n")
        f.write(f"min_dist_list = {min_dist_list}\n")
        f.write(f"dp_default = {dp_default}\n\n")

        f.write("Best parameter set\n")
        f.write("-" * 60 + "\n")
        for key in [
            "DP",
            "Minimum Distance",
            "Median Blur",
            "Param 1",
            "Param 2",
            "Minimum Radius",
            "Maximum Radius",
            "Mean Found/Manual",
            "Mean Absolute Error",
            "Mean Percent Error",
            "SD Percent Error",
            "Max Percent Error",
            "Metric Score",
        ]:
            f.write(f"{key}: {best_row[key]}\n")

        f.write("\nRange-edge warnings\n")
        f.write("-" * 60 + "\n")
        if edge_warnings:
            for warning in edge_warnings:
                f.write(f"- {warning}\n")
        else:
            f.write("No best parameters were at the edge of the tested ranges.\n")


def print_best_result(best_row: Dict[str, object], edge_warnings: Sequence[str]) -> None:
    """Print best result to console."""
    print("\nBest parameter set:")
    print(f"  DP: {best_row['DP']}")
    print(f"  Minimum Distance: {best_row['Minimum Distance']}")
    print(f"  Median Blur: {best_row['Median Blur']}")
    print(f"  Param 1: {best_row['Param 1']}")
    print(f"  Param 2: {best_row['Param 2']}")
    print(f"  Minimum Radius: {best_row['Minimum Radius']}")
    print(f"  Maximum Radius: {best_row['Maximum Radius']}")
    print(f"  Mean Percent Error: {100 * float(best_row['Mean Percent Error']):.2f}%")
    print(f"  SD Percent Error: {100 * float(best_row['SD Percent Error']):.2f}%")
    print(f"  Metric Score: {float(best_row['Metric Score']):.6f}")

    if edge_warnings:
        print("\nRange-edge warnings:")
        for warning in edge_warnings:
            print(f"  - {warning}")


# =============================================================================
# MAIN PROGRAM
# =============================================================================

def run_optimizer() -> None:
    start_time = time.time()

    image_folder = resolve_image_folder()
    validate_user_settings(image_folder)

    output_folder = make_output_folder()
    image_names = list(test_images_dict.keys())

    print("Hough Circle Transform improved optimizer started.")
    print(f"Image folder: {image_folder}")
    print(f"Output folder: {output_folder}")
    print(f"Training images: {len(image_names)}")

    parameter_grid = build_parameter_grid()
    total_parameter_sets = len(parameter_grid)
    print(f"Total parameter sets to be tested: {total_parameter_sets}")
    print(f"Time: {dt.datetime.now()}")

    print("\nPreloading images in parent process for validation and output image generation...")
    parent_blurred_cache = build_blurred_image_cache(
        image_folder=image_folder,
        image_names=image_names,
        blur_values=median_blur_list,
    )
    print("Parent image cache ready.")

    results: List[Dict[str, object]] = []

    if N_WORKERS == 1:
        print("\nRunning in single-worker mode.")
        worker_initializer(
            image_folder_string=str(image_folder),
            manual_counts=test_images_dict,
            blur_values=median_blur_list,
            remove_duplicates=REMOVE_DUPLICATE_CIRCLES,
            duplicate_distance_factor=DUPLICATE_DISTANCE_FACTOR,
        )

        for completed, params in enumerate(parameter_grid, start=1):
            results.append(analyze_parameter_set(params))

            if completed % PROGRESS_EVERY == 0 or completed == total_parameter_sets:
                percent_done = 100 * completed / total_parameter_sets
                print(f"Progress: {completed}/{total_parameter_sets} ({percent_done:.2f}%)")

    else:
        workers_to_use = min(N_WORKERS, total_parameter_sets)
        print(f"\nUsing {workers_to_use} worker process(es).")
        print("Each worker will build its own cached grayscale/blurred images once.")

        with mp.Pool(
            processes=workers_to_use,
            initializer=worker_initializer,
            initargs=(
                str(image_folder),
                test_images_dict,
                median_blur_list,
                REMOVE_DUPLICATE_CIRCLES,
                DUPLICATE_DISTANCE_FACTOR,
            ),
        ) as pool:
            completed = 0
            for row in pool.imap_unordered(analyze_parameter_set, parameter_grid, chunksize=CHUNK_SIZE):
                results.append(row)
                completed += 1

                if completed % PROGRESS_EVERY == 0 or completed == total_parameter_sets:
                    percent_done = 100 * completed / total_parameter_sets
                    print(f"Progress: {completed}/{total_parameter_sets} ({percent_done:.2f}%)")

    print("\nSorting results by Metric Score...")
    ranked_results = sorted(results, key=lambda row: float(row["Metric Score"]))
    best_row = ranked_results[0]
    top_rows = ranked_results[:TOP_N_TO_SAVE]

    columns = build_csv_columns(image_names)

    print("Writing CSV outputs...")
    write_csv(output_folder / "all_params_ranked.csv", ranked_results, columns)
    write_csv(output_folder / "top_n_params.csv", top_rows, columns)

    # Keep a filename compatible with the older workflow.
    # This contains the single best row plus any exact ties for best Metric Score.
    best_metric = float(best_row["Metric Score"])
    tied_best_rows = [
        row for row in ranked_results
        if math.isclose(float(row["Metric Score"]), best_metric, rel_tol=0.0, abs_tol=1e-12)
    ]
    write_csv(output_folder / "top_scoring_params.csv", tied_best_rows, columns)

    # Extra easy-to-find best-result files.
    # Open BEST_PARAMETER_SET_FOR_COPY_PASTE.txt first if you only need the final values.
    save_best_parameter_files(
        best_row=best_row,
        tied_best_rows=tied_best_rows,
        output_folder=output_folder,
        columns=columns,
    )

    if SAVE_BEST_COUNTS_CSV:
        save_best_counts_csv(best_row, output_folder)

    if SAVE_ANNOTATED_IMAGES:
        print("Saving annotated images for top parameter sets...")
        save_annotated_images_for_top_rows(
            top_rows=top_rows,
            image_folder=image_folder,
            output_folder=output_folder,
            parent_blurred_cache=parent_blurred_cache,
        )

    edge_warnings = find_range_edge_warnings(best_row)

    elapsed_seconds = time.time() - start_time
    write_run_summary(
        output_folder=output_folder,
        image_folder=image_folder,
        parameter_count=total_parameter_sets,
        best_row=best_row,
        edge_warnings=edge_warnings,
        elapsed_seconds=elapsed_seconds,
    )

    print_best_result(best_row, edge_warnings)

    print("\nProgram complete.")
    print(f"Main output folder:\n  {output_folder}")
    print("Most important files:")
    print("  BEST_PARAMETER_SET_FOR_COPY_PASTE.txt   <-- open this first")
    print("  BEST_PARAMETER_SET.csv")
    print("  BEST_PARAMETER_SET_TIES.csv")
    print("  best_parameter_set_counts.csv")
    print("  top_scoring_params.csv")
    print("  top_n_params.csv")
    print("  all_params_ranked.csv")
    print("  run_summary.txt")
    print(f"Elapsed time: {elapsed_seconds:.2f} seconds")


if __name__ == "__main__":
    # Required for safe multiprocessing on Windows.
    mp.freeze_support()
    run_optimizer()
