"""
HCT_optimizer_final_fixed.py

Fixed and cleaned version of HCT_optimizer_final.py for Hough Circle Transform
parameter training.

Main fixes compared with the original script:
1. Correct parameter order:
   [DP, Minimum Distance, Median Blur, Param 1, Param 2, Minimum Radius, Maximum Radius]
2. Dynamic image columns, so the script works with any number of training images.
3. Average accuracy is divided by the actual number of training images, not hard-coded as 10.
4. Only hough_secondary_analysis_*.csv files are combined, so old all_params.csv files do not contaminate new runs.
5. The script reads images relative to the folder containing this .py file.
6. No pandas dependency is required.

Before running:
- Put this script in the same folder as your training images, or set IMAGE_DIR below.
- Replace test_images_dict with your exact training image file names and manual counts.
- Adjust the Hough parameter ranges using values estimated from the GUI.
"""

from __future__ import annotations

import csv
import datetime as dt
import math
import multiprocessing as mp
import shutil
import time
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import cv2 as cv
import numpy as np


# ==========================================
# USER ACTION REQUIRED: INPUT YOUR DATA HERE
# ==========================================
# Format: "your_image_name.jpg": manual_count,
# Put the images in the same folder as this script, or change IMAGE_DIR below.

test_images_dict: Dict[str, int] = {
    "cropImage1.jpg": 134,
    "cropImage2.jpg": 156,
    "cropImage3.jpg": 173,
    "cropImage4.jpg": 182,
    "cropImage5.jpg": 182,
    "cropImage6.jpg": 180,
    "cropImage7.jpg": 170,
    "cropImage8.jpg": 185,
    "cropImage9.jpg": 203,
    "cropImage10.jpg": 195,
}


# ==========================================
# USER ACTION REQUIRED: SET PARAMETER RANGES
# ==========================================
# Use the GUI to estimate reasonable ranges before training.
# median_blur values must be positive odd numbers: 1, 3, 5, 7, etc.

median_blur_list = [*range(1, 5, 2)]      # 1, 3, 5, 7, etc.
param_1_list = [*range(20, 30, 1)]         # Canny edge detector threshold
param_2_list = [*range(20, 30, 1)]         # Center detection threshold
min_radius_list = [*range(6, 8, 1)]       # Minimum bead radius, in pixels
max_radius_list = [*range(15, 18, 1)]       # Maximum bead radius, in pixels
min_dist_list = [*range(1, 3, 1)]         # Minimum distance between bead centers

# dp is a Hough Circle Transform parameter that can often be kept constant.
dp_default = 1


# ==========================================
# OUTPUT AND PERFORMANCE SETTINGS
# ==========================================
SCRIPT_DIR = Path(__file__).resolve().parent
IMAGE_DIR = SCRIPT_DIR
OUTPUT_DIR = SCRIPT_DIR / "optimizer_output"

# Keep this at 8 if your computer has 8 or more CPU cores.
# Reduce to 4 if the computer becomes slow or hot.
NUM_WORKERS = min(8, max(1, mp.cpu_count()))

# If True, old optimizer CSV and top-image outputs in OUTPUT_DIR are removed at the start.
OVERWRITE_OLD_OUTPUTS = True


ParamSet = Tuple[int, int, int, int, int, int, int]
PARAMETER_FIELDS = [
    "DP",
    "Minimum Distance",
    "Median Blur",
    "Param 1",
    "Param 2",
    "Minimum Radius",
    "Maximum Radius",
]
SUMMARY_FIELDS = [
    "Average % of Correct Count",
    "Metric 1",
    "Metric 2 (S.D.)",
    "MetricSum",
]


def image_path(image_name: str) -> Path:
    """Return the absolute path for an image listed in test_images_dict."""
    path = Path(image_name)
    if path.is_absolute():
        return path
    return IMAGE_DIR / path


def build_output_fields() -> List[str]:
    """Build CSV headers for however many training images are in test_images_dict."""
    fields = list(PARAMETER_FIELDS)
    for index, image_name in enumerate(test_images_dict.keys(), start=1):
        fields.append(f"Image {index} Beads Found ({image_name})")
        fields.append(f"Image {index} Found/Total ({image_name})")
    fields.extend(SUMMARY_FIELDS)
    return fields


def validate_user_inputs() -> None:
    """Catch common setup mistakes before a long training run starts."""
    if not test_images_dict:
        raise ValueError("test_images_dict is empty. Add training image filenames and manual counts.")

    missing_images = [str(image_path(name)) for name in test_images_dict if not image_path(name).exists()]
    if missing_images:
        formatted = "\n".join(missing_images)
        raise FileNotFoundError(
            "The following training images were not found. "
            "Check spelling, file extension, and folder location:\n" + formatted
        )

    bad_counts = {name: count for name, count in test_images_dict.items() if count <= 0}
    if bad_counts:
        raise ValueError(f"Manual counts must be positive numbers. Check: {bad_counts}")

    bad_blur_values = [value for value in median_blur_list if value <= 0 or value % 2 == 0]
    if bad_blur_values:
        raise ValueError(
            "median_blur_list can only contain positive odd numbers. "
            f"Bad values: {bad_blur_values}"
        )

    if not all([
        param_1_list,
        param_2_list,
        min_radius_list,
        max_radius_list,
        min_dist_list,
        median_blur_list,
    ]):
        raise ValueError("One or more parameter lists are empty. Check the range() values.")


def clean_old_outputs() -> None:
    """Remove old result files so a rerun starts cleanly."""
    OUTPUT_DIR.mkdir(exist_ok=True)
    if not OVERWRITE_OLD_OUTPUTS:
        return

    patterns = [
        "hough_secondary_analysis_*.csv",
        "all_params.csv",
        "top_scoring_params.csv",
        "top_param_set_*.png",
    ]
    for pattern in patterns:
        for path in OUTPUT_DIR.glob(pattern):
            path.unlink(missing_ok=True)


def build_param_list() -> List[ParamSet]:
    """
    Build all parameter combinations.

    IMPORTANT: The order below matches the CSV header and the analysis function:
    DP, Minimum Distance, Median Blur, Param 1, Param 2, Minimum Radius, Maximum Radius.
    """
    parameter_sets_all: List[ParamSet] = []

    for median_blur in median_blur_list:
        for param_1 in param_1_list:
            for param_2 in param_2_list:
                for min_radius in min_radius_list:
                    for max_radius in max_radius_list:
                        if max_radius < min_radius:
                            continue
                        for min_dist in min_dist_list:
                            parameter_sets_all.append(
                                (
                                    int(dp_default),
                                    int(min_dist),
                                    int(median_blur),
                                    int(param_1),
                                    int(param_2),
                                    int(min_radius),
                                    int(max_radius),
                                )
                            )

    if not parameter_sets_all:
        raise ValueError("No valid parameter sets were created. Check radius ranges and other settings.")

    print(f"Total parameter sets to be tested: {len(parameter_sets_all)}")
    print(f"Time: {dt.datetime.now()}")
    return parameter_sets_all


def split_into_chunks(items: Sequence[ParamSet], number_of_chunks: int) -> List[List[ParamSet]]:
    """Split a parameter list into approximately equal chunks for multiprocessing."""
    if number_of_chunks <= 1:
        return [list(items)]

    chunk_size = math.ceil(len(items) / number_of_chunks)
    chunks = [list(items[i:i + chunk_size]) for i in range(0, len(items), chunk_size)]
    return [chunk for chunk in chunks if chunk]


def detect_circles_for_image(image_name: str, param_set: ParamSet) -> int:
    """Run HoughCircles on one image and return the number of circles found."""
    d_p, min_dist, median_blur, parameter_one, parameter_two, min_radius, max_radius = param_set

    img = cv.imread(str(image_path(image_name)), cv.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"OpenCV could not read image: {image_path(image_name)}")

    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    gray = cv.medianBlur(gray, median_blur)

    circles = cv.HoughCircles(
        image=gray,
        method=cv.HOUGH_GRADIENT,
        dp=d_p,
        minDist=min_dist,
        param1=parameter_one,
        param2=parameter_two,
        minRadius=min_radius,
        maxRadius=max_radius,
    )

    if circles is None:
        return 0
    return int(len(circles[0, :]))


def evaluate_param_set(param_set: ParamSet) -> List[object]:
    """Evaluate one parameter set against all manually counted training images."""
    row: List[object] = list(param_set)
    found_div_correct_values: List[float] = []

    for image_name, manual_count in test_images_dict.items():
        circles_found = detect_circles_for_image(image_name, param_set)
        found_div_correct = circles_found / manual_count

        row.append(circles_found)
        row.append(found_div_correct)
        found_div_correct_values.append(found_div_correct)

    average_found_div_correct = float(np.mean(found_div_correct_values))
    metric_1 = abs(1.0 - average_found_div_correct)
    metric_2_sd = float(np.std(found_div_correct_values))
    metric_sum = metric_1 + metric_2_sd

    row.append(average_found_div_correct)
    row.append(metric_1)
    row.append(metric_2_sd)
    row.append(metric_sum)

    return row


def analyze_subset(subset_id: int, subset: Sequence[ParamSet], fields: Sequence[str]) -> None:
    """Analyze one subset of parameter sets and save one CSV file."""
    output_csv = OUTPUT_DIR / f"hough_secondary_analysis_{subset_id}.csv"
    subset_count = len(subset)

    print(f"Subset {subset_id}: analysis beginning with {subset_count} parameter sets.")

    with output_csv.open("w", newline="") as file_handle:
        writer = csv.writer(file_handle)
        writer.writerow(fields)

        for calc_counter, param_set in enumerate(subset, start=1):
            writer.writerow(evaluate_param_set(param_set))

            if calc_counter % 1000 == 0 or calc_counter == subset_count:
                percent_done = round(100 * calc_counter / subset_count, 2)
                print(
                    f"Subset {subset_id} calculations: {calc_counter}/{subset_count} "
                    f"({percent_done}%)    Time: {dt.datetime.now()}"
                )

    print(f"Subset {subset_id} complete. Saved: {output_csv}")


def read_subset_results(fields: Sequence[str]) -> List[dict]:
    """Read all subset CSVs and return the rows as dictionaries."""
    subset_files = sorted(OUTPUT_DIR.glob("hough_secondary_analysis_*.csv"))
    if not subset_files:
        raise FileNotFoundError("No hough_secondary_analysis_*.csv files were found.")

    rows: List[dict] = []
    for subset_file in subset_files:
        with subset_file.open("r", newline="") as file_handle:
            reader = csv.DictReader(file_handle)
            for row in reader:
                rows.append({field: row.get(field, "") for field in fields})
    return rows


def write_all_params(rows: Sequence[dict], fields: Sequence[str]) -> Path:
    """Save all subset rows into all_params.csv."""
    all_params_path = OUTPUT_DIR / "all_params.csv"
    with all_params_path.open("w", newline="") as file_handle:
        writer = csv.DictWriter(file_handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return all_params_path


def write_top_scoring_params(rows: Sequence[dict], fields: Sequence[str]) -> Tuple[Path, List[dict]]:
    """Find and save the parameter row(s) with the lowest MetricSum."""
    if not rows:
        raise ValueError("No analysis rows were available to score.")

    best_metric = min(float(row["MetricSum"]) for row in rows)
    top_rows = [row for row in rows if float(row["MetricSum"]) == best_metric]

    top_params_path = OUTPUT_DIR / "top_scoring_params.csv"
    with top_params_path.open("w", newline="") as file_handle:
        writer = csv.DictWriter(file_handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(top_rows)

    print(f"Best MetricSum: {best_metric}")
    print(f"Number of top-scoring parameter set(s): {len(top_rows)}")
    return top_params_path, top_rows


def draw_top_param_images(top_rows: Sequence[dict]) -> None:
    """Save detected-circle preview images for each top-scoring parameter set."""
    for param_index, row in enumerate(top_rows, start=1):
        param_set: ParamSet = (
            int(float(row["DP"])),
            int(float(row["Minimum Distance"])),
            int(float(row["Median Blur"])),
            int(float(row["Param 1"])),
            int(float(row["Param 2"])),
            int(float(row["Minimum Radius"])),
            int(float(row["Maximum Radius"])),
        )
        d_p, min_dist, median_blur, parameter_one, parameter_two, min_radius, max_radius = param_set

        for image_index, image_name in enumerate(test_images_dict.keys(), start=1):
            img = cv.imread(str(image_path(image_name)), cv.IMREAD_COLOR)
            if img is None:
                raise ValueError(f"OpenCV could not read image: {image_path(image_name)}")

            gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            gray = cv.medianBlur(gray, median_blur)
            circles = cv.HoughCircles(
                image=gray,
                method=cv.HOUGH_GRADIENT,
                dp=d_p,
                minDist=min_dist,
                param1=parameter_one,
                param2=parameter_two,
                minRadius=min_radius,
                maxRadius=max_radius,
            )

            if circles is not None:
                circles = np.uint16(np.around(circles))
                for circle in circles[0, :]:
                    center = (int(circle[0]), int(circle[1]))
                    radius = int(circle[2])
                    cv.circle(img, center, 1, (0, 100, 100), 1)
                    cv.circle(img, center, radius, (0, 255, 0), 1)

            safe_stem = Path(image_name).stem.replace(" ", "_")
            output_image = OUTPUT_DIR / f"top_param_set_{param_index}_image_{image_index}_{safe_stem}.png"
            cv.imwrite(str(output_image), img)

    print("Top parameter output images saved.")


def run_parallel_analysis(parameter_sets: Sequence[ParamSet], fields: Sequence[str]) -> None:
    """Run subset analyses in parallel."""
    worker_count = min(NUM_WORKERS, len(parameter_sets))
    subsets = split_into_chunks(parameter_sets, worker_count)

    print(f"Using {len(subsets)} worker process(es).")

    processes: List[mp.Process] = []
    for subset_id, subset in enumerate(subsets, start=1):
        process = mp.Process(target=analyze_subset, args=(subset_id, subset, fields))
        process.start()
        processes.append(process)

    for process in processes:
        process.join()

    failed = [process.exitcode for process in processes if process.exitcode != 0]
    if failed:
        raise RuntimeError(f"At least one worker process failed. Exit codes: {failed}")


def main() -> None:
    start_time_total = time.time()

    print("Hough Circle Transform optimizer started.")
    print(f"Script folder: {SCRIPT_DIR}")
    print(f"Image folder: {IMAGE_DIR}")
    print(f"Output folder: {OUTPUT_DIR}")
    print(f"Training images: {len(test_images_dict)}")

    validate_user_inputs()
    clean_old_outputs()
    fields = build_output_fields()
    parameter_sets = build_param_list()
    run_parallel_analysis(parameter_sets, fields)

    rows = read_subset_results(fields)
    all_params_path = write_all_params(rows, fields)
    top_params_path, top_rows = write_top_scoring_params(rows, fields)
    draw_top_param_images(top_rows)

    print(f"All parameter results saved to: {all_params_path}")
    print(f"Top-scoring parameter set(s) saved to: {top_params_path}")
    print("Program complete. Bye for now!")
    print("--- %s seconds ---" % round(time.time() - start_time_total, 2))


if __name__ == "__main__":
    mp.freeze_support()  # Helpful for Windows multiprocessing.
    main()
