# Optimized Hough Circle Transform for Fast and Accurate Microparticle Detection and Counting

This repository contains Python scripts and example workflows for optimized Hough Circle Transform (HCT)-based microparticle detection and counting in microfluidic image sequences. The workflow was developed to reduce detection error and improve counting consistency for bead images, including 3 µm and 5 µm microparticles.

The repository includes graphical user interface (GUI) tools for visually testing HCT parameters and optimizer scripts for selecting HCT parameter sets based on manually counted training images.

## Repository contents

- `HCT_optimizer_manuscript_publication.py`  
  Cleaned optimizer script corresponding to the manuscript/publication workflow. This version fixes the parameter order, supports dynamic image columns, reads images relative to the script folder, and avoids unnecessary pandas dependency. 

- `HCT_optimizer_latest_optimized_version.py`  
  Expanded optimized version of the HCT optimizer. This version includes duplicate-circle removal, improved scoring based on percent error, multiprocessing, ranked output files, best-parameter summaries, annotated preview images, and range-edge warnings. 

- `HoughGUI_1size_saveScreenshots.py`  
  GUI-based HCT parameter tester for one bead-size setting. It allows users to adjust HCT parameters, preview detections, and save annotated screenshots with filenames indicating 3 µm or 5 µm bead counts. 

- `HoughGUI_2size_saveScreenshots.py`  
  GUI-based HCT parameter tester for two bead-size detection. It uses separate parameter sets for smaller and larger beads, with red and green detection outlines.

- `requirements.txt`  
  Python package requirements.

- `LICENSE`  
  Software license information.

## Recommended workflow

1. Use the GUI script to visually estimate appropriate HCT parameters.
2. Enter training image filenames and manual bead counts into the optimizer script.
3. Set HCT parameter ranges based on the GUI results.
4. Run the optimizer to test parameter combinations.
5. Review the ranked output files and annotated preview images.
6. Select the best parameter set.
7. Apply the optimized parameter set to the validation image set.
8. Compare HCT-detected counts with manual counts and calculate error rate.

## User instructions

This repository includes Hough Circle Transform-based tools for microparticle detection and counting in microfluidic image sequences. The included image files are demonstration examples only. Users should replace the example filenames, manual counts, and parameter ranges with values appropriate for their own experimental images.

## 1. Install required Python packages

Before running the scripts, install the required packages:

```bash
pip install -r requirements.txt
````

The main required packages are:

```bash
opencv-python
numpy
```

The GUI scripts also use Python's built-in `tkinter` package. In most Windows Python installations, `tkinter` is already included.

## 2. Prepare image files

Place the image files in the same folder as the Python script, or update the image folder path in the script.

Use raw experimental images for training and validation. Do not use images that already contain drawn circles, labels, scale bars, or screenshots from the GUI, because these markings may affect Hough Circle Transform detection.

Recommended image organization:

```text
project_folder/
│
├── HCT_optimizer_manuscript_publication.py
├── HCT_optimizer_latest_optimized_version.py
├── HoughGUI_1size_saveScreenshots.py
├── HoughGUI_2size_saveScreenshots.py
├── requirements.txt
│
├── cropImage1.jpg
├── cropImage2.jpg
├── cropImage3.jpg
└── ...
```

## 3. Use the GUI to estimate HCT parameters

Before running the optimizer, use the GUI scripts to visually estimate reasonable Hough Circle Transform parameter ranges.

Use:

```text
HoughGUI_1size_saveScreenshots.py
```

for images containing one main bead size.

Use:

```text
HoughGUI_2size_saveScreenshots.py
```

for images containing two bead sizes, such as smaller 3 µm beads and larger 5 µm beads.

The GUI allows users to adjust:

* Canny edge detector threshold, `param1`
* Center detection threshold, `param2`
* Minimum circle radius
* Maximum circle radius
* Minimum distance between detected circle centers
* Median blur level

After adjusting the sliders, preview the detection result. The detected circles and bead count will be shown visually and printed in the console. The GUI can also save annotated screenshots for documentation and comparison.

## 4. Enter training image filenames and manual counts

Open the optimizer script and locate the section labeled:

```python
USER ACTION REQUIRED
```

Replace the example image filenames and manual counts with your own data.

Example:

```python
test_images_dict = {
    "cropImage1.jpg": 134,
    "cropImage2.jpg": 156,
    "cropImage3.jpg": 173,
    "cropImage4.jpg": 182,
    "cropImage5.jpg": 182,
}
```

The filename must match exactly, including spelling, capitalization, and file extension.

For example, these are treated as different filenames:

```text
cropImage1.jpg
CropImage1.jpg
cropImage1.png
```

The manual count should be the human-counted number of beads in each image.

## 5. Set the HCT parameter ranges

After using the GUI to estimate reasonable values, edit the parameter ranges in the optimizer script.

Example:

```python
median_blur_list = [*range(1, 5, 2)]
param_1_list = [*range(20, 30, 1)]
param_2_list = [*range(20, 30, 1)]
min_radius_list = [*range(6, 8, 1)]
max_radius_list = [*range(15, 18, 1)]
min_dist_list = [*range(1, 3, 1)]
dp_default = 1
```

For Python `range(start, stop, step)`, the stop value is not included.

For example:

```python
range(20, 30, 1)
```

tests:

```text
20, 21, 22, 23, 24, 25, 26, 27, 28, 29
```

The median blur value must be a positive odd number, such as:

```text
1, 3, 5, 7
```

A value of `1` means little or no effective blurring.

## 6. Run the optimizer

For the manuscript-compatible workflow, run:

```bash
python HCT_optimizer_manuscript_publication.py
```

For the expanded optimized workflow with additional output files, duplicate-circle removal, range-edge warnings, and best-parameter summaries, run:

```bash
python HCT_optimizer_latest_optimized_version.py
```

The optimizer tests combinations of HCT parameters against the manually counted training images. The goal is to find parameter sets that minimize counting error across the training images.

## 7. Review output files

After the optimizer finishes, it will generate output files such as:

```text
all_params.csv
top_scoring_params.csv
top_n_params.csv
BEST_PARAMETER_SET.csv
BEST_PARAMETER_SET_FOR_COPY_PASTE.txt
best_parameter_set_counts.csv
run_summary.txt
annotated preview images
```

The most important file to open first is usually:

```text
BEST_PARAMETER_SET_FOR_COPY_PASTE.txt
```

This file contains the best HCT parameter values in a simple format that can be copied into the final validation workflow.

The annotated preview images should always be checked visually. A low numerical error is useful, but the detected circles should still be inspected to confirm that the algorithm is detecting real beads rather than image artifacts.

## 8. Apply the optimized parameters to validation images

After selecting the best parameter set, apply the same parameters to the validation image set.

For manuscript revision, the validation workflow used at least 50 frames. Manual counts should be obtained independently and then compared with the HCT-based detected counts.

Recommended validation data table:

```text
Frame number
Manual count from annotator 1
Manual count from annotator 2
Final manual count or averaged manual count
HCT detected count
Absolute error
Percent error
Success rate
```

## 9. Calculate counting error

For each image or frame:

```text
Absolute Error = |HCT detected count - manual count|
```

```text
Percent Error = Absolute Error / manual count × 100%
```

```text
Success Rate = 100% - Percent Error
```

If two independent annotators are used, the manual reference count can be calculated as the average of the two manual counts, or the discrepancy between annotators can be reported separately.

## 10. Important notes

* Use raw images for optimization and validation.
* Do not use GUI screenshots as optimizer input images.
* If the best parameter value is at the edge of the tested range, expand the range and rerun the optimizer.
* Radius values are in pixels, not micrometers.
* The expected radius range should be adjusted for each microscope magnification, camera setting, and bead size.
* Parameters optimized for 3 µm beads should not automatically be assumed to work for 5 µm beads.
* Parameters optimized on one imaging condition may need to be retrained if lighting, focus, magnification, exposure, or background changes.
* Duplicate-circle removal is a practical post-processing step, but it may not perfectly distinguish true overlapping beads from duplicate detections.
* Visual inspection of annotated output images is recommended before applying optimized parameters to final validation data.

## Output interpretation

The optimizer ranks parameter sets based on agreement between detected bead counts and manual bead counts.

For the expanded optimized workflow, the main scoring approach is based on percent error:

```text
Percent Error = |Detected Count - Manual Count| / Manual Count
```

The optimized version ranks parameter sets using:

```text
Metric Score = Mean Percent Error + SD Percent Error
```

Lower values indicate better performance. This scoring approach favors parameter sets that are accurate on average and consistent across the training images.

## Example application

A typical use case is:

1. Collect a sequence of raw microfluidic bead images.
2. Select representative training images.
3. Manually count the beads in each training image.
4. Use the GUI to estimate reasonable HCT parameter ranges.
5. Run the optimizer using the training images and manual counts.
6. Select the best parameter set.
7. Apply the selected parameter set to the validation image sequence.
8. Compare HCT-based counts with manual counts to calculate detection error and success rate.

## Citation

If you use this repository, please cite the associated manuscript and archived software release.

```text
Optimized Hough Circle Transform for Fast and Accurate Microparticle Detection and Counting.
GitHub repository and Zenodo software archive.
```

A version-specific Zenodo DOI should be cited after the GitHub release is archived on Zenodo.

## License

This software is licensed under the MIT License.

Please see the `LICENSE` file for details.

Copyright 2026 Trevor Gerdes. All rights reserved.

```
