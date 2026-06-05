# Optimized Hough Circle Transform for Fast and Accurate Microparticle Detection and Counting
This repository contains Python scripts and example image datasets for optimized Hough Circle Transform-based microparticle detection and counting in microfluidic image sequences. The workflow was developed to reduce detection error and improve counting consistency for 3 µm and 5 µm bead images.

## Repository contents

- `HCT_optimizer_manuscript_publication.py`  
  Cleaned optimizer script corresponding to the manuscript/publication workflow.

- `HCT_optimizer_latest_optimized_version.py`  
  Improved optimizer script with expanded documentation, duplicate-circle removal, ranked parameter outputs, and best-parameter summary files.

- `HoughGUI_1size_saveScreenshots.py`  
  GUI-based Hough Circle Transform parameter tester for one bead-size setting, with screenshot export.

- `HoughGUI_2size_saveScreenshots.py`  
  GUI-based Hough Circle Transform parameter tester for two bead-size detection, using separate parameter sets for smaller and larger beads.

- `requirements.txt`  
  Python package requirements.

- `LICENSE`  
  Software license information.

## Revision update

This version includes updates made during manuscript revision, including:

- Improved Hough Circle Transform parameter optimization workflow.
- Separate optimization and training support for 3 µm and 5 µm bead datasets.
- Expanded validation workflow using at least 50 frames.
- Manual counting comparison using two independent annotators.
- Updated success-rate and error-rate calculation workflow.
- Improved documentation for image filenames, manual counts, and particle radius settings.

## Usage instructions

The included images are demonstration examples only.

To use the code with your own experimental images:

1. Place the image files in the same folder as the Python script, or update the image folder path in the script.
2. Open the Python script.
3. Locate the section labeled `USER ACTION REQUIRED`.
4. Replace the example image filenames with your own image filenames.
5. Replace the example manual bead counts with your own manually counted values.
6. Adjust the expected particle radius range if analyzing particles with a different diameter.
7. Run the script in Python or PyCharm.

Update README for revised HCT workflow
