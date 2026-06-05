# Optimized Hough Circle Transform for Fast and Accurate Microparticle Detection and Counting

This repository contains Python scripts and example workflows for optimized Hough Circle Transform (HCT)-based microparticle detection and counting in microfluidic image sequences. The workflow was developed to reduce detection error and improve counting consistency for bead images, including 3 µm and 5 µm microparticles.

The repository includes graphical user interface (GUI) tools for visually testing HCT parameters and optimizer scripts for selecting HCT parameter sets based on manually counted training images.

## Repository contents

- `HCT_optimizer_manuscript_publication.py`  
  Cleaned optimizer script corresponding to the manuscript/publication workflow. This version fixes the parameter order, supports dynamic image columns, reads images relative to the script folder, and avoids unnecessary pandas dependency. :contentReference[oaicite:0]{index=0}

- `HCT_optimizer_latest_optimized_version.py`  
  Expanded optimized version of the HCT optimizer. This version includes duplicate-circle removal, improved scoring based on percent error, multiprocessing, ranked output files, best-parameter summaries, annotated preview images, and range-edge warnings. :contentReference[oaicite:1]{index=1}

- `HoughGUI_1size_saveScreenshots.py`  
  GUI-based HCT parameter tester for one bead-size setting. It allows users to adjust HCT parameters, preview detections, and save annotated screenshots with filenames indicating 3 µm or 5 µm bead counts. :contentReference[oaicite:2]{index=2}

- `HoughGUI_2size_saveScreenshots.py`  
  GUI-based HCT parameter tester for two bead-size detection. It uses separate parameter sets for smaller and larger beads, with red and green detection outlines. :contentReference[oaicite:3]{index=3}

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
