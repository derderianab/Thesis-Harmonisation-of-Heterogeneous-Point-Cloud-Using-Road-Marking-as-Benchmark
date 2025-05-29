# Harmonisation of Heterogeneous Point Cloud Using Road Markings as Benchmarks

This repository contains the processing scripts used in the MSc Geomatics thesis project at TU Delft.  
The project focuses on using road markings as geometric benchmarks to align heterogeneous point cloud datasets acquired from different platforms and acquisition periods.

The work builds upon the method developed by Heide (2024), which introduced an adaptive thresholding approach for automatic extraction of road markings from LiDAR datasets with varying intensity ranges.  
The automatically extracted road markings used in this project were provided by Rijkswaterstaat and are used exclusively for evaluating the performance of road markings as co-registration benchmarks.

---


## Repository Structure

- **`scripts/`**: Contains the main Python scripts that implement the core methodology of the thesis:
  1. `0_calc_thresholds.py`: Computes geometric feature thresholds as part of the automatic road marking extraction pipeline.
  2. `1_correspondence_ransac.py`: Establishes correspondences between road markings using a RANSAC-based inlier voting approach.
  3. `2_alignment_weighted_centroid.py`: Computes the transformation matrix using a RANSAC-weighted centroid alignment method.
  4. `3_apply_transformation.py`: Applies the estimated transformation to the target point cloud dataset.

- **`data/`**: Placeholder for datasets used in the project:
  - `road_mark_segments/`: Training data for computing geometric thresholds.
  - `las_point_cloud/`: LiDAR point cloud datasets from Prorail SpoorInBeeld ([spoorinbeeld.nl](https://spoorinbeeld.nl)) and AHN via [geotiles.nl](https://geotiles.nl).
  - `geopackage_road_markings/`: Road marking benchmarks used in the alignment process (automatically extracted and manually digitised).

---


## Setup

**Python version:** 3.10+

Install the required Python packages:

```bash
pip install -r requirements.txt
```

---


## How To Use The Scripts

These are example usage to run the script in command prompt


### Calculate Road Markings Geometric Feature Thresholds

This script runs on road marking segments stored in the same folder in LAS or LAZ format.

Example use:
```
python 0_rm_geom_threshold_calc.py \
  --input_dir "D:\geomatics\geom_threshold_calc\roadmarks_segments" \
  --hist_output_dir "D:\geomatics\geom_threshold_calc\output\histograms" \
  --txt_output_path "D:\geomatics\geom_threshold_calc\output\" \
```

#### Required Arguments:
- `--input_dir`: Folder containing segmented `.las` files
- `--hist_output_dir`: Directory where histogram plots will be saved
- `--txt_output_path`: Output `.txt` file for writing the computed threshold ranges

#### Optional Argument:
- `--z_threshold`: Z-score threshold for intensity filtering (default is `-1.5`)


### Calculate Road Markings Geometric Feature Thresholds

This script runs on two Geopackages containing road marking line features, one from the reference point cloud and one from the target. The output will be a new Geopackage containing established target lines with respect to reference lines.

Example use:
```
python 1_correspondence_ransac.py \
  --ref_path "D:\geomatics\corr_establish\reference_line_benchmark.gpkg" \
  --tgt_path "D:\geomatics\corr_establish\target_line_benchmark.gpkg" \
  --output_path "D:\geomatics\corr_establish\output\<custom_output_name>.gpkg" \
```

#### Required Arguments:

- `--ref_path` : Path to the reference Geopackage  
- `--tgt_path` : Path to the target Geopackage  
- `--output_path` : Path where the output Geopackage will be saved, output name is custom

#### Optional Arguments:

- `--nn_search_radius` : Radius (in meters) for initial nearest-neighbour matching (default: `1.0`)  
- `--ransac_iterations` : Number of RANSAC iterations (default: `1000`)  
- `--ransac_nn_radius` : Distance threshold for inliers in RANSAC (default: `0.112`)  
- `--ransac_sample_n` : Number of correspondences sampled per RANSAC iteration (default: `3`)


### Apply Weighted Centroid Alignment Transformation

This script runs on a Geopackage containing target road marking lines and their differences with the corresponding reference road marking lines saved in their attribute 
(e.g., centroid differences and inlier counts). It computes a weighted rigid transformation based on these attributes and applies it to the line geometries.

Example use:

```
python 3_apply_transformation.py \
  --tgt_path "D:\geomatics\corr_establish\output\align_Prorail2019_to_AHN3.gpkg" \
  --output_gpkg "D:\geomatics\alignment\transformed_lines\<custom_output_name>.gpkg" \
  --transform_txt "D:\geomatics\alignment\transformation_matrix\<custom_output_name>.txt"
```

#### Required Arguments:

- `--tgt_path` : Path to the input Geopackage (target lines after correspondence established) 
- `--output_gpkg` : Path where the transformed Geopackage will be saved, the output name is custom
- `--transform_txt` : Path where the transformation matrix will be saved, the output name is custom


### Apply Transformation Matrix

This script applies a 4×4 transformation matrix (rotation + translation) to a LAS/LAZ point cloud.
The matrix should be generated from the weighted centroid alignment code and saved as a .txt file.

Example use:

```
python apply_las_transformation.py \
  --transformation_txt "D:\geomatics\alignment\transformation_matrix\align_Prorail2019_to_AHN3.txt" \
  --input_las_path "D:\geomatics\point_cloud\Prorail2019.laz" \
  --output_las_path "D:\geomatics\point_cloud\transformed\aligned_Prorail2019_to_AHN3.las"
```

#### Required Arguments:

- `--transformation_txt` : Path to the `.txt` file containing the 4×4 transformation matrix  
- `--input_las_path` : Path to the input point cloud (`.las` or `.laz`)  
- `--output_las_path` : Path to save the transformed point cloud (`.las` or `.laz`)
