import os
import numpy as np
import laspy
import matplotlib.pyplot as plt
import argparse

# Load las file
def load_las(file_path):
    las = laspy.read(file_path)
    xyz = np.vstack((las.x, las.y, las.z)).T
    intensity = np.array(las.intensity)
    return xyz, intensity

# Z-score filtering
def z_score_filtering(intensity, threshold):
    average = intensity.mean()
    std = np.std(intensity)
    z_scores = (intensity - average) / std
    filtered_indices = np.where(z_scores >= threshold)[0]
    return filtered_indices

# Eigenvalues computation
def compute_eigenvalues(xyz, filtered_indices):
    filtered_xyz = xyz[filtered_indices]
    mean_xyz = np.mean(filtered_xyz, axis=0)
    centered_points = filtered_xyz - mean_xyz
    cov_matrix = np.cov(centered_points, rowvar=False)
    eigenvalues, _ = np.linalg.eig(cov_matrix)
    return np.sort(eigenvalues)[::-1]

# Geometric features computation (referring to Heide, 2024)
def compute_parameter(eigenvalues):
    l1, l2, l3 = eigenvalues
    linearity = (l1 - l2) / l1
    planarity = (l2 - l3) / l1
    sphericity = l3 / l1
    anisotropy = (l1 - l3) / l1
    sum_eigenvalues = l1 + l2 + l3
    curvature = l3 / sum_eigenvalues
    return linearity, planarity, sphericity, anisotropy, sum_eigenvalues, curvature

# Interquartile range filtering
def iqr_filtering(values):
    Q1 = np.percentile(values, 25)
    Q3 = np.percentile(values, 75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return [v for v in values if lower_bound <= v <= upper_bound]

# Save histogram
def save_histogram(name, values, out_dir):
    plt.figure(figsize=(8, 5))
    plt.hist(values, bins=20, edgecolor="black", alpha=0.7)
    plt.xlabel("Feature Value")
    plt.ylabel("Frequency")
    plt.title("Histogram of " + name)
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    out_path = os.path.join(out_dir, name.replace(" ", "_").lower() + ".png")
    plt.savefig(out_path)
    plt.close()

# Save thresholds
def save_thresholds_txt(threshold_dict, output_path):
    with open(output_path, "w") as f:
        for name, values in threshold_dict.items():
            if values:
                f.write(f"{name} Value Range: {min(values):.4f} to {max(values):.4f}\n")

# Main
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute geometric thresholds from LAS files.")
    parser.add_argument("--input_dir", required=True, help="Input directory with LAS files")
    parser.add_argument("--hist_output_dir", required=True, help="Directory to save histograms")
    parser.add_argument("--txt_output_path", required=True, help="Path to save threshold values")
    parser.add_argument("--z_threshold", type=float, default=-1.5, help="Z-score threshold for intensity filtering")
    args = parser.parse_args()

    os.makedirs(args.hist_output_dir, exist_ok=True)
    os.makedirs(os.path.dirname(args.txt_output_path), exist_ok=True)

    linearity_values = []
    planarity_values = []
    sphericity_values = []
    anisotropy_values = []
    sum_eigenvalues_values = []
    curvature_values = []
    filenames = []

    for file_name in os.listdir(args.input_dir):
        file_path = os.path.join(args.input_dir, file_name)
        xyz, intensity = load_las(file_path)
        filtered_indices = z_score_filtering(intensity, args.z_threshold)
        if len(filtered_indices) == 0:
            print(f"Skipping {file_name}: no points passed filtering.")
            continue

        eigenvalues = compute_eigenvalues(xyz, filtered_indices)
        if len(eigenvalues) != 3 or np.any(np.isnan(eigenvalues)):
            print(f"Skipping {file_name}: invalid eigenvalue computation.")
            continue

        linearity, planarity, sphericity, anisotropy, sum_eigenvalues, curvature = compute_parameter(eigenvalues)
        linearity_values.append(linearity)
        planarity_values.append(planarity)
        sphericity_values.append(sphericity)
        anisotropy_values.append(anisotropy)
        sum_eigenvalues_values.append(sum_eigenvalues)
        curvature_values.append(curvature)
        filenames.append(file_name)

    original_values = {
        "Linearity": linearity_values,
        "Planarity": planarity_values,
        "Sphericity": sphericity_values,
        "Anisotropy": anisotropy_values,
        "Sum of Eigenvalues": sum_eigenvalues_values,
        "Change in Curvature": curvature_values
    }

    # Apply IQR filtering
    filtered_linearity = iqr_filtering(linearity_values)
    filtered_planarity = iqr_filtering(planarity_values)
    filtered_sphericity = iqr_filtering(sphericity_values)
    filtered_anisotropy = iqr_filtering(anisotropy_values)
    filtered_sum_eigenvalues = iqr_filtering(sum_eigenvalues_values)
    filtered_curvature = iqr_filtering(curvature_values)

    # Save histograms
    save_histogram("Linearity (All)", linearity_values, args.hist_output_dir)
    save_histogram("Linearity (Filtered)", filtered_linearity, args.hist_output_dir)

    save_histogram("Planarity (All)", planarity_values, args.hist_output_dir)
    save_histogram("Planarity (Filtered)", filtered_planarity, args.hist_output_dir)

    save_histogram("Sphericity (All)", sphericity_values, args.hist_output_dir)
    save_histogram("Sphericity (Filtered)", filtered_sphericity, args.hist_output_dir)

    save_histogram("Anisotropy (All)", anisotropy_values, args.hist_output_dir)
    save_histogram("Anisotropy (Filtered)", filtered_anisotropy, args.hist_output_dir)

    save_histogram("Sum of Eigenvalues (All)", sum_eigenvalues_values, args.hist_output_dir)
    save_histogram("Sum of Eigenvalues (Filtered)", filtered_sum_eigenvalues, args.hist_output_dir)

    save_histogram("Change in Curvature (All)", curvature_values, args.hist_output_dir)
    save_histogram("Change in Curvature (Filtered)", filtered_curvature, args.hist_output_dir)

    thresholds = {
        "Linearity": filtered_linearity,
        "Planarity": filtered_planarity,
        "Sphericity": filtered_sphericity,
        "Anisotropy": filtered_anisotropy,
        "Sum of Eigenvalues": filtered_sum_eigenvalues,
        "Change in Curvature": filtered_curvature
    }

    save_thresholds_txt(thresholds, args.txt_output_path)

    print(f"\nThresholds saved to: {args.txt_output_path}")
    print(f"Histograms saved to: {args.hist_output_dir}")
