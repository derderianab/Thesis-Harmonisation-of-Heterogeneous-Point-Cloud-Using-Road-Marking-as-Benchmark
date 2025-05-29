# Script: 1_correspondence_ransac.py

import argparse
import geopandas as gpd
import numpy as np
import os
import uuid
from scipy.spatial import cKDTree
from shapely.geometry import LineString
import open3d as o3d

def parse_args():
    parser = argparse.ArgumentParser(description="RANSAC-based correspondence matching between road markings.")
    parser.add_argument('--ref_path', required=True, help='Path to the reference GeoPackage file')
    parser.add_argument('--tgt_path', required=True, help='Path to the target GeoPackage file')
    parser.add_argument('--output_path', required=True, help='Path to the output GeoPackage file')
    parser.add_argument('--nn_search_radius', type=float, default=1.0, help='Initial nearest neighbour search radius (in meters)')
    parser.add_argument('--ransac_iterations', type=int, default=1000, help='Number of RANSAC iterations')
    parser.add_argument('--ransac_nn_radius', type=float, default=0.112, help='RANSAC inlier threshold radius')
    parser.add_argument('--ransac_sample_n', type=int, default=3, help='Number of samples per RANSAC iteration')
    return parser.parse_args()

def load_data(reference_path, target_path):
    ref_gdf = gpd.read_file(reference_path)
    tgt_gdf = gpd.read_file(target_path)

    for gdf in [ref_gdf, tgt_gdf]:
        if not gdf.geometry.iloc[0].has_z:
            gdf.geometry = gdf.geometry.apply(
                lambda geom: LineString(np.hstack([np.array(geom.coords), np.zeros((len(geom.coords), 1))])))

    return ref_gdf, tgt_gdf

def compute_centroids(gdf):
    centroids = []
    for line in gdf.geometry:
        if line.geom_type == "MultiLineString":
            line = line[0]
        coords = np.array(line.coords)
        centroids.append(coords.mean(axis=0))
    return np.vstack(centroids)

def nearest_neighbor_match(ref_centroids, tgt_centroids, search_radius=1.0):
    tree = cKDTree(ref_centroids)
    distances, indices = tree.query(tgt_centroids, k=1, distance_upper_bound=search_radius)
    valid = distances < search_radius
    matches = np.column_stack((np.where(valid)[0], indices[valid]))
    return matches.tolist()

def angle_between(v1, v2):
    v1_u = v1 / (np.linalg.norm(v1) + 1e-8)
    v2_u = v2 / (np.linalg.norm(v2) + 1e-8)
    dot = np.clip(np.dot(v1_u, v2_u), -1.0, 1.0)
    return np.degrees(np.arccos(dot))

def calculate_differences(ref_gdf, tgt_gdf, matches):
    diffs = []
    for tgt_idx, ref_idx in matches:
        ref_geom = ref_gdf.iloc[ref_idx].geometry
        tgt_geom = tgt_gdf.iloc[tgt_idx].geometry

        ref_coords = np.array(ref_geom.coords)
        tgt_coords = np.array(tgt_geom.coords)

        ref_vec = ref_coords[-1] - ref_coords[0]
        tgt_vec = tgt_coords[-1] - tgt_coords[0]
        if np.dot(ref_vec, tgt_vec) < 0:
            tgt_coords = tgt_coords[::-1]

        ref_centroid = np.mean(ref_coords, axis=0)
        tgt_centroid = np.mean(tgt_coords, axis=0)
        centroid_diff = np.linalg.norm(ref_centroid - tgt_centroid)
        centroid_diff_xyz = tgt_centroid - ref_centroid

        angle_horizontal = angle_between(ref_vec[:2], tgt_vec[:2])
        ref_slope = np.array([np.linalg.norm(ref_vec[:2]), ref_vec[2]])
        tgt_slope = np.array([np.linalg.norm(tgt_vec[:2]), tgt_vec[2]])
        angle_vertical = angle_between(ref_slope, tgt_slope)

        length_ref = np.sum(np.linalg.norm(np.diff(ref_coords, axis=0), axis=1))
        length_tgt = np.sum(np.linalg.norm(np.diff(tgt_coords, axis=0), axis=1))
        length_diff = length_tgt - length_ref

        diffs.append((
            tgt_idx, ref_idx,
            centroid_diff, *centroid_diff_xyz,
            length_diff,
            angle_horizontal,
            angle_vertical
        ))
    return diffs

def ransac_with_inlier_tracking(ref_centroids, tgt_centroids, matches, n_iterations=100, distance_thresh=0.5,
                                ransac_n=3):
    inlier_counter = np.zeros(len(tgt_centroids), dtype=int)
    total_iterations = n_iterations
    best_R, best_t = None, None
    best_inliers = []
    best_inlier_count = 0

    correspondences = np.array(matches)

    for _ in range(n_iterations):
        try:
            result = o3d.pipelines.registration.registration_ransac_based_on_correspondence(
                o3d.geometry.PointCloud(points=o3d.utility.Vector3dVector(tgt_centroids)),
                o3d.geometry.PointCloud(points=o3d.utility.Vector3dVector(ref_centroids)),
                o3d.utility.Vector2iVector(correspondences),
                max_correspondence_distance=distance_thresh,
                estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
                ransac_n=ransac_n,
                criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(1000, 0.999)
            )

            T = result.transformation
            R, t = T[:3, :3], T[:3, 3]

            for corr in np.asarray(result.correspondence_set):
                tgt_idx = corr[0]
                inlier_counter[tgt_idx] += 1

            if len(result.correspondence_set) > best_inlier_count:
                best_inlier_count = len(result.correspondence_set)
                best_R, best_t = R, t

        except Exception:
            continue

    return best_R, best_t, inlier_counter.tolist(), total_iterations

if __name__ == "__main__":
    args = parse_args()

    print("Loading data...")
    ref_gdf, tgt_gdf = load_data(args.ref_path, args.tgt_path)

    print("Computing centroids...")
    ref_centroids = compute_centroids(ref_gdf)
    tgt_centroids = compute_centroids(tgt_gdf)

    print("Finding matches...")
    matches = nearest_neighbor_match(ref_centroids, tgt_centroids, args.nn_search_radius)
    print(f"Found {len(matches)} matches within {args.nn_search_radius}m")

    if not matches:
        print("No matches found - check your search radius or data")
        exit()

    print("Running Open3D RANSAC with inlier tracking...")
    R, t, inlier_counts, total_iters = ransac_with_inlier_tracking(
        ref_centroids, tgt_centroids, matches,
        n_iterations=args.ransac_iterations,
        distance_thresh=args.ransac_nn_radius,
        ransac_n=args.ransac_sample_n
    )

    print("Calculating difference metrics...")
    matched_indices = [m[0] for m in matches]
    ref_indices = [m[1] for m in matches]
    ref_gdf_matched = ref_gdf.iloc[ref_indices].copy()
    tgt_gdf_matched_original = tgt_gdf.iloc[matched_indices].copy()
    diffs = calculate_differences(ref_gdf_matched, tgt_gdf_matched_original, [(i, i) for i in range(len(matches))])

    diff_columns = [
        "centroid_diff_3D", "centroid_diff_X", "centroid_diff_Y", "centroid_diff_Z",
        "length_diff_3D", "angle_horizontal_diff", "angle_vertical_diff"
    ]
    for i, col in enumerate(diff_columns):
        tgt_gdf_matched_original[col] = [d[2 + i] for d in diffs]

    tgt_gdf_matched_original["inlier_count"] = [inlier_counts[i] for i in matched_indices]
    tgt_gdf_matched_original["ransac_iterations"] = total_iters
    tgt_gdf_matched_original["matching_ID"] = [str(uuid.uuid4())[:8] for _ in matches]
    tgt_gdf_matched_original["matched_file"] = os.path.basename(args.ref_path)
    tgt_gdf_matched_original["nn_search_radius"] = args.nn_search_radius

    print("Saving results with inlier tracking...")
    tgt_gdf_matched_original.to_file(args.output_path)
    print(f"Saved aligned data with inlier statistics to {args.output_path}")
