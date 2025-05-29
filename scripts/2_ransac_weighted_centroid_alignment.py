import numpy as np
import geopandas as gpd
from shapely.geometry import LineString
import argparse
import os

def estimate_weighted_rigid_transform(source_points, target_points, weights):
    assert source_points.shape == target_points.shape, "Point sets must have the same shape"
    assert source_points.shape[0] == weights.shape[0], "Weights must match number of points"

    weights = weights / np.sum(weights)
    src_centroid = np.sum(weights[:, np.newaxis] * source_points, axis=0)
    tgt_centroid = np.sum(weights[:, np.newaxis] * target_points, axis=0)
    src_centered = source_points - src_centroid
    tgt_centered = target_points - tgt_centroid
    H = (weights[:, np.newaxis] * src_centered).T @ tgt_centered
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[2, :] *= -1
        R = Vt.T @ U.T
    t = tgt_centroid - R @ src_centroid
    return R, t, src_centroid

def extract_line_centroids(gdf):
    return np.array([np.mean(np.array(geom.coords), axis=0) for geom in gdf.geometry])

def parse_arguments():
    parser = argparse.ArgumentParser(description="Apply weighted transformation to a set of line features.")
    parser.add_argument('--tgt_path', required=True, help="Path to the target GeoPackage with correspondence info")
    parser.add_argument('--output_gpkg', required=True, help="Path to save the transformed GeoPackage")
    parser.add_argument('--transform_txt', required=True, help="Path to save the transformation matrix text file")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()

    target_gdf = gpd.read_file(args.tgt_path)
    target_centroids = extract_line_centroids(target_gdf)
    centroid_diffs = np.stack([
        target_gdf["centroid_diff_X"].values,
        target_gdf["centroid_diff_Y"].values,
        target_gdf["centroid_diff_Z"].values
    ], axis=1)
    ref_centroids = target_centroids - centroid_diffs
    inlier_counts = target_gdf["inlier_count"].values
    ransac_iterations = target_gdf["ransac_iterations"].iloc[0]
    weights = inlier_counts / ransac_iterations

    R, t, mean_target_centroid = estimate_weighted_rigid_transform(target_centroids, ref_centroids, weights)
    transformed = (R @ target_centroids.T).T + t

    print("Final Rotation matrix (R):")
    print(R)
    print("\nFinal Translation vector (t):")
    print(t)

    transform_4x4 = np.eye(4)
    transform_4x4[:3, :3] = R
    transform_4x4[:3, 3] = t

    os.makedirs(os.path.dirname(args.transform_txt), exist_ok=True)
    with open(args.transform_txt, 'w') as f:
        for row in transform_4x4:
            f.write(' '.join(map(str, row)) + '\n')
        f.write(' '.join(map(str, mean_target_centroid)) + '\n')

    transformed_geometries = []
    for geom in target_gdf.geometry:
        coords = np.array(geom.coords)
        transformed_coords = (R @ coords.T).T + t
        transformed_geometries.append(LineString(transformed_coords))

    transformed_gdf = target_gdf.copy()
    transformed_gdf.geometry = transformed_geometries
    transformed_gdf.to_file(args.output_gpkg, driver='GPKG')

    rmse = np.sqrt(np.mean(np.sum((transformed - ref_centroids) ** 2, axis=1)))
    print(f"\nRMSE after alignment: {rmse:.4f} m")
