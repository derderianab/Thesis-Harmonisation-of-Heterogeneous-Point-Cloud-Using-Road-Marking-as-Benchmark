import numpy as np
import laspy
import argparse

def load_transformation_matrix(txt_path):
    with open(txt_path, "r") as f:
        lines = f.readlines()
    matrix_lines = lines[:4]
    matrix = np.array([[float(val) for val in line.strip().split()] for line in matrix_lines])
    return matrix

def transform_point_cloud(input_las_path, output_las_path, transformation_matrix):
    las = laspy.read(input_las_path)
    coords = np.vstack((las.x, las.y, las.z)).T

    # Apply transformation directly
    transformed_coords = (transformation_matrix[:3, :3] @ coords.T).T + transformation_matrix[:3, 3]

    las.x = transformed_coords[:, 0]
    las.y = transformed_coords[:, 1]
    las.z = transformed_coords[:, 2]
    las.write(output_las_path)
    print(f"Transformed point cloud saved to: {output_las_path}")

def parse_arguments():
    parser = argparse.ArgumentParser(description="Apply a transformation matrix to a LAS/LAZ point cloud")
    parser.add_argument('--transformation_txt', required=True, help='Path to transformation matrix text file')
    parser.add_argument('--input_las_path', required=True, help='Path to input LAS/LAZ file')
    parser.add_argument('--output_las_path', required=True, help='Path to output LAS/LAZ file')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    transformation_matrix = load_transformation_matrix(args.transformation_txt)
    transform_point_cloud(args.input_las_path, args.output_las_path, transformation_matrix)
