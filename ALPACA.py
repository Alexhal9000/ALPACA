import numpy as np
import open3d as o3d
from cpdalp import DeformableRegistration
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import copy
import time
import gc
import os
import pandas as pd
import json


# This is the ALPACA class that performs the alignment of the source landmarks to the target mesh. 
# This version was written by Alejandro Gutierrez and was based on the original ALPACA code by Arthur Porto. 
# The original code can be found at https://github.com/SlicerMorph/SlicerMorph/blob/master/ALPACA/ALPACA.py


class ALPACA:
    def __init__(self):
        # Default parameters based on ALPACA
        self.params = {
            "point_density": 0.5,
            "normal_search_radius": 1,
            "fpfh_search_radius": 2.5,
            "distance_threshold": 0.5,
            "max_ransac_iter": 10000000,
            "icp_distance_threshold": 0.1,
            "alpha": 2.0,  # CPD rigidity parameter
            "beta": 2.0,   # CPD motion coherence
            "cpd_iterations": 100,
            "cpd_tolerance": 0.0001
        }

    def create_landmarks_from_mesh(self, vertices, faces, n_landmarks=5000):
        """
        Creates evenly distributed landmarks on a mesh surface
        
        Args:
            vertices: np.array of shape (N, 3) containing vertex coordinates
            faces: np.array of shape (M, 3) containing face indices
            n_landmarks: Number of landmarks to generate
            
        Returns:
            np.array of shape (n_landmarks, 3) containing landmark coordinates
        """
        # Convert to Open3D mesh
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(vertices)
        mesh.triangles = o3d.utility.Vector3iVector(faces)
        
        # Compute mesh normals
        mesh.compute_vertex_normals()
        
        # Sample points uniformly
        pcd = mesh.sample_points_uniformly(number_of_points=n_landmarks)
        
        return np.asarray(pcd.points)

    def plot_debug_step(self, source_points, target_points, title):
        # Disable this function for now
        return

        """Helper function to plot point clouds at each step"""
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(source_points[:, 0], source_points[:, 1], source_points[:, 2], 
                  c='red', label='Source', s=1)
        ax.scatter(target_points[:, 0], target_points[:, 1], target_points[:, 2], 
                  c='blue', label='Target', s=1)
        ax.set_title(title)
        ax.legend()
        # Make all axes equal scale
        ax.set_box_aspect([1,1,1])
        plt.savefig(f'debug_{title}.jpg', dpi=300, bbox_inches='tight')
        plt.close(fig)

    def align_landmarks_to_mesh(self, source_landmarks, target_vertices, target_faces, target_metadata_path):

        # Find correspondences with mutual nearest neighbor and ratio test
        def find_robust_correspondences(source_fpfh, target_fpfh, ratio_threshold=0.8):
            corres = []
            for i in range(source_fpfh.shape[1]):
                # Compute distances to all target features
                distances = np.linalg.norm(source_fpfh[:, i:i+1] - target_fpfh, axis=0)
                sorted_idx = np.argsort(distances)
                
                # Ratio test
                if distances[sorted_idx[0]] < ratio_threshold * distances[sorted_idx[1]]:
                    # Mutual nearest neighbor check
                    back_distances = np.linalg.norm(target_fpfh[:, sorted_idx[0]:sorted_idx[0]+1] - source_fpfh, axis=0)
                    if np.argmin(back_distances) == i:
                        corres.append([i, sorted_idx[0]])
            
            return corres

        # Validate RANSAC result
        def validate_alignment(source_pcd, target_pcd, transformation, random_params):
            """Validates alignment quality using average distance and orientation consistency.
            Returns True if alignment seems good, False otherwise."""
            
            # Transform source points
            source_transformed = copy.deepcopy(source_pcd)
            source_transformed.transform(transformation)
            
            # Get point clouds as numpy arrays
            source_points = np.asarray(source_transformed.points)
            source_normals = np.asarray(source_transformed.normals)
            target_points = np.asarray(target_pcd.points)
            target_normals = np.asarray(target_pcd.normals)
            
            # Build KD-tree for target
            target_tree = o3d.geometry.KDTreeFlann(target_pcd)
            
            total_points = len(source_points)
            good_alignments = 0
            avg_distance = 0
            
            for i, (point, normal) in enumerate(zip(source_points, source_normals)):
                # Find nearest neighbor in target
                [_, idx, dist] = target_tree.search_knn_vector_3d(point, 1)
                avg_distance += np.sqrt(dist[0])
                
                # Check if normals are roughly aligned (dot product > 0.7 means angle < ~45 degrees)
                normal_alignment = abs(np.dot(normal, target_normals[idx[0]]))
                if normal_alignment > 0.7:
                    good_alignments += 1
            
            avg_distance /= total_points
            normal_consistency = good_alignments / total_points
            print("Validation of RANSAC results:")
            print(f"Average distance: {avg_distance:.4f}")
            print(f"Normal consistency: {normal_consistency:.4f}")

            random_params["average_distance"] = avg_distance
            random_params["normal_consistency"] = normal_consistency

            # Add the random params as a new line to the csv file if threshold is met
            if avg_distance < 0.01 and normal_consistency > 0.75:
                # Only write to CSV if params were randomly generated (not extracted)
                if not random_params.get('extracted_from_csv', False):
                    csv_file = "random_params.csv"
                    new_row_dict = {
                        "normal_search_radius": random_params['normal_search_radius'],
                        "max_nn_normals": random_params['max_nn_normals'],
                        "max_nn_fpfh": random_params['max_nn_fpfh'],
                        "distance_threshold": random_params['distance_threshold'],
                        "fpfh_scale_1": random_params['fpfh_scales'][0],
                        "fpfh_scale_2": random_params['fpfh_scales'][1],
                        "fpfh_scale_3": random_params['fpfh_scales'][2],
                        "fpfh_scale_4": random_params['fpfh_scales'][3],
                        "ransac_max_correspondence_distance": random_params['ransac_max_correspondence_distance'],
                        "ransac_n": random_params['ransac_n'],
                        "ransac_ratio_threshold": random_params['ransac_ratio_threshold'],
                        "normal_consistency": random_params['normal_consistency'],
                        "average_distance": random_params['average_distance']
                    }

                    # Create DataFrame with single row
                    new_row = pd.DataFrame([new_row_dict])
                    
                    try:
                        if os.path.exists(csv_file):
                            # Append without header
                            new_row.to_csv(csv_file, mode='a', header=False, index=False)
                        else:
                            # Create new file with header
                            new_row.to_csv(csv_file, mode='w', header=True, index=False)
                        print(f"Successfully wrote new parameters to {csv_file}")
                    except Exception as e:
                        print(f"Error writing to CSV: {str(e)}")

            # These thresholds might need adjustment based on your data scale
            return avg_distance < 0.01 and normal_consistency > 0.75


        # Use multiple random subsets for better coverage
        max_retries = 31
        attempt = 0
        valid_alignment = False
        
        while not valid_alignment and attempt < max_retries:

            # Phase 1: FPFH + RANSAC with low resolution
            gc.collect()

            # Set different random seeds for each attempt
            current_seed = int(time.time() * 1000) % (2**32 - 1)
            np.random.seed(current_seed)
            random_state = np.random.RandomState(current_seed)

            if attempt == 0 or attempt > 10:
                # Initial point clouds
                source_pcd = o3d.geometry.PointCloud()
                source_pcd.points = o3d.utility.Vector3dVector(source_landmarks)
                
                target_mesh = o3d.geometry.TriangleMesh()
                target_mesh.vertices = o3d.utility.Vector3dVector(target_vertices)
                target_mesh.triangles = o3d.utility.Vector3iVector(target_faces)
                target_pcd = target_mesh.sample_points_uniformly(number_of_points=int(len(source_landmarks) * 2))

                # Create initial point clouds at different resolutions
                source_pcd_low = o3d.geometry.PointCloud()
                source_pcd_med = o3d.geometry.PointCloud()
                source_pcd_full = o3d.geometry.PointCloud()
                
                # Sample source points at different resolutions
                n_points_low = len(source_landmarks) // 5
                n_points_med = len(source_landmarks) // 3
                source_indices_low = np.linspace(0, len(source_landmarks)-1, n_points_low, dtype=int)
                source_indices_med = np.linspace(0, len(source_landmarks)-1, n_points_med, dtype=int)
                
                source_pcd_low.points = o3d.utility.Vector3dVector(source_landmarks[source_indices_low])
                source_pcd_med.points = o3d.utility.Vector3dVector(source_landmarks[source_indices_med])
                source_pcd_full.points = o3d.utility.Vector3dVector(source_landmarks)
                
                # Create target point clouds at different resolutions
                target_pcd_low = target_mesh.sample_points_uniformly(number_of_points=n_points_low * 2)
                target_pcd_med = target_mesh.sample_points_uniformly(number_of_points=n_points_med * 2)
                target_pcd_full = target_mesh.sample_points_uniformly(number_of_points=len(source_landmarks) * 2)
                
                # Plot initial state
                self.plot_debug_step(np.asarray(source_pcd_low.points), 
                                    np.asarray(target_pcd_low.points), 
                                    "Initial Point Clouds")
                
                # Calculate centering and normalization from full resolution
                source_centroid = np.mean(source_landmarks, axis=0)
                target_centroid = np.mean(np.asarray(target_pcd.points), axis=0)
                
                # Center full resolution point clouds
                source_pcd_full.translate(-source_centroid)
                target_pcd_full.translate(-target_centroid)
                
                # Calculate scale factor from full resolution
                scale_factor = np.linalg.norm(np.asarray(source_pcd_full.points))
                center = np.zeros((3, 1))
                
                # Scale full resolution
                source_pcd_full.scale(1 / scale_factor, center)
                target_pcd_full.scale(1 / scale_factor, center)
                
                # Apply same transformations to medium and low resolution
                source_pcd_med.translate(-source_centroid)
                source_pcd_low.translate(-source_centroid)
                target_pcd_med.translate(-target_centroid)
                target_pcd_low.translate(-target_centroid)
                
                source_pcd_med.scale(1 / scale_factor, center)
                source_pcd_low.scale(1 / scale_factor, center)
                target_pcd_med.scale(1 / scale_factor, center)
                target_pcd_low.scale(1 / scale_factor, center)
                
                # Plot debug steps with low resolution (for visualization)
                self.plot_debug_step(np.asarray(source_pcd_low.points), 
                                np.asarray(target_pcd_low.points), 
                                "After Centering and Normalization")

            
            attempt += 1
            print(f"\nAttempting RANSAC alignment: try {attempt}/{max_retries} with seed {current_seed}")

            # Update parameters with randomized values
            random_params = {
                "normal_search_radius": 0.1 * (1 + np.random.uniform(-0.1, 0.1)),
                "max_nn_normals": np.random.randint(90, 110),
                "max_nn_fpfh": np.random.randint(180, 220),
                "distance_threshold": np.random.uniform(10, 20),
                "fpfh_scales": [max(0.05, s + j) for s, j in zip([0.1, 0.2, 0.3, 0.4], 
                                                                np.random.uniform(-0.02, 0.02, 4))],
                "ransac_max_correspondence_distance": np.random.uniform(0.27, 0.33),
                "ransac_n": 5,#np.random.randint(4, 5),
                "ransac_ratio_threshold": np.random.uniform(0.76, 0.83),
                "normal_consistency": 0.0,
                "average_distance": 0.0,
                "extracted_from_csv": False
            }


            # Compute normals with more neighbors for stability
            try:
                source_pcd_low.estimate_normals(
                    o3d.geometry.KDTreeSearchParamHybrid(
                        radius=random_params["normal_search_radius"], 
                        max_nn=random_params["max_nn_normals"]))
                target_pcd_low.estimate_normals(
                    o3d.geometry.KDTreeSearchParamHybrid(
                        radius=random_params["normal_search_radius"], 
                        max_nn=random_params["max_nn_normals"]))
            except Exception as e:
                print(f"Error estimating normals: {str(e)}")

            # Normal orientation without seed parameter
            source_pcd_low.orient_normals_consistent_tangent_plane(100)
            target_pcd_low.orient_normals_consistent_tangent_plane(100)

            # Compute multi-scale FPFH features
            source_fpfh_list = []
            target_fpfh_list = []
            
            for scale in random_params["fpfh_scales"]:
                try:
                    source_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
                        source_pcd_low, 
                        o3d.geometry.KDTreeSearchParamHybrid(
                            radius=scale, 
                            max_nn=random_params["max_nn_fpfh"]))
                    target_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
                        target_pcd_low, 
                        o3d.geometry.KDTreeSearchParamHybrid(
                            radius=scale, 
                            max_nn=random_params["max_nn_fpfh"]))
                    
                    source_fpfh_list.append(source_fpfh.data)
                    target_fpfh_list.append(target_fpfh.data)
                except Exception as e:
                    print(f"Error computing FPFH features at scale {scale}: {str(e)}")

            # Concatenate multi-scale features
            try:
                source_fpfh_combined = np.concatenate(source_fpfh_list, axis=0)
                target_fpfh_combined = np.concatenate(target_fpfh_list, axis=0)
            except Exception as e:
                print(f"Error concatenating FPFH features: {str(e)}")

            # Normalize combined features
            source_fpfh_norm = source_fpfh_combined / np.linalg.norm(source_fpfh_combined, axis=0, keepdims=True)
            target_fpfh_norm = target_fpfh_combined / np.linalg.norm(target_fpfh_combined, axis=0, keepdims=True)

            n_subsets = 3
            best_ransac_result = None
            best_fitness = -1

            for subset in range(n_subsets):
                # Add randomness to correspondence selection
                max_attempts_corres = 5  # Maximum number of attempts to find correspondences
                attempt_corres = 0
                corres = []
                
                while len(corres) == 0 and attempt_corres < max_attempts_corres:
                    ratio_threshold = np.random.uniform(0.75 - attempt_corres*0.05, 0.85)  # Gradually lower threshold
                    # store the ratio threshold
                    random_params["ransac_ratio_threshold"] = ratio_threshold
                    corres = find_robust_correspondences(source_fpfh_norm, target_fpfh_norm, ratio_threshold)
                    attempt_corres += 1
                
                if len(corres) == 0:
                    continue
                
                # Randomly sample with varying sizes
                if len(corres) > 1000:
                    sample_size = np.random.randint(800, 1000)
                    indices = random_state.choice(len(corres), sample_size, replace=False)
                    corres = [corres[i] for i in indices]
                
                corres_np = np.array(corres, dtype=np.int32)
                corres_vector = o3d.utility.Vector2iVector(corres_np)
                
                try:
                    current_ransac = o3d.pipelines.registration.registration_ransac_based_on_correspondence(
                        source_pcd_low, target_pcd_low, corres_vector,
                        max_correspondence_distance=random_params["ransac_max_correspondence_distance"],
                        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(with_scaling=False),
                        ransac_n=random_params["ransac_n"],
                        criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(
                            max_iteration=self.params["max_ransac_iter"],
                            confidence=0.99999
                        )
                    )
                    
                    if current_ransac.fitness > best_fitness:
                        best_fitness = current_ransac.fitness
                        best_ransac_result = current_ransac
                except Exception as e:
                    print(f"Error during RANSAC: {str(e)}")

            ransac_result = best_ransac_result
            # Validate the result
            valid_alignment = validate_alignment(source_pcd_low, target_pcd_low, ransac_result.transformation, random_params)
            
            if not valid_alignment and attempt < max_retries:
                print(f"Alignment attempt {attempt} failed validation, retrying...")
            elif not valid_alignment:
                print("Warning: Maximum retries reached. Proceeding with best available alignment.")
                # Update metadata to indicate alignment failure
                try:
                    metadata_path = target_metadata_path
                    with open(metadata_path, 'r') as f:
                        metadata = json.load(f)
                    metadata['alignment_error'] = True
                    with open(metadata_path, 'w') as f:
                        json.dump(metadata, f, indent=4)
                    print(f"Updated metadata to indicate alignment failure")
                except Exception as e:
                    print(f"Failed to update metadata: {str(e)}")
                break
            else:
                print(f"Successfully found valid alignment on attempt {attempt}")
                # Update metadata to indicate alignment success
                try:
                    metadata_path = target_metadata_path
                    with open(metadata_path, 'r') as f:
                        metadata = json.load(f)
                    metadata['alignment_error'] = False
                    with open(metadata_path, 'w') as f:
                        json.dump(metadata, f, indent=4)
                    print(f"Updated metadata to indicate alignment success")
                except Exception as e:
                    print(f"Failed to update metadata: {str(e)}")


        # Plot intermediate RANSAC result
        source_ransac = copy.deepcopy(source_pcd_low)
        source_ransac.transform(ransac_result.transformation)
        self.plot_debug_step(np.asarray(source_ransac.points), 
                           np.asarray(target_pcd_low.points), 
                           f"RANSAC Result")

        

         # Phase 2: Low resolution ICP refinement
        icp_low = o3d.pipelines.registration.registration_icp(
            source_pcd_low, target_pcd_low,
            max_correspondence_distance=self.params["icp_distance_threshold"] * 2,  # Larger threshold initially
            init=ransac_result.transformation,
            estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(with_scaling=False),
            criteria=o3d.pipelines.registration.ICPConvergenceCriteria(
                max_iteration=100,
                relative_fitness=1e-6,
                relative_rmse=1e-6
            )
        )

        # Plot intermediate ICP result
        source_icp_low = copy.deepcopy(source_pcd_low)
        source_icp_low.transform(icp_low.transformation)
        self.plot_debug_step(np.asarray(source_icp_low.points), 
                           np.asarray(target_pcd_low.points), 
                           f"ICP Result low res")
        
        # Phase 3: Medium resolution ICP refinement
        icp_med = o3d.pipelines.registration.registration_icp(
            source_pcd_med, target_pcd_med,
            max_correspondence_distance=self.params["icp_distance_threshold"],
            init=icp_low.transformation,
            estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(with_scaling=False),
            criteria=o3d.pipelines.registration.ICPConvergenceCriteria(
                max_iteration=50,
                relative_fitness=1e-6,
                relative_rmse=1e-6
            )
        )
        
        # Plot intermediate ICP result
        source_icp_med = copy.deepcopy(source_pcd_med)
        source_icp_med.transform(icp_med.transformation)
        self.plot_debug_step(np.asarray(source_icp_med.points), 
                           np.asarray(target_pcd_med.points), 
                           f"ICP Result med res")
        
        # Phase 4: Full resolution ICP for final refinement
        icp_result = o3d.pipelines.registration.registration_icp(
            source_pcd_full, target_pcd_full,
            max_correspondence_distance=self.params["icp_distance_threshold"],
            init=icp_med.transformation,
            estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(with_scaling=False),
            criteria=o3d.pipelines.registration.ICPConvergenceCriteria(
                max_iteration=50,  # Fewer iterations needed for final refinement
                relative_fitness=1e-8,
                relative_rmse=1e-8
            )
        )
        
        # Use the final transformation
        transformation_matrix = icp_result.transformation
        
        # Apply transformation to source for visualization
        source_transformed = copy.deepcopy(source_pcd_full)
        source_transformed.transform(icp_result.transformation)
        
        # Plot after RANSAC
        self.plot_debug_step(np.asarray(source_transformed.points), 
                               np.asarray(target_pcd_full.points), 
                               "After RANSAC and ICP")

        # Adjust transformation matrix for denormalization
        transformation_matrix = np.linalg.inv(icp_result.transformation)
        transformation_matrix = transformation_matrix.copy()  # Make a writeable copy
        transformation_matrix[:3, 3] *= scale_factor



        # Create homogeneous coordinates for target points 4D (x,y,z,1)
        target_pcd_full.scale(scale_factor, center)
        target_landmarks_homogeneous = np.hstack([target_pcd_full.points, np.ones((len(target_pcd_full.points), 1))])
        
        # Apply transformation
        target_landmarks_transformed = (transformation_matrix @ target_landmarks_homogeneous.T).T[:, :3]
        
        # Get centered source landmarks and add back the source centroid
        source_pcd_full.scale(scale_factor, center)
        source_landmarks_in_source_space = np.asarray(source_pcd_full.points)  # Convert to numpy array
        
        # Plot final result: the adjusted source landmarks and the transformed target landmarks
        self.plot_debug_step(source_landmarks_in_source_space, 
                           target_landmarks_transformed, 
                           "Final Result")

        return {
            'transformation_matrix': transformation_matrix,
            'source_centroid': source_centroid,
            'target_centroid': target_centroid
        }
