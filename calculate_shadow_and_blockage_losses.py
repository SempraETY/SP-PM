import numpy as np
import pandas as pd

# Load the solar positions data using pandas
solar_positions_df = pd.read_csv('solar_positions.csv', delimiter=',')
solar_positions = solar_positions_df.values[:, 1:]  

heliostat_positions = np.load('heliostat_positions.npy')
optimized_surface_normals = np.load('optimized_surface_normals.npy')

# Define the size of the standard heliostat
heliostat_length = 6.0  
heliostat_width = 6.0  

# Define the vertices of the standard heliostat in its local coordinate system
standard_heliostat_vertices = np.array([
    [-heliostat_length / 2, -heliostat_width / 2, 0],
    [heliostat_length / 2, -heliostat_width / 2, 0],
    [heliostat_length / 2, heliostat_width / 2, 0],
    [-heliostat_length / 2, heliostat_width / 2, 0]
])

# Pre-calculate the center of the standard heliostat vertices
standard_heliostat_center = np.mean(standard_heliostat_vertices, axis=0)

def calculate_basis_vectors(position, normal):
    """
    Calculate the basis vectors of a coordinate system based on the position and normal vector.
    
    Parameters:
    position (np.array): The position vector.
    normal (np.array): The normal vector.
    
    Returns:
    np.array: The basis vectors of the coordinate system.
    """
    # Normalize the normal vector
    normal = normal / np.linalg.norm(normal)
    
    # Find two other perpendicular vectors
    # Here we find one arbitrary vector which is not parallel to the normal
    if normal[0] != 0 or normal[1] != 0:
        vec = np.array([-normal[1], normal[0], 0])
    else:
        vec = np.array([0, -normal[2], normal[1]])
    
    # We find another vector which is perpendicular to both the normal and the vector we found
    vec2 = np.cross(normal, vec)
    
    # Create a matrix with these three vectors as the columns
    basis_vectors = np.vstack((normal, vec, vec2)).T
    
    return basis_vectors

def calculate_transformation_matrix(matrix1, matrix2):
    """
    Calculate the transformation matrix from one coordinate system to another.
    
    Parameters:
    matrix1 (np.array): The basis vectors of the first coordinate system.
    matrix2 (np.array): The basis vectors of the second coordinate system.
    
    Returns:
    np.array: The transformation matrix.
    """
    return np.linalg.inv(matrix2) @ matrix1


def ray_plane_intersection(ray_origin, ray_direction, plane_point, plane_normal):
    """Find the intersection point of a ray with a plane."""
    t = np.dot(plane_normal, plane_point - ray_origin) / np.dot(plane_normal, ray_direction)
    intersection_point = ray_origin + t * ray_direction
    return intersection_point

def is_point_inside_rectangle(point, rectangle_vertices):
    """Check if a point is inside a rectangle defined by its vertices."""
    AB = rectangle_vertices[1] - rectangle_vertices[0]
    AM = point - rectangle_vertices[0]
    BC = rectangle_vertices[2] - rectangle_vertices[1]
    BM = point - rectangle_vertices[1]
    
    dot1 = np.dot(AB, AM)
    dot2 = np.dot(AB, AB)
    dot3 = np.dot(BC, BM)
    dot4 = np.dot(BC, BC)
    
    return 0 <= dot1 <= dot2 and 0 <= dot3 <= dot4

def calculate_shadow_and_blockage_losses():
    num_heliostats = len(heliostat_positions)
    num_time_points = solar_positions.shape[0]
    
    shadow_and_blockage_losses = np.zeros((num_time_points, num_heliostats))
    
    # Calculate the transformation matrices for all the heliostats
    transformation_matrices = np.array([calculate_transformation_matrix(pos, norm) 
                                        for pos, norm in zip(heliostat_positions, optimized_surface_normals)])
    
    for t in range(num_time_points):
        solar_direction = solar_positions[t, 1:]
        
        for i in range(num_heliostats):
            for j in range(i + 1, num_heliostats):
                transformation_matrix = calculate_transformation_matrix(transformation_matrices[i], transformation_matrices[j])
                
                ray_start_in_j = np.dot(transformation_matrix, np.append(standard_heliostat_center, 1))[:3]
                solar_direction_in_j = np.dot(transformation_matrix, np.append(solar_direction, 0))[:3]
                
                intersection_point = ray_plane_intersection(ray_start_in_j, solar_direction_in_j, standard_heliostat_vertices[0], optimized_surface_normals[j])
                
                if is_point_inside_rectangle(intersection_point, standard_heliostat_vertices):
                    shadow_and_blockage_losses[t, j] += 1

    np.save('shadow_and_blockage_losses.npy', shadow_and_blockage_losses)

# Call the function to calculate the shadow and blockage losses
calculate_shadow_and_blockage_losses()

