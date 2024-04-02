import numpy as np
from scipy.optimize import minimize_scalar

# Load the necessary data
new_reflection_paths = np.load('D:/2023_mathematical_modeling/A/A题/new_reflection_paths.npy')
heliostat_positions = np.load('D:/2023_mathematical_modeling/A/A题/heliostat_positions.npy')
receiver_pos = np.load('D:/2023_mathematical_modeling/A/A题/receiver_pos.npy')

# Define the receiver (cylindrical collector) geometry
receiver_height = 8.0  # in meters
receiver_diameter = 7.0  # in meters
receiver_radius = receiver_diameter / 2

# Define a function to find the intersection point of a ray and the cylindrical collector
def find_intersection_point(ray_origin, ray_direction, receiver_pos, receiver_radius, receiver_height):
    # Define the function to minimize
    def func(t):
        # Calculate the point on the ray at parameter t
        point_on_ray = ray_origin + t * ray_direction
        # Calculate the distance to the receiver axis
        dist_to_axis = np.linalg.norm(point_on_ray[:2] - receiver_pos[:2])
        # Return a penalty function value
        return (dist_to_axis - receiver_radius)**2 + (max(0, abs(point_on_ray[2] - receiver_pos[2]) - receiver_height / 2))**2
    
    # Perform the minimization
    result = minimize_scalar(func)
    
    # Calculate the intersection point
    intersection_point = ray_origin + result.x * ray_direction
    return intersection_point

# Find the intersection points for all heliostats and save them in a list
intersection_points = [find_intersection_point(heliostat_pos, reflection_path, receiver_pos, receiver_radius, receiver_height) 
                       for heliostat_pos, reflection_path in zip(heliostat_positions, new_reflection_paths)]

# Save the intersection points to a numpy file for easy access
np.save('intersection_points.npy', np.array(intersection_points))
