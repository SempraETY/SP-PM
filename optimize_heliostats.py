import numpy as np
from scipy.optimize import minimize
import pandas as pd

# Load the variables from the file
with np.load('D:/2023_mathematical_modeling/A/A题/optimization_variables.npz') as data:
    sun_vector = data['sun_vector']
    receiver_pos = data['receiver_pos']
    initial_paths = data['initial_paths']
    ideal_paths = data['ideal_paths']
    surface_normals_initial_guess = data['surface_normals_initial_guess']
# 这里是我们之前定义的所有函数和变量
def calculate_sun_vector(azimuth_deg, elevation_deg):
    azimuth_rad = np.radians(azimuth_deg)
    elevation_rad = np.radians(elevation_deg)
    x = np.cos(elevation_rad) * np.cos(azimuth_rad)
    y = np.cos(elevation_rad) * np.sin(azimuth_rad)
    z = np.sin(elevation_rad)
    return np.array([x, y, z])

def calculate_receiver_vector(heliostat_pos, receiver_pos):
    return receiver_pos - heliostat_pos

def calculate_reflection_vector(incident_vector, normal_vector):
    reflection_vector = incident_vector - 2 * np.dot(incident_vector, normal_vector) * normal_vector
    return reflection_vector

def calculate_ideal_path(heliostat_pos, receiver_pos):
    return receiver_pos - heliostat_pos

# 请用实际值替换以下变量
tower_height = 0 # 请替换为实际值
heliostat_installation_height = 0 # 请替换为实际值
# 在加载其他变量之后添加以下行来设置num_heliostats变量
num_heliostats = len(initial_paths)
 # 请替换为实际值
sun_vector = np.array([0, 0, 0]) # 请替换为实际值
receiver_pos = np.array([0, 0, 0]) # 请替换为实际值
initial_paths = np.array([[0, 0, 0]]) # 请替换为实际值
ideal_paths = np.array([[0, 0, 0]]) # 请替换为实际值

# 优化目标函数
def objective_function(surface_normals):
    surface_normals = surface_normals.reshape((num_heliostats, 3))
    reflection_paths = np.array([calculate_reflection_vector(ip, sn) for ip, sn in zip(initial_paths, surface_normals)])
    
    # Calculate the norms of the vectors and add a small epsilon to avoid division by zero
    reflection_norms = np.linalg.norm(reflection_paths, axis=1) + 1e-8
    ideal_norms = np.linalg.norm(ideal_paths, axis=1) + 1e-8
    
    # Calculate the dot product and divide by the product of the norms
    dot_product = np.einsum('ij,ij->i', reflection_paths, ideal_paths) 
    cosine_theta = dot_product / (reflection_norms * ideal_norms)
    
    # Clamp the values to the range [-1, 1] to avoid errors in the arccos function
    cosine_theta = np.clip(cosine_theta, -1, 1)
    
    # Calculate the angles in degrees
    angles_between_paths = np.degrees(np.arccos(cosine_theta))
    
    return np.sum(angles_between_paths)


# 请用实际值替换以下变量
initial_guess = np.array([0, 0, 0]) # 请替换为实际值

# 执行优化
result = minimize(objective_function, surface_normals_initial_guess.flatten(), method='SLSQP')


# 输出优化结果
print("Optimization was successful:", result.success)
print("Final sum of angles:", result.fun)

# 提取优化后的表面法线向量
optimized_surface_normals = result.x.reshape((num_heliostats, 3))

# 保存到文件或进一步分析
np.save('optimized_surface_normals1.npy', optimized_surface_normals)

