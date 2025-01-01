'''
Example script to demonstrate how to generate and visualize a sphere mesh.
'''
# ----------------------------------------
# Importing Libraries
# ----------------------------------------
import SphereMesh as sm

# ----------------------------------------
# Main script
# ----------------------------------------
# Generate a sphere mesh
P, tri = sm.generate_sphere_mesh(3, 'oct')

# Visualize the mesh
sm.visualize_mesh(P, tri)