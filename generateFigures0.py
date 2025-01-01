'''
Generate the mesh figures published in the original repository
'''
# ----------------------------------------
# Importing Libraries
# ----------------------------------------
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.tri import Triangulation
import SphereMesh as sm

# ----------------------------------------
# Main script
# ----------------------------------------
# Define mesh generation parameters
gens = [0, 2, 4]
mesh_codes = ['tet', 'oct', 'ico']
mesh_names = ['Tetrahedral', 'Octahedral', 'Icosahedral']

# intialize figure
fig = plt.figure(figsize=(10, 10))

# Generate and visualize sphere meshes
for j, mesh_code in enumerate(mesh_codes):
    mesh_name = mesh_names[j]
    for i, gen in enumerate(gens):
        P, tri = sm.generate_sphere_mesh(gen, mesh_code)
        # Transpose the points for easier indexing
        X, Y, Z = P[0, :], P[1, :], P[2, :]
        # Create a Triangulation object
        triang = Triangulation(X, Y, tri)
        # Plot the triangular surface
        ax = fig.add_subplot(3, 3, 3*j+i+1, projection='3d')
        ax.plot_trisurf(triang, Z, edgecolor='k', linewidth=0.5, alpha=0.8, cmap='viridis') 
        # Adjust the view
        ax.set_box_aspect([1, 1, 1])  # Equal aspect ratio
        # Add attributes
        ax.set_title(f'{mesh_name} (Generation {gen})', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.subplots_adjust(hspace=0.4)
plt.savefig('python_meshes.png', dpi=300)
plt.show()
