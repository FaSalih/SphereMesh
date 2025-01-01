'''
Generate mesh figures for varying number of generations and sphere sizes for a fixed mesh type
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
gens = np.array([0, 2, 4])
radii = np.array([0.5, 1, 3])

mesh_codes = ['tet', 'oct', 'ico']
mesh_names = ['Tetrahedral', 'Octahedral', 'Icosahedral']
mesh_code = mesh_codes[0]
mesh_name = mesh_names[0]

# intialize figure
fig = plt.figure(figsize=(10, 10))

# Generate and visualize sphere meshes
for j, radius in enumerate(radii):
    for i, gen in enumerate(gens):
        P, tri = sm.generate_sphere_mesh(gen, mesh_code , radius=radius)
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
        ax.set_title(f'{mesh_name}\n(Generation {gen}, Radius {radius})', fontsize=14, fontweight='bold')
        ax.set_xlim(-1*radii.max(), radii.max())
        ax.set_ylim(-1*radii.max(), radii.max())
        ax.set_zlim(-1*radii.max(), radii.max())

plt.tight_layout()
plt.subplots_adjust(hspace=0.4)
plt.savefig('python_meshes_scaled.png', dpi=300)
plt.show()
