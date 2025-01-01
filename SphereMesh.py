'''
Translating the  `SphereMesh` library from matlab to python and adding adjustable sphere radii.

SECTIONS:
    . Importing Libraries
    . Main Functions
        . generateSphereMesh
        . refineMesh
    . Auxillary Functions
        . getTetrahedralMesh
        . getOctahedralMesh
        . getIcosahedralMesh
        . randomlyRotate
    . Sample Usage

Author: Fathya Salih
Date: 31-12-2024
'''

# ----------------------------------------
# Importing Libraries
# ----------------------------------------
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.tri import Triangulation

# ----------------------------------------
# Main Functions
# ----------------------------------------
def generate_sphere_mesh(generation, mesh_type, radius=1):
    '''Generates a sphere mesh by refining a platonic solid projected onto a sphere of radius 1
    Args:
        generation (int): number of refinements
        mesh_type (str): platonic solid to start with (tet, oct, ico)
        radius (float): radius of the sphere that points are projected onto (default: 1)
    Returns:
        P: 3 x N array of vertices
        tri: M x 3 array of vertices forming each face
    '''
    if mesh_type == 'tet':  # starts with tetrahedron
        P, tri = get_tetrahedral_mesh(radius=radius)
    elif mesh_type == 'oct':  # starts with octahedron
        P, tri = get_octahedral_mesh(radius=radius)
    elif mesh_type == 'ico':  # starts with icosahedron
        P, tri = get_icosahedral_mesh(radius=radius)
    else:
        print('Error: Unrecognized polyhedron type.')
        P = np.eye(3)
        tri = np.array([[1, 2, 3]])
    
    for n in range(1, generation + 1):
        P, tri = refine_mesh(P, tri, radius=radius)
    
    return P, tri

def refine_mesh(Pin, triin, radius=1):
    """
    Refines a 3D triangular mesh by adding midpoints of edges as new vertices.

    Parameters:
        Pin: Input points (3 x N array).
        triin: Input triangles (M x 3 array, each row contains indices of vertices).
        radius (float): radius of the sphere that points are projected onto (default: 1)
    Returns:
        P: Refined points.
        tri: Refined triangles.
    """
    N = Pin.shape[1]

    # Extract edges from triangles
    edges_1 = np.vstack((triin[:, [0, 1]], triin[:, [1, 2]], triin[:, [2, 0]]))
    edges_s1 = np.sort(edges_1, axis=1)

    edges_2 = np.vstack((triin[:, [1, 2]], triin[:, [2, 0]], triin[:, [0, 1]]))
    edges_s2 = np.sort(edges_2, axis=1)

    edges_3 = np.vstack((triin[:, [2, 0]], triin[:, [0, 1]], triin[:, [1, 2]]))
    edges_s3 = np.sort(edges_3, axis=1)

    # Find unique edges
    edges_unique = np.unique(edges_s1, axis=0)

    # Compute new vertices as midpoints of edges
    Pnew = Pin[:, edges_unique[:, 0]] + Pin[:, edges_unique[:, 1]]
    Pnew = Pnew / np.linalg.norm(Pnew, axis=0, keepdims=True)  # Normalize to unit sphere
    Pnew *= radius  # Scale to the specified sphere radius

    # Find indices of new vertices
    def find_indices(edges_s, edges_unique):
        _, idx = np.where((edges_s[:, None] == edges_unique).all(axis=2))
        return idx

    idx1 = find_indices(edges_s1, edges_unique)
    idx2 = find_indices(edges_s2, edges_unique)
    idx3 = find_indices(edges_s3, edges_unique)

    # Create new triangles
    tri = np.hstack((
        edges_1[:, [0]].reshape(-1, 1),
        N + idx1.reshape(-1, 1),
        N + idx3.reshape(-1, 1)
    ))
    sz = len(idx1) // 3
    tri = np.vstack((
        tri,
        N + np.hstack((idx1[:sz].reshape(-1, 1), idx2[:sz].reshape(-1, 1), idx3[:sz].reshape(-1, 1)))
    ))

    # Combine old and new vertices
    P = np.hstack((Pin, Pnew))

    return P, tri

def visualize_mesh(P, tri):
    """
    Visualizes a triangular mesh using Matplotlib.

    Parameters:
        P: Points array (3 x N).
        tri: Triangles array (M x 3).
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    # Transpose the points for easier indexing
    X, Y, Z = P[0, :], P[1, :], P[2, :]
    
    # Create a Triangulation object
    triang = Triangulation(X, Y, tri)
    
    # Plot the triangular surface
    ax.plot_trisurf(triang, Z, edgecolor='k', linewidth=0.5, alpha=0.8)
    
    # Adjust the view
    ax.set_box_aspect([1, 1, 1])  # Equal aspect ratio
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    plt.show()
# ----------------------------------------
# Auxillary Functions
# ----------------------------------------
def get_tetrahedral_mesh(radius=1):
    '''Initializes a tetrahedral mesh as as platonic solid projected onto a sphere of radius 1
    Args:
        radius (float): radius of the sphere that points are projected onto (default: 1)
    Returns:
        P: 3 x 4 array of vertices
        tri: 4 x 3 array of vertices forming each face
    '''
    # Define the vertices of a tetrahedron
    P = np.array([[1, 1, 1],
                  [1, -1, -1],
                  [-1, 1, -1],
                  [-1, -1, 1]]).T
    # Normalize vertices to unit sphere
    P = P / np.tile(np.sqrt(np.sum(P**2, axis=0)), (3, 1))
    # Scale vertices to the desired radius
    P *= radius
    # Center the vertices around the origin
    P = P - np.tile(np.mean(P, axis=1, keepdims=True), (1, P.shape[1]))
    # Apply random rotation
    P = randomly_rotate(P)
    # Index the vertices of each face
    tri = np.array([[1, 2, 3],
                    [4, 2, 1],
                    [1, 3, 4],
                    [4, 3, 2]]) - 1
    return P, tri

def get_octahedral_mesh(radius=1):
    '''Initializes an octahedral mesh as a platonic solid projected onto a sphere of radius 1
    Args:
        radius (float): radius of the sphere that points are projected onto (default: 1)
    Returns:
        P: 3 x 6 array of vertices
        tri: 8 x 3 array of vertices forming each face
    '''
    # Define the vertices of an octahedron
    P = np.array([
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
        [-1, 0, 0],
        [0, -1, 0],
        [0, 0, -1]
    ]).T
    # Normalize vertices to unit sphere
    P = P / np.sqrt(np.sum(P**2, axis=0))
    # Scale vertices to the desired radius
    P *= radius
    # Center the vertices around the origin
    P = P - np.mean(P, axis=1, keepdims=True)
    # Apply random rotation
    P = randomly_rotate(P)
    # Index the vertices of each face
    tri = np.array([
        [3, 1, 2],
        [3, 2, 4],
        [3, 4, 5],
        [3, 5, 1],
        [6, 2, 1],
        [6, 4, 2],
        [6, 5, 4],
        [6, 1, 5]
    ]) - 1
    
    return P, tri

def get_icosahedral_mesh(radius=1):
    '''Initializes an icosahedral mesh as a platonic solid projected onto a sphere of radius 1
    Args:
        radius (float): radius of the sphere that points are projected onto (default: 1)
    Returns:
        P: 3 x 12 array of vertices
        tri: 20 x 3 array of vertices forming each face
    '''
    # Define the vertices of an icosahedron
    phi = (1 + np.sqrt(5)) / 2
    P = np.array([
        [-1, phi, 0],
        [1, phi, 0],
        [-1, -phi, 0],
        [1, -phi, 0],
        [0, -1, phi],
        [0, 1, phi],
        [0, -1, -phi],
        [0, 1, -phi],
        [phi, 0, -1],
        [phi, 0, 1],
        [-phi, 0, -1],
        [-phi, 0, 1]
    ]).T
    # Normalize vertices to unit sphere
    P = P / np.sqrt(np.sum(P**2, axis=0))
    # Scale vertices to the desired radius
    P *= radius
    # Center the vertices around the origin
    P = P - np.mean(P, axis=1, keepdims=True)
    # Apply random rotation
    P = randomly_rotate(P)
    # Index the vertices of each face
    tri = np.array([
        [1, 12, 6],
        [1, 6, 2],
        [1, 2, 8],
        [1, 8, 11],
        [1, 11, 12],
        [2, 6, 10],
        [6, 12, 5],
        [12, 11, 3],
        [11, 8, 7],
        [8, 2, 9],
        [4, 10, 5],
        [4, 5, 3],
        [4, 3, 7],
        [4, 7, 9],
        [4, 9, 10],
        [5, 10, 6],
        [3, 5, 12],
        [7, 3, 11],
        [9, 7, 8],
        [10, 9, 2]
    ]) - 1
    
    return P, tri

def randomly_rotate(Pin):
    '''Randomly rotates the input points
    Args:
        Pin: 3 x N array of points
    Returns:
        Pout: 3 x N array of point
    '''
    _, _, R = np.linalg.svd(np.random.rand(3, 3))
    if np.linalg.det(R) < 0:
        R = R[:, [2, 1, 0]]
    Pout = R @ Pin
    return Pout

# ----------------------------------------
# Sample Usage
# ----------------------------------------

# # Generate a sphere mesh
# P, tri = generate_sphere_mesh(3, 'oct')

# # Visualize the mesh
# visualize_mesh(P, tri)
