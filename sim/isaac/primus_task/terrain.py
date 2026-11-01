"""Deterministic mixed course [m], shared by collision mesh and height observations.

The robot's forward axis is local +Y. Grid edges bevel stair risers over 5 cm.
The maximum stair riser is 2.5 cm; ramp grade is 8%; roughness is +/-12 mm.
"""
import numpy as np

X_MIN, Y_MIN, SPACING = -1.5, -2.0, 0.05
SPAWN_Y = (-0.6, 0.7, 3.7, 6.7)


def course_grid():
    x = np.linspace(X_MIN, 1.5, 61)
    y = np.linspace(Y_MIN, 12.0, 281)
    xx, yy = np.meshgrid(x, y, indexing='ij')
    ramp = np.clip(np.minimum((yy-1.0)*0.08, (3.0-yy)*0.08), 0, 0.08)
    # Three steps up, a landing, and three steps down.
    up = np.floor((yy-4.0+1e-7)/0.3)+1
    down = np.floor((6.1-yy+1e-7)/0.3)+1
    stairs = 0.025*np.clip(np.minimum(up, down), 0, 3)
    envelope = np.clip(np.minimum((yy-7)*4, (10-yy)*4), 0, 1)
    rough = envelope*(0.007*np.sin(13*yy+3*xx)+0.005*np.sin(21*xx-7*yy))
    z = ramp+stairs+rough
    return x, y, z


def course_mesh():
    x, y, z = course_grid()
    xx, yy = np.meshgrid(x, y, indexing='ij')
    vertices = np.stack((xx, yy, z), axis=-1).reshape(-1, 3).astype(np.float32)
    a = np.arange((len(x)-1)*len(y)).reshape(len(x)-1, len(y))[:, :-1].ravel()
    b = a+len(y)
    faces = np.concatenate((np.stack((a,b,b+1),-1),np.stack((a,b+1,a+1),-1)))
    return vertices, faces.astype(np.int32)


def sample_height(x, y, grid):
    """Sample the exact triangular collision surface with torch tensors [m]."""
    import torch
    u = ((x-X_MIN)/SPACING).clamp(0, grid.shape[0]-1.00001)
    v = ((y-Y_MIN)/SPACING).clamp(0, grid.shape[1]-1.00001)
    i, j = u.long(), v.long()
    du, dv = u-i, v-j
    a,b,c,d = grid[i,j],grid[i+1,j],grid[i+1,j+1],grid[i,j+1]
    return torch.where(du>=dv, a+du*(b-a)+dv*(c-b), a+dv*(d-a)+du*(c-d))
