import plotly.graph_objects as go
import torch
import numpy as np

from model.mesh_fusion.util import torch_to_o3d_mesh


def generate_plotly_figure(
    vertices: torch.Tensor,
    faces: torch.Tensor,
    colors: torch.Tensor,
) -> go.Figure:
    mesh = torch_to_o3d_mesh(vertices, faces, colors)

    triangle_array = np.asarray(mesh.triangles)
    vertex_array = np.asarray(mesh.vertices)
    color_array = np.asarray(mesh.vertex_colors)

    plotly_figure = go.Figure(
        data=[
            go.Mesh3d(
                x=vertex_array[:,0],
                y=vertex_array[:,1],
                z=vertex_array[:,2],
                i=triangle_array[:,0],
                j=triangle_array[:,1],
                k=triangle_array[:,2],
                vertexcolor=color_array,
                opacity=1
            )
        ],
        layout=dict(
            scene=dict(
                xaxis=dict(visible=False),
                yaxis=dict(visible=False),
                zaxis=dict(visible=False),
                
            ),
            width=1000,
            height=1000,
        ),
    )

    return plotly_figure