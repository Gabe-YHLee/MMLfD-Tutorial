
from plotly.subplots import make_subplots
import plotly.graph_objects as go

def visualize_SE3(T1, T2):
    bs = len(T1)
    fig = make_subplots(rows=bs, cols=2, specs=bs*[[{'type': 'surface'}, {'type': 'surface'}]])
    for b in range(bs):
        for j, T in enumerate([T1[b], T2[b]]):
            R_traj = T[:, :3, :3]
            p_traj = T[:, :3, 3]
            for i, (p, rotation_matrices) in enumerate(zip(p_traj, R_traj)):
                x_axis, y_axis, z_axis = 0.1*rotation_matrices[:, 0], 0.1*rotation_matrices[:, 1], 0.1*rotation_matrices[:, 2]

                # Create traces for the X, Y, and Z axes
                fig.add_trace(
                    go.Scatter3d(
                        x=[p[0], p[0]+x_axis[0]], y=[p[1], p[1]+x_axis[1]], z=[p[2], p[2]+x_axis[2]], 
                        line=dict(color='red', width=1), 
                        mode='lines'),
                    row= b+1, col=j+1)
                fig.add_trace(
                    go.Scatter3d(
                        x=[p[0], p[0]+y_axis[0]], y=[p[1], p[1]+y_axis[1]], z=[p[2], p[2]+y_axis[2]], 
                        line=dict(color='green', width=1), 
                        mode='lines'),
                    row= b+1, col=j+1)
                fig.add_trace(
                    go.Scatter3d(
                        x=[p[0], p[0]+z_axis[0]], y=[p[1], p[1]+z_axis[1]], z=[p[2], p[2]+z_axis[2]], 
                        line=dict(color='blue', width=1), 
                        mode='lines'),
                    row= b+1, col=j+1)
                fig.add_trace(
                    go.Scatter3d(
                        x=p, 
                        marker=dict(color='black', size=10), 
                        mode='markers'),
                    row= b+1, col=j+1)
                
            # Customize the layout
            fig.update_layout(
                title="SE3 Frame Visualization (Left: data, Right: recon)",
                scene=dict(
                    aspectmode="data",
                    aspectratio=dict(x=1, y=1, z=1)
                ),
                margin=dict(l=0, r=0, b=0, t=50))
    return fig