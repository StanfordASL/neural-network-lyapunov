import plotly
import plotly.graph_objs as go
import torch
import numpy as np
import random
from robust_value_approx.test import slip_utils

ROBUST_COLOR = '#e32249'
BASELINE_COLOR = '#0f4c75'

ROBUST_BCK_COLOR = '#E3B7C0'
BASELINE_BCK_COLOR = '#93BBD6'

DELTA_COLOR = '#8cba51'
DELTA_BCK_COLOR = '#D4E8A2'
DELTA_ZERO_COLOR = '#586045'
DELTA_OVERUNDER_BCK_COLOR = '#A8B2E6'
DELTA_OVERUNDER_COLOR = '#5f6caf'

ROBUST_NAME = 'Sample-efficient'
BASELINE_NAME = 'Baseline'

FONT_SIZE = 18


def buffer_plot(state_log, x0_lo, x0_up, ix, iy,
                lim_eps=.1, cmax=1.):
    fig = plotly.subplots.make_subplots(rows=1, cols=2, subplot_titles=(
        ROBUST_NAME, BASELINE_NAME))
    fig.update_layout(showlegend=False)
    fig.update_xaxes(showgrid=False, zeroline=True,
                     range=[x0_lo[ix]-lim_eps, x0_up[ix]+lim_eps])
    fig.update_yaxes(showgrid=False, zeroline=True,
                     range=[x0_lo[iy]-lim_eps, x0_up[iy]+lim_eps])
    for state in state_log:
        fig.add_trace(go.Scatter(
            visible=False,
            x=state['adv_data_buffer'][:, ix],
            y=state['adv_data_buffer'][:, iy],
            mode='markers',
            marker=dict(
                size=7,
                color=state['robust_buffer_loss'],
                colorscale='Viridis',
                cmin=0.,
                cmax=cmax,
                symbol=0,
                showscale=False)), row=1, col=1)
        fig.add_trace(go.Scatter(
            visible=False,
            x=state['rand_data_buffer'][:, ix],
            y=state['rand_data_buffer'][:, iy],
            mode='markers',
            marker=dict(
                size=7,
                color=state['baseline_buffer_loss'],
                colorscale='Viridis',
                cmin=0.,
                cmax=cmax,
                showscale=False)), row=1, col=2)
    fig.data[0].visible = True
    fig.data[1].visible = True
    steps = []
    for i in range(int(len(fig.data)/2)):
        step = dict(
            method="restyle",
            args=["visible", [False] * len(fig.data)],
        )
        step["args"][1][2*i] = True
        step["args"][1][2*i+1] = True
        steps.append(step)
    sliders = [dict(
        active=0,
        currentvalue={"prefix": "Step: "},
        pad={"t": 50},
        steps=steps
    )]
    fig.update_layout(
        sliders=sliders
    )
    return fig


def bilevel_plot(state, adv, ix, iy, show_buffer=False):
    fig = go.Figure()
    colorscale = [[0.0, "rgb(165,0,38)"],
                  [0.1111111111111111, "rgb(215,48,39)"],
                  [0.2222222222222222, "rgb(244,109,67)"],
                  [0.3333333333333333, "rgb(253,174,97)"],
                  [0.4444444444444444, "rgb(254,224,144)"],
                  [0.5555555555555556, "rgb(224,243,248)"],
                  [0.6666666666666666, "rgb(171,217,233)"],
                  [0.7777777777777778, "rgb(116,173,209)"],
                  [0.8888888888888888, "rgb(69,117,180)"],
                  [1.0, "rgb(49,54,149)"]]
    z = state['robust_val_loss']
    # x = adv.x_samples_validation[:, ix].squeeze()
    # y = adv.x_samples_validation[:, iy].squeeze()
    # TODO get grid from actual samples, this is hardcoded for double int!!
    fig.add_trace(go.Surface(
        z=z.reshape(100, 100).t(),
        x=torch.linspace(-1, 1, 100),
        y=torch.linspace(-1, 1, 100),
        opacity=.7,
        showscale=False,
        colorscale=colorscale))
    if show_buffer:
        fig.add_trace(go.Scatter3d(
            x=state['adv_data_buffer'][:, ix],
            y=state['adv_data_buffer'][:, iy],
            z=state['robust_buffer_loss'],
            mode='markers',
            marker=dict(
                color=['#f6eec7']*state['robust_buffer_loss'].shape[0],
                size=2,
            )
        ))
    for (run, loss) in state['x_adv_opt_loss_log']:
        fig.add_trace(go.Scatter3d(
            x=run[:, ix],
            y=run[:, iy],
            z=loss,
            marker=dict(
                size=3,
                color=['#deff8b']*len(loss),
            ),
            line=dict(
                color='#deff8b',
                width=5,
                dash='dash',
            ),
        ))
        fig.add_trace(go.Scatter3d(
            x=[run[0, ix]],
            y=[run[0, iy]],
            z=[loss[0]],
            mode='markers',
            marker=dict(
                color=['#8cba51'],
                size=6,
            )
        ))
        fig.add_trace(go.Scatter3d(
            x=[run[-1, ix]],
            y=[run[-1, iy]],
            z=[loss[-1]],
            mode='markers',
            marker=dict(
                color="#313695",
                size=8,
            )
        ))
    fig.update_layout(showlegend=False)
    fig.update_layout(scene=dict(
        xaxis=dict(
            gridcolor="white",
            showbackground=True,
            showgrid=False,
            showticklabels=False,
            title="",
            zerolinecolor="white",),
        yaxis=dict(
            gridcolor="white",
            showbackground=True,
            showgrid=False,
            showticklabels=False,
            title="",
            zerolinecolor="white"),
        zaxis=dict(
            gridcolor="white",
            showgrid=False,
            showbackground=True,
            showticklabels=False,
            title="",
            zerolinecolor="white",),
    ),
        # margin=dict(
        # r=10, l=10,
        # b=10, t=10)
    )
    fig.update_layout(scene_aspectmode='manual',
                      scene_aspectratio=dict(x=1.5, y=1.5, z=1))
    return fig


def training_loss(state, window=100):
    raw_robust = state['robust_loss_log']
    raw_baseline = state['baseline_loss_log']
    assert(len(raw_robust) == len(raw_baseline))
    moving_average_robust = []
    moving_average_baseline = []
    moving_average_robust_std = []
    moving_average_baseline_std = []
    for i in range(1, len(raw_robust)):
        start = max(0, i-window)
        moving_average_robust.append(np.mean(raw_robust[start:i]))
        moving_average_baseline.append(np.mean(raw_baseline[start:i]))
        moving_average_robust_std.append(np.std(raw_robust[start:i]))
        moving_average_baseline_std.append(np.std(raw_baseline[start:i]))
    moving_average_robust = np.array(moving_average_robust)
    moving_average_baseline = np.array(moving_average_baseline)
    moving_average_robust_std = np.array(moving_average_robust_std)
    moving_average_baseline_std = np.array(moving_average_baseline_std)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        y=moving_average_robust-moving_average_robust_std,
        marker=dict(
            color=ROBUST_BCK_COLOR,
        ),
        line=dict(
            width=0
        ),
        mode='lines',
        showlegend=False))
    fig.add_trace(go.Scatter(
        y=moving_average_robust+moving_average_robust_std,
        marker=dict(
            color=ROBUST_BCK_COLOR,
        ),
        line=dict(
            width=0
        ),
        mode='lines',
        fill='tonexty',
        showlegend=False))
    fig.add_trace(go.Scatter(
        y=moving_average_baseline-moving_average_baseline_std,
        marker=dict(
            color=BASELINE_BCK_COLOR,
        ),
        line=dict(
            width=0
        ),
        mode='lines',
        showlegend=False))
    fig.add_trace(go.Scatter(
        y=moving_average_baseline+moving_average_baseline_std,
        marker=dict(
            color=BASELINE_BCK_COLOR,
        ),
        line=dict(
            width=0
        ),
        mode='lines',
        fill='tonexty',
        showlegend=False))
    fig.add_trace(go.Scatter(
        y=moving_average_robust,
        name=ROBUST_NAME,
        line=dict(
            width=6,
            color=ROBUST_COLOR,
        )
    ))
    fig.add_trace(go.Scatter(
        y=moving_average_baseline,
        name=BASELINE_NAME,
        line=dict(
            width=6,
            color=BASELINE_COLOR,
        )
    ))
    fig.update_layout(
        # title=dict(text="Trainning Batch Loss", xanchor='center', x=.5),
        xaxis_title="Training step",
        yaxis_title="MSE of sampled batch",
    )
    fig.update_layout(width=800, height=400)
    fig.update_layout(
        legend=dict(
            x=.76,
            y=.94,
            traceorder="normal",
        )
    )
    return fig


def validation_delta(states, window=100):
    assert(len(states) > 0)

    robust = []
    baseline = []
    for state in states:
        raw_robust = np.array(state["robust_val_loss_log"])
        raw_baseline = np.array(state["baseline_val_loss_log"])
        moving_average_robust = []
        moving_average_baseline = []
        for i in range(1, len(raw_robust)):
            start = max(0, i-window)
            moving_average_robust.append(np.mean(raw_robust[start:i]))
            moving_average_baseline.append(np.mean(raw_baseline[start:i]))
        robust.append(np.array(moving_average_robust))
        baseline.append(np.array(moving_average_baseline))

    num_samples = len(robust[0])
    losses_delta = np.zeros((len(states), num_samples))
    for i in range(len(states)):
        losses_delta[i, :] = (baseline[i] - robust[i]) / np.abs(baseline[i])
    losses_delta_mean = np.mean(losses_delta, axis=0)
    losses_delta_std = np.std(losses_delta, axis=0)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        y=(losses_delta_mean-losses_delta_std)*100.,
        marker=dict(
            color=DELTA_BCK_COLOR,
        ),
        line=dict(
            width=0
        ),
        mode='lines',
        showlegend=False))
    fig.add_trace(go.Scatter(
        y=(losses_delta_mean+losses_delta_std)*100.,
        marker=dict(
            color=DELTA_BCK_COLOR,
        ),
        line=dict(
            width=0
        ),
        mode='lines',
        fill='tonexty',
        showlegend=False))
    fig.add_trace(go.Scatter(
        y=(losses_delta_mean)*100.,
        line=dict(
            width=5,
            color=DELTA_COLOR,
        ),
        showlegend=True,
        name="Percent improvement of MSE",
    ))
    fig.update_layout(
        # title=dict(
        #     text="Improvement from Sample-Efficient Method on Test Set",
        #            xanchor='center', x=.5),
        # xaxis_title="Training step",
        yaxis_title="Percent decrease of MSE over test set",
    )
    fig.update_yaxes(zeroline=True, zerolinewidth=3,
                     zerolinecolor=DELTA_ZERO_COLOR)
    fig.update_layout(width=800, height=800)
    fig.update_layout(
        legend=dict(
            x=.06,
            y=.96,
            traceorder="normal",
        )
    )
    return fig


def validation_delta_overunder(states, window=100):
    assert(len(states) > 0)

    robust = []
    baseline = []
    for state in states:
        raw_robust = np.array(state["robust_val_loss_log"])
        raw_baseline = np.array(state["baseline_val_loss_log"])
        moving_average_robust = []
        moving_average_baseline = []
        for i in range(1, len(raw_robust)):
            start = max(0, i-window)
            moving_average_robust.append(np.mean(raw_robust[start:i]))
            moving_average_baseline.append(np.mean(raw_baseline[start:i]))
        robust.append(np.array(moving_average_robust))
        baseline.append(np.array(moving_average_baseline))

    num_samples = len(robust[0])
    losses_delta = np.zeros((len(states), num_samples))
    for i in range(len(states)):
        losses_delta[i, :] = (baseline[i] - robust[i]) / np.abs(baseline[i])
    losses_delta_mean = np.mean(losses_delta, axis=0)
    losses_delta_std = np.std(losses_delta, axis=0)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        y=(losses_delta_mean-losses_delta_std)*100.,
        marker=dict(
            color=DELTA_OVERUNDER_BCK_COLOR,
        ),
        line=dict(
            width=0
        ),
        mode='lines',
        showlegend=False))
    fig.add_trace(go.Scatter(
        y=(losses_delta_mean+losses_delta_std)*100.,
        marker=dict(
            color=DELTA_OVERUNDER_BCK_COLOR,
        ),
        line=dict(
            width=0
        ),
        mode='lines',
        fill='tonexty',
        showlegend=False))
    fig.add_trace(go.Scatter(
        y=(losses_delta_mean)*100.,
        line=dict(
            width=5,
            color=DELTA_OVERUNDER_COLOR,
        ),
        showlegend=True,
        name="Percent improvement of MSE",
    ))
    fig.update_layout(
        # title=dict(
        #     text="Improvement from Sample-Efficient Method on Test Set",
        #            xanchor='center', x=.5),
        # xaxis_title="Training step",
        yaxis_title="Percent decrease of MSE",
    )
    fig.update_yaxes(zeroline=True, zerolinewidth=3,
                     zerolinecolor=DELTA_ZERO_COLOR)
    fig.update_layout(width=800, height=400)
    fig.update_layout(
        legend=dict(
            x=.62,
            y=.16,
            traceorder="normal",
        )
    )
    return fig


def rollout_range(vf, x0_lo, x0_up, state_indices, names, n=10):
    fig = go.Figure()
    V = vf.get_value_function()
    x_traj_min = float('inf') * torch.ones(vf.sys.x_dim, vf.N).type(vf.dtype)
    x_traj_max = float('-inf') * torch.ones(vf.sys.x_dim, vf.N).type(vf.dtype)
    for i in range(n):
        x0 = torch.rand(vf.sys.x_dim, dtype=vf.dtype) * (x0_up - x0_lo) + x0_lo
        (x_traj, u_traj, _) = vf.sol_to_traj(x0, *V(x0)[1:])
        if x_traj is None:
            continue
        x_traj_min = torch.min(x_traj_min, x_traj)
        x_traj_max = torch.max(x_traj_max, x_traj)
    for x0 in [x0_lo, x0_up]:
        (x_traj, u_traj, _) = vf.sol_to_traj(x0, *V(x0)[1:])
        if x_traj is None:
            continue
        x_traj_min = torch.min(x_traj_min, x_traj)
        x_traj_max = torch.max(x_traj_max, x_traj)
    colors = [
        ("#%06x" % random.randint(0, 0xFFFFFF)) for i in range(len(names))]
    x0 = .5 * (x0_lo + x0_up)
    (x_traj, u_traj, _) = vf.sol_to_traj(x0, *V(x0)[1:])
    if x_traj is not None:
        x_traj_min = torch.min(x_traj_min, x_traj)
        x_traj_max = torch.max(x_traj_max, x_traj)
        for i in state_indices:
            fig.add_trace(go.Scatter(y=x_traj[i, :], name=names[i],
                                     marker=dict(color=[colors[i]]*vf.N),
                                     line=dict(color=colors[i])))
    (x_traj, u_traj, _) = vf.sol_to_traj(x0_lo, *V(x0_lo)[1:])
    if x_traj is not None:
        x_traj_min = torch.min(x_traj_min, x_traj)
        x_traj_max = torch.max(x_traj_max, x_traj)
        for i in state_indices:
            fig.add_trace(go.Scatter(y=x_traj[i, :], name=names[i],
                                     marker=dict(color=[colors[i]]*vf.N),
                                     line=dict(color=colors[i])))
    (x_traj, u_traj, _) = vf.sol_to_traj(x0_up, *V(x0_up)[1:])
    if x_traj is not None:
        x_traj_min = torch.min(x_traj_min, x_traj)
        x_traj_max = torch.max(x_traj_max, x_traj)
        for i in state_indices:
            fig.add_trace(go.Scatter(y=x_traj[i, :], name=names[i],
                                     marker=dict(color=[colors[i]]*vf.N),
                                     line=dict(color=colors[i])))
    for i in state_indices:
        fig.add_trace(go.Scatter(
            y=x_traj_min[i, :],
            marker=dict(
                color="#444",
            ),
            line=dict(
                width=0
            ),
            mode='lines',
            name=names[i],
            showlegend=False))
        fig.add_trace(go.Scatter(
            y=x_traj_max[i, :],
            marker=dict(
                color="#444"
            ),
            line=dict(
                width=0
            ),
            mode='lines',
            name=names[i],
            fill='tonexty',
            showlegend=False))
    return fig


def control_perf(cost_opt, cost_baseline, cost_robust,
                 nbin=100, bartop=None, clamp_val=10000):
    if bartop is None:
        bartop = cost_opt.shape[0]
    sub_opt_baseline_ = (cost_baseline - cost_opt).squeeze()
    sub_opt_robust_ = (cost_robust - cost_opt).squeeze()

    if clamp_val is not None:
        sub_opt_baseline = torch.clamp(sub_opt_baseline_, 0, clamp_val)
        sub_opt_robust = torch.clamp(sub_opt_robust_, 0, clamp_val)

    annotations = []
    annotations.append(dict(showarrow=False, x=torch.mean(sub_opt_baseline),
                            y=int(.65*bartop),
                            text=BASELINE_NAME+": mean",
                            xanchor="left",
                            xshift=4,
                            opacity=.95,
                            textangle=0,
                            font=dict(size=FONT_SIZE)))
    annotations.append(dict(showarrow=False,
                            x=torch.mean(sub_opt_robust),
                            y=int(.45*bartop),
                            text=ROBUST_NAME+": mean",
                            xanchor="left",
                            xshift=4,
                            opacity=.95,
                            textangle=0,
                            font=dict(size=FONT_SIZE)))
    if torch.max(sub_opt_baseline_) <= clamp_val:
        annotations.append(dict(showarrow=False,
                                x=torch.max(sub_opt_baseline),
                                y=int(.65*bartop),
                                text=BASELINE_NAME+": max",
                                xanchor="left",
                                xshift=4,
                                opacity=.95,
                                textangle=0.,
                                font=dict(size=FONT_SIZE)))
    else:
        annotations.append(dict(showarrow=False,
                                x=torch.max(sub_opt_baseline),
                                y=int(.65*bartop),
                                text=BASELINE_NAME +
                                ": max (" + str("%.1f" % torch.max(
                                    sub_opt_baseline_).item()) + ")",
                                xanchor="right",
                                xshift=-4,
                                opacity=.95,
                                textangle=0,
                                font=dict(size=FONT_SIZE)))
    if torch.max(sub_opt_robust_) <= clamp_val:
        annotations.append(dict(showarrow=False,
                                x=torch.max(sub_opt_robust),
                                y=int(.45*bartop),
                                text=ROBUST_NAME+": max",
                                xanchor="left",
                                xshift=4,
                                opacity=.95,
                                textangle=0,
                                font=dict(size=FONT_SIZE)))
    else:
        annotations.append(dict(showarrow=False,
                                x=torch.max(sub_opt_robust),
                                y=int(.45*bartop),
                                text=ROBUST_NAME +
                                ": max (" + str("%.1f" % torch.max(
                                    sub_opt_robust_).item()) + ")",
                                xanchor="right",
                                xshift=-4,
                                opacity=.95,
                                textangle=0,
                                font=dict(size=FONT_SIZE)))

    layout = go.Layout(annotations=annotations)

    fig = go.Figure(layout=layout)
    fig.update_layout(barmode='overlay')
    fig.add_trace(go.Histogram(x=sub_opt_baseline,
                               name=BASELINE_NAME,
                               nbinsx=nbin,
                               bingroup=1,
                               marker_color=BASELINE_COLOR))
    fig.add_trace(go.Histogram(x=sub_opt_robust,
                               name=ROBUST_NAME,
                               bingroup=1,
                               marker_color=ROBUST_COLOR))
    fig.update_traces(opacity=0.75)
    fig.add_shape(go.layout.Shape(type='line',
                                  xref='x',
                                  yref='y',
                                  x0=torch.mean(sub_opt_baseline),
                                  y0=0,
                                  x1=torch.mean(sub_opt_baseline),
                                  y1=bartop,
                                  line=dict(
                                      dash='dash',
                                      color='#d8c962'),
                                  opacity=1.))
    fig.add_shape(go.layout.Shape(type='line',
                                  xref='x',
                                  yref='y',
                                  x0=torch.mean(sub_opt_robust),
                                  y0=0,
                                  x1=torch.mean(sub_opt_robust),
                                  y1=bartop,
                                  line=dict(
                                      dash='dash',
                                      color='#d8c962'),
                                  opacity=1.))
    if torch.max(sub_opt_baseline_) <= clamp_val:
        fig.add_shape(go.layout.Shape(type='line',
                                      xref='x',
                                      yref='y',
                                      x0=torch.max(sub_opt_baseline),
                                      y0=0,
                                      x1=torch.max(sub_opt_baseline),
                                      y1=bartop,
                                      line=dict(
                                          dash='dash',
                                          color='#d8c962'),
                                      opacity=1.))
    if torch.max(sub_opt_robust_) <= clamp_val:
        fig.add_shape(go.layout.Shape(type='line',
                                      xref='x',
                                      yref='y',
                                      x0=torch.max(sub_opt_robust),
                                      y0=0,
                                      x1=torch.max(sub_opt_robust),
                                      y1=bartop,
                                      line=dict(
                                          dash='dash',
                                          color='#d8c962'),
                                      opacity=1.))
    fig.update_layout(width=800, height=600)
    fig.update_layout(
        legend=dict(
            x=.65,
            y=.96,
            traceorder="normal",
            font=dict(
                size=18)
        )
    )
    fig.update_layout(
        # title=dict(text="Trainning Batch Loss", xanchor='center', x=.5),
        yaxis_title="Number of rollouts",
        xaxis_title="Suboptimality"
    )
    return fig


def slip_traj(slip, x_traj, u_traj, xf):
    x0 = x_traj[:, 0]
    x_traj_nonlinear, x_traj_apex_nonlinear = slip_utils.sim_slip(
        slip, x0, u_traj)
    fig = go.Figure()
    x_goal = np.linspace(torch.min(x_traj_apex_nonlinear)-2, torch.max(
        x_traj_apex_nonlinear)+2, 35)
    fig.add_trace(go.Scatter(
        x=x_goal,
        y=[xf[1]]*len(x_goal),
        mode='markers',
        name='Goal',
        marker=dict(
            size=20,
            color=["#bbcfff"]*len(x_goal),
            # color=["#484848"]*len(x_goal),
            symbol=4,
            opacity=.5,
            line=dict(width=0,
                      color=['#35495e']*len(x_goal))),
    ))
    for i in range(u_traj.shape[1]):
        x_com = x_traj[:, i]
        x_foot = x_com[0] + slip.l0 * np.sin(u_traj[0, i])
        y_foot = x_com[1] - slip.l0 * np.cos(u_traj[0, i])
        fig.add_trace(go.Scatter(
            x=[x_com[0], x_foot],
            y=[x_com[1], y_foot],
            showlegend=False,
            line=dict(
                width=6,
                dash='dashdot',
                color="#484848"),
        ))
        fig.add_trace(go.Scatter(
            x=[x_foot],
            y=[y_foot],
            showlegend=False,
            marker=dict(
                size=15,
                color=["#484848"],
                symbol=0,
                opacity=1.,)
        ))
    fig.add_trace(go.Scatter(
        x=x_traj[0, :],
        y=x_traj[1, :],
        mode='markers',
        name='Piecewise Affine SLIP',
        marker=dict(
            size=40,
            color=["#c9485b"]*x_traj.shape[1],
            line=dict(width=3, color='#35495e')),
    ))
    fig.add_trace(go.Scatter(
        x=x_traj_nonlinear[0, :],
        y=x_traj_nonlinear[1, :],
        showlegend=False,
        mode='markers',
        marker=dict(
            size=7,
            color=["#96d1c7"]*x_traj_nonlinear.shape[1],
            line=dict(
                color="#484848",
                width=0,
            ))
    ))
    fig.add_trace(go.Scatter(
        x=x_traj_apex_nonlinear[0, :],
        y=x_traj_apex_nonlinear[1, :],
        name='Nonlinear SLIP',
        mode='markers',
        marker=dict(
            size=20,
            color=["#96d1c7"]*x_traj_apex_nonlinear.shape[1],
            line=dict(
                color="#484848",
                width=1,
            )),
    ))

    fig.update_yaxes(range=[0, 1.5])
    fig.update_xaxes(
        range=[torch.min(x_traj_apex_nonlinear)-1.,
               torch.max(x_traj_apex_nonlinear)+1.])
    fig.update_yaxes(showgrid=False, zeroline=True, zerolinewidth=5,
                     zerolinecolor="#484848")
    fig.update_xaxes(showgrid=False, zeroline=False)
    fig.update_layout(plot_bgcolor="#f0efef")
    fig.update_layout(
        legend=dict(
            x=.68,
            y=.08,
            traceorder="normal",
        ),
        font=dict(
            size=18)
    )
    fig.update_layout(
        # title=dict(text="Trainning Batch Loss", xanchor='center', x=.5),
        yaxis_title="y position (m)",
        xaxis_title="x position (m)"
    )

    return fig
