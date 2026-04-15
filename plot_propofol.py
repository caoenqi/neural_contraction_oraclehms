# %%

import jax
import jax.numpy as jnp
import immrax as irx
import equinox as eqx
import numpy as onp
import diffrax
import matplotlib.pyplot as plt

from matplotlib.animation import FuncAnimation, FFMpegWriter
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from ncm_trainer import NCMTrainer
from propofol import (
    HRPropofol,
    ncm,
    control,
    cur_params,
    x_eq,
    inputs,
    BASE,
    NCM,
    CONTROLLER,
)

plt.rcParams.update({"text.usetex": True, "font.family": "Helvetica", "font.size": 14})

sys = HRPropofol(cur_params)

# Define function for control input
def HRPInput(t, x):
    idx = jnp.minimum(jnp.floor(t/0.01).astype(int), inputs.shape[0]-1)
    return inputs[idx, :]

# %%

# --------------------------------------------------------------------------- #
# Trained models                                                               #
# --------------------------------------------------------------------------- #

ncm_net = irx.NeuralNetwork(NCM, load=True)
control_net = irx.NeuralNetwork(CONTROLLER, load=True)
ix = irx.ut2i(jnp.load(CONTROLLER / "ix_ut.npy"))

print(ix)

# --------------------------------------------------------------------------- #
# Verify trainer loss <= 0 on loaded networks and ix                          #
# --------------------------------------------------------------------------- #

# trainer = NCMTrainer(
#     sys=sys,
#     ncm_fn=ncm,
#     control_fn=control,
#     a=0.01,
#     b=100.0,
#     c=0.001,
#     partition_indices=[6, 7, 8],
# )
# _loss_val, (_a_term, _b_term, _c_term) = trainer.loss((ncm_net, control_net), ix, 9)
# print(
#     f"Trainer loss: {_loss_val:.6f}  (a={_a_term:.4f}, b={_b_term:.4f}, c={_c_term:.4f})"
# )
# assert float(_loss_val) <= 0, (
#     f"Trainer loss {float(_loss_val)} > 0: networks do not satisfy contraction conditions on ix"
# )

# %%

# --------------------------------------------------------------------------- #
# Simulation parameters                                                        #
# --------------------------------------------------------------------------- #

t0 = 0.0
tf = 140.0
dt = 0.01
max_steps = int((tf - t0) / dt)
mc_N = 100
mc_method = diffrax.Dopri8()

# %%

# --------------------------------------------------------------------------- #
# Generate base trajectory from inputs                                        #
# --------------------------------------------------------------------------- #
x_ref_sol = sys.compute_trajectory(
    t0,
    tf,
    x_eq,
    (
        HRPInput,
    ),
    dt=dt,
    solver='euler',
    max_steps=max_steps
)
x_ref = x_ref_sol.ys

def HRP_xref(t):
    idx = jnp.minimum(jnp.floor(t/0.01).astype(int), x_ref.shape[0]-1)
    return x_ref[idx, :]

# %%
# --------------------------------------------------------------------------- #
# Drone arm geometry                                                           #
# --------------------------------------------------------------------------- #

_ARM_LEN = 0.4
_ARM_COLOR = "tab:blue"
_OUT_COLOR = "tab:red"
_c = _ARM_LEN / onp.sqrt(2)
_ARMS_BODY = onp.array(
    [
        [[-_c, -_c, 0], [_c, _c, 0]],
        [[-_c, _c, 0], [_c, -_c, 0]],
    ]
)
def _drone_arms_world(pos, phi, theta, psi):
    cphi, sphi = onp.cos(phi), onp.sin(phi)
    cth, sth = onp.cos(theta), onp.sin(theta)
    cpsi, spsi = onp.cos(psi), onp.sin(psi)
    Rx = onp.array([[1, 0, 0], [0, cphi, -sphi], [0, sphi, cphi]])
    Ry = onp.array([[cth, 0, sth], [0, 1, 0], [-sth, 0, cth]])
    Rz = onp.array([[cpsi, -spsi, 0], [spsi, cpsi, 0], [0, 0, 1]])
    return onp.einsum("ij,klj->kli", Rz @ Ry @ Rx, _ARMS_BODY) + pos  # (2, 2, 3)
# --------------------------------------------------------------------------- #
# Plot functions                                                               #
# --------------------------------------------------------------------------- #


def plot_static(mc_trajs, x_ref_traj, mc_N):
    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111, projection="3d")

    for i in range(0, mc_N, 10):
        ax.plot(
            onp.array(mc_trajs.ys[i, :, 0]),
            onp.array(mc_trajs.ys[i, :, 1]),
            onp.array(mc_trajs.ys[i, :, 2]),
            color="gray",
            alpha=1.0,
            lw=0.8,
        )
        pos = onp.array(mc_trajs.ys[i, 0, :3])
        arms = _drone_arms_world(
            pos, mc_trajs.ys[i, 0, 7], mc_trajs.ys[i, 0, 8], mc_trajs.ys[i, 0, 9]
        )
        ax.plot(
            arms[0, :, 0], arms[0, :, 1], arms[0, :, 2], color="gray", lw=1.5, alpha=1.0
        )
        ax.plot(
            arms[1, :, 0], arms[1, :, 1], arms[1, :, 2], color="gray", lw=1.5, alpha=1.0
        )

    ax.plot(
        x_ref_traj[:, 0],
        x_ref_traj[:, 1],
        x_ref_traj[:, 2],
        color="red",
        alpha=0.2,
        label="Reference",
    )
    ax.set_xlabel("$p_x$")
    ax.set_ylabel("$p_y$")
    ax.set_zlabel("$p_z$")
    fig.tight_layout()
    return fig

def plot_states_time(mc_trajs, ts, x_ref_traj, ix, mc_N):
    state_names = [
        "$Va$",
        "$Vv$",
        "$rF$",
        "$sR$",
        "$Q$",
        "$sVU$",
        "$m1$",
        "$m2$",
        "$m3$",
        "$Ce$",
    ]
    fig, axs = plt.subplots(2, 5, figsize=(18, 6), sharex=True)
    for j, (ax, name) in enumerate(zip(axs.flat, state_names)):
        for i in range(mc_N):
            ax.plot(ts, mc_trajs.ys[i, :, j], color="gray", alpha=0.3, lw=0.6)
        ax.plot(ts, x_ref_traj[:, j], color="red", lw=1.2, label="ref")
        irx.utils.plot_interval_t(
            ax, ts, ix[j] * jnp.ones_like(ts), color="steelblue", label="ix"
        )
        ax.set_title(name)
        ax.set_xlabel("$t$")
    fig.tight_layout()
    return fig


def plot_controls_time(mc_controls, ts, u_ref_traj, mc_N):
    control_names = [r"$JI$", r"$JP$"]
    fig, axs = plt.subplots(1, 2, figsize=(9, 4), sharex=True)
    for j, (ax, name) in enumerate(zip(axs.flat, control_names)):
        for i in range(mc_N):
            ax.plot(ts, mc_controls[i, :, j], color="gray", alpha=0.3, lw=0.6)
        ax.plot(ts, u_ref_traj[:, j], color="red", lw=1.2, label="ref")
        ax.set_title(name)
        ax.set_xlabel("$t$")
    fig.tight_layout()
    return fig


def save_video(mc_trajs, ts, x_ref_traj, ix, mc_N, out_path):
    trail_len = 100
    frame_stride = 5
    n_frames = len(ts)

    fig_vid = plt.figure(figsize=(10, 8))
    ax_vid = fig_vid.add_subplot(111, projection="3d")

    ax_vid.plot(
        x_ref_traj[:, 0],
        x_ref_traj[:, 1],
        x_ref_traj[:, 2],
        color="red",
        linewidth=1.5,
        label="Reference",
        zorder=5,
    )

    all_states = onp.array(mc_trajs.ys)  # (mc_N, T, xlen)
    xs = all_states[:, :, 0]
    ys_dat = all_states[:, :, 1]
    zs = all_states[:, :, 2]
    phis = all_states[:, :, 7]
    thetas = all_states[:, :, 8]
    psis = all_states[:, :, 9]
    ix_lo = onp.array(ix.lower)
    ix_hi = onp.array(ix.upper)
    in_ix_mask = onp.all(
        (all_states >= ix_lo) & (all_states <= ix_hi), axis=2
    )  # (mc_N, T)

    pad = 0.5
    ax_vid.set_xlim(xs.min() - pad, xs.max() + pad)
    ax_vid.set_ylim(ys_dat.min() - pad, ys_dat.max() + pad)
    ax_vid.set_zlim(zs.min() - pad, zs.max() + pad)
    ax_vid.set_xlabel("$p_x$ (m)")
    ax_vid.set_ylabel("$p_y$ (m)")
    ax_vid.set_zlabel("$p_z$ (m)")
    ax_vid.legend(loc="upper right")
    fig_vid.tight_layout()

    time_text = ax_vid.text2D(0.02, 0.95, "", transform=ax_vid.transAxes)
    trails_in = [
        ax_vid.plot([], [], [], color=_ARM_COLOR, alpha=0.3, lw=0.8)[0]
        for _ in range(mc_N)
    ]
    trails_out = [
        ax_vid.plot([], [], [], color=_OUT_COLOR, alpha=0.3, lw=0.8)[0]
        for _ in range(mc_N)
    ]
    drone_arm1 = [
        ax_vid.plot([], [], [], color=_ARM_COLOR, lw=2.0)[0] for _ in range(mc_N)
    ]
    drone_arm2 = [
        ax_vid.plot([], [], [], color=_ARM_COLOR, lw=2.0)[0] for _ in range(mc_N)
    ]
    all_lines = trails_in + trails_out + drone_arm1 + drone_arm2

    def init_vid():
        for line in all_lines:
            line.set_data([], [])
            line.set_3d_properties([])
        time_text.set_text("")
        return all_lines + [time_text]

    def update_vid(frame):
        i_start = max(0, frame - trail_len)
        for i in range(mc_N):
            sl = slice(i_start, frame + 1)
            mask = in_ix_mask[i, sl]
            tx, ty, tz = xs[i, sl], ys_dat[i, sl], zs[i, sl]

            trails_in[i].set_data(
                onp.where(mask, tx, onp.nan), onp.where(mask, ty, onp.nan)
            )
            trails_in[i].set_3d_properties(onp.where(mask, tz, onp.nan))
            trails_out[i].set_data(
                onp.where(~mask, tx, onp.nan), onp.where(~mask, ty, onp.nan)
            )
            trails_out[i].set_3d_properties(onp.where(~mask, tz, onp.nan))

            pos = onp.array([xs[i, frame], ys_dat[i, frame], zs[i, frame]])
            arms = _drone_arms_world(
                pos, phis[i, frame], thetas[i, frame], psis[i, frame]
            )
            arm_color = _ARM_COLOR if in_ix_mask[i, frame] else _OUT_COLOR
            drone_arm1[i].set_data(arms[0, :, 0], arms[0, :, 1])
            drone_arm1[i].set_3d_properties(arms[0, :, 2])
            drone_arm1[i].set_color(arm_color)
            drone_arm2[i].set_data(arms[1, :, 0], arms[1, :, 1])
            drone_arm2[i].set_3d_properties(arms[1, :, 2])
            drone_arm2[i].set_color(arm_color)
        time_text.set_text(f"$t = {float(ts[frame]):.2f}$ s")
        return all_lines + [time_text]

    anim = FuncAnimation(
        fig_vid,
        update_vid,
        frames=range(0, n_frames, frame_stride),
        init_func=init_vid,
        blit=False,
        interval=20,
    )
    anim.save(out_path, writer=FFMpegWriter(fps=50, bitrate=4000), dpi=150)
    plt.close(fig_vid)


# %%

# --------------------------------------------------------------------------- #
# Generate outputs for each curve                                              #
# --------------------------------------------------------------------------- #

mc_x0s = irx.utils.gen_ics(ix.scale(1.0), mc_N)

curve_name = "propofol_ref"
print(f"\n=== {curve_name} ===")
out_dir = BASE / "outputs" / curve_name
out_dir.mkdir(parents=True, exist_ok=True)

ts = jnp.arange(t0, tf + dt, dt)


x_ff = lambda t: HRP_xref(t)
u_ff = lambda t: HRPInput(t, x_ff)

def mc_compute(x0):
    return sys.compute_trajectory(
        t0,
        tf,
        x0,
        (
            lambda t, x: (
                control(x, control_net) - control(x_ff(t), control_net) + u_ff(t)
            ),
        ),
        dt=dt,
        max_steps=max_steps,
        solver=mc_method,
    )

print("  Running MC simulations...")
mc_trajs = jax.vmap(mc_compute)(mc_x0s).to_convenience()
x_ref_traj = jax.vmap(x_ff)(ts)

mc_controls = onp.array(
    jax.vmap(
        lambda traj: jax.vmap(
            lambda t, x: (
                control(x, control_net) - control(x_ff(t), control_net) + u_ff(t)
            )
        )(ts, traj)
    )(mc_trajs.ys)
)  # (mc_N, T, 4)
u_ref_traj = jax.vmap(u_ff)(ts)  # (T, 4)

fig = plot_static(mc_trajs, x_ref_traj, mc_N)
fig.savefig(out_dir / "states_static.png", dpi=150)
fig.savefig(out_dir / "states_static.pdf")
plt.close(fig)
print("  Saved states_static")

fig = plot_states_time(mc_trajs, ts, x_ref_traj, ix, mc_N)
fig.savefig(out_dir / "states_time.png", dpi=150)
fig.savefig(out_dir / "states_time.pdf")
plt.close(fig)
print("  Saved states_time")

fig = plot_controls_time(mc_controls, ts, onp.array(u_ref_traj), mc_N)
fig.savefig(out_dir / "controls_time.png", dpi=150)
fig.savefig(out_dir / "controls_time.pdf")
plt.close(fig)
print("  Saved controls_time")

save_video(mc_trajs, ts, x_ref_traj, ix, mc_N, out_dir / "video.mp4")
print("  Saved video.mp4")

# %%
