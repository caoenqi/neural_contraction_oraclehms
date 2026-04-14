# %%

import jax
import jax.numpy as jnp
import immrax as irx
import equinox as eqx
from functools import partial
import optax

from ncm_trainer import NCMTrainer
from propofol import (
    HRPropofol,
    cur_params,
    inputs,
    x_eq,
    u_eq,
    ncm,
    control,
    K_eq,
    S_eq,
    A_eq,
    B_eq,
    NCM,
    CONTROLLER,
)

device = "gpu"
jit = partial(eqx.filter_jit, backend=device)

sys = HRPropofol(cur_params)
print(sys.f(0.0, x_eq, u_eq))

# Hyperparameters for training
a = 0.01
b = 100.0
c = 0.001
lr = 1e-5
print(f"a={a:.4f}, b={b:.4f}, c={c:.4f}")

trainer = NCMTrainer(
    sys=sys,
    ncm_fn=ncm,
    control_fn=control,
    a=a,
    b=b,
    c=c,
    partition_indices=[6, 7, 8],
    device=device,
)

# %%

# ix_gen: linearly grows perturbation from 1% to 100% of max over num_pert levels.
_interval_max = jnp.array([0.983568240,
2.17185407,
0.758232405,
0.0,
0.0,
4.33893652,
0.0,
27.6860217,
72.8639679,
180.680146,
2.37861721
])
_max_diff = _interval_max - x_eq
_interval_min = jnp.array([0.0,
0.0,
0.0,
-0.605837295,
-948.155976,
0.0,
-948.155976,
0.0,
0.0,
0.0,
0.0
])
_min_diff = _interval_min - x_eq

_num_levels = 100


def ix_gen(i):
    alpha = i / _num_levels
    return irx.interval(x_eq + alpha * _min_diff, x_eq + alpha * _max_diff)


# %%
ncm_net = irx.NeuralNetwork(NCM, load=False)
control_net = irx.NeuralNetwork(CONTROLLER, load=False)
ncm_net = ncm_net.loadzeros()
control_net = control_net.loadzeros()

params = (ncm_net, control_net)
optim = optax.adamw(lr)

(ncm_net, control_net), ix, perti = trainer.train(
    params,
    optim,
    ix_gen,
    num_samples=0,
    divs=9,
    steps=1_000_000,
    ix_save_path=CONTROLLER / "ix_ut.npy",
    num_pert=_num_levels,
)

# %%

ncm_net = irx.NeuralNetwork(NCM, load=True)
control_net = irx.NeuralNetwork(CONTROLLER, load=True)

aM, bM = trainer.get_bounds_iM(trainer.M_crown(ix, ncm_net))
print(aM, bM)

print(f"Valid: {aM > 0}, Metric contracts at rate {c}")

# %%
