import math
from functools import partial

import jax.numpy as jnp
import numpy as np
import tensorflow as tf
import torch
from jax import lax
from numba import njit

_numba_available = False


def tridiagonal_solve(a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray) -> np.ndarray:
    n = len(b)

    c_prime = np.empty(n - 1, dtype=np.float64)
    d_prime = np.empty(n, dtype=np.float64)
    x = np.empty(n, dtype=np.float64)

    # Alias arrays to local variables to avoid repeated attribute lookups
    a_arr = a
    b_arr = b
    c_arr = c
    d_arr = d
    cp = c_prime
    dp = d_prime
    x_arr = x

    # First element
    prev_cprime = c_arr[0] / b_arr[0]
    cp[0] = prev_cprime
    prev_dprime = d_arr[0] / b_arr[0]
    dp[0] = prev_dprime

    # Forward sweep (compute c_prime and d_prime)

    for i in range(1, n - 1):
        ai_1 = a_arr[i - 1]
        denom = b_arr[i] - ai_1 * prev_cprime
        curr_cprime = c_arr[i] / denom
        curr_dprime = (d_arr[i] - ai_1 * prev_dprime) / denom
        cp[i] = curr_cprime
        dp[i] = curr_dprime
        prev_cprime = curr_cprime
        prev_dprime = curr_dprime

    # Last d_prime entry
    denom = b_arr[n - 1] - a_arr[n - 2] * prev_cprime
    dp[n - 1] = (d_arr[n - 1] - a_arr[n - 2] * prev_dprime) / denom

    # Back substitution using a scalar for the "next x" value
    prev_x = dp[n - 1]
    x_arr[n - 1] = prev_x
    for i in range(n - 2, -1, -1):
        xi = dp[i] - cp[i] * prev_x
        x_arr[i] = xi
        prev_x = xi

    return x


def leapfrog_integration(
    positions: np.ndarray,
    velocities: np.ndarray,
    masses: np.ndarray,
    dt: float,
    n_steps: int,
    softening: float = 0.01
) -> tuple[np.ndarray, np.ndarray]:
    n_particles = len(masses)
    pos = positions.copy()
    vel = velocities.copy()
    acc = np.zeros_like(pos)

    G = 1.0

    # Use the JIT-compiled core when available; it operates in-place on pos and vel.
    # If Numba is not installed, _leapfrog_core is a plain-Python function (behaves identically).
    _leapfrog_core(pos, vel, masses, dt, n_steps, softening)

    return pos, vel


def longest_increasing_subsequence_length(arr: np.ndarray) -> int:
    n = len(arr)
    if n == 0:
        return 0

    dp = np.ones(n, dtype=np.int64)

    for i in range(1, n):
        for j in range(i):
            if arr[j] < arr[i]:
                if dp[j] + 1 > dp[i]:
                    dp[i] = dp[j] + 1

    max_length = dp[0]
    for i in range(1, n):
        if dp[i] > max_length:
            max_length = dp[i]

    return max_length


def _tridiagonal_forward_step_jax(carry, inputs):
    c_prev, d_prev = carry
    a_i, b_i, c_i, d_i = inputs
    denom = b_i - a_i * c_prev
    c_new = c_i / denom
    d_new = (d_i - a_i * d_prev) / denom
    return (c_new, d_new), (c_new, d_new)


def _tridiagonal_back_step_jax(x_next, inputs):
    d_prime_i, c_prime_i = inputs
    x_i = d_prime_i - c_prime_i * x_next
    return x_i, x_i


def tridiagonal_solve_jax(a, b, c, d):
    n = b.shape[0]

    c_prime_0 = c[0] / b[0]
    d_prime_0 = d[0] / b[0]

    scan_inputs = (a[:-1], b[1:-1], c[1:], d[1:-1])

    _, (c_prime_rest, d_prime_mid) = lax.scan(
        _tridiagonal_forward_step_jax,
        (c_prime_0, d_prime_0),
        scan_inputs
    )

    c_prime = jnp.concatenate([jnp.array([c_prime_0]), c_prime_rest])

    denom_last = b[n - 1] - a[n - 2] * c_prime[n - 2]
    d_prime_last = (d[n - 1] - a[n - 2] * d_prime_mid[-1]) / denom_last
    d_prime = jnp.concatenate([jnp.array([d_prime_0]), d_prime_mid, jnp.array([d_prime_last])])

    x_last = d_prime[n - 1]
    _, x_rest = lax.scan(
        _tridiagonal_back_step_jax,
        x_last,
        (d_prime[:-1], c_prime),
        reverse=True
    )

    x = jnp.concatenate([x_rest, jnp.array([x_last])])
    return x


def _leapfrog_compute_accelerations_jax(pos, masses, softening):
    G = 1.0
    diff = pos[jnp.newaxis, :, :] - pos[:, jnp.newaxis, :]

    dist_sq = jnp.sum(diff ** 2, axis=-1) + softening ** 2
    dist = jnp.sqrt(dist_sq)
    dist_cubed = dist_sq * dist

    dist_cubed = jnp.where(dist_cubed == 0, 1.0, dist_cubed)

    force_factor = G * masses[jnp.newaxis, :] / dist_cubed

    acc = jnp.sum(force_factor[:, :, jnp.newaxis] * diff, axis=1)
    return acc


def _leapfrog_step_jax(carry, _, masses, softening, dt):
    pos, vel = carry
    acc = _leapfrog_compute_accelerations_jax(pos, masses, softening)

    vel = vel + 0.5 * dt * acc
    pos = pos + dt * vel
    vel = vel + 0.5 * dt * acc

    return (pos, vel), None


def leapfrog_integration_jax(
    positions,
    velocities,
    masses,
    dt: float,
    n_steps: int,
    softening: float = 0.01
):
    step_fn = partial(_leapfrog_step_jax, masses=masses, softening=softening, dt=dt)
    (final_pos, final_vel), _ = lax.scan(step_fn, (positions, velocities), None, length=n_steps)
    return final_pos, final_vel


def _lis_inner_body_jax(j, dp_inner, arr, i):
    condition = (arr[j] < arr[i]) & (dp_inner[j] + 1 > dp_inner[i])
    new_val = jnp.where(condition, dp_inner[j] + 1, dp_inner[i])
    return dp_inner.at[i].set(new_val)


def _lis_outer_body_jax(i, dp, arr):
    inner_fn = partial(_lis_inner_body_jax, arr=arr, i=i)
    dp = lax.fori_loop(0, i, inner_fn, dp)
    return dp


def longest_increasing_subsequence_length_jax(arr):
    n = arr.shape[0]

    if n == 0:
        return 0

    outer_fn = partial(_lis_outer_body_jax, arr=arr)
    dp = jnp.ones(n, dtype=jnp.int32)
    dp = lax.fori_loop(1, n, outer_fn, dp)

    return int(jnp.max(dp))


def tridiagonal_solve_torch(a, b, c, d):
    device = b.device
    dtype = b.dtype
    n = b.shape[0]

    c_prime = torch.zeros(n - 1, device=device, dtype=dtype)
    d_prime = torch.zeros(n, device=device, dtype=dtype)
    x = torch.zeros(n, device=device, dtype=dtype)

    c_prime[0] = c[0] / b[0]
    d_prime[0] = d[0] / b[0]

    for i in range(1, n - 1):
        denom = b[i] - a[i - 1] * c_prime[i - 1]
        c_prime[i] = c[i] / denom
        d_prime[i] = (d[i] - a[i - 1] * d_prime[i - 1]) / denom

    denom = b[n - 1] - a[n - 2] * c_prime[n - 2]
    d_prime[n - 1] = (d[n - 1] - a[n - 2] * d_prime[n - 2]) / denom

    x[n - 1] = d_prime[n - 1]
    for i in range(n - 2, -1, -1):
        x[i] = d_prime[i] - c_prime[i] * x[i + 1]

    return x


def leapfrog_integration_torch(
    positions,
    velocities,
    masses,
    dt: float,
    n_steps: int,
    softening: float = 0.01
):
    G = 1.0

    pos = positions.clone()
    vel = velocities.clone()

    for _ in range(n_steps):
        diff = pos.unsqueeze(0) - pos.unsqueeze(1)

        dist_sq = torch.sum(diff ** 2, dim=-1) + softening ** 2
        dist = torch.sqrt(dist_sq)
        dist_cubed = dist_sq * dist

        dist_cubed = torch.where(dist_cubed == 0, torch.ones_like(dist_cubed), dist_cubed)

        force_factor = G * masses.unsqueeze(0) / dist_cubed

        acc = torch.sum(force_factor.unsqueeze(-1) * diff, dim=1)

        vel = vel + 0.5 * dt * acc
        pos = pos + dt * vel
        vel = vel + 0.5 * dt * acc

    return pos, vel


def longest_increasing_subsequence_length_torch(arr):
    n = arr.shape[0]

    if n == 0:
        return 0

    device = arr.device
    dp = torch.ones(n, device=device, dtype=torch.int64)

    for i in range(1, n):
        for j in range(i):
            if arr[j] < arr[i]:
                if dp[j] + 1 > dp[i]:
                    dp[i] = dp[j] + 1

    return int(torch.max(dp).item())


def _tridiagonal_forward_cond_tf(i, _c_prime, _d_prime, n, _a, _b, _c, _d):
    return i < n - 1


def _tridiagonal_forward_body_tf(i, c_prime, d_prime, n, a, b, c, d):
    c_prev = c_prime[i - 1]
    d_prev = d_prime[i - 1]
    denom = b[i] - a[i - 1] * c_prev
    c_val = c[i] / denom
    d_val = (d[i] - a[i - 1] * d_prev) / denom
    c_prime = tf.tensor_scatter_nd_update(c_prime, tf.reshape(i, [1, 1]), tf.reshape(c_val, [1]))
    d_prime = tf.tensor_scatter_nd_update(d_prime, tf.reshape(i, [1, 1]), tf.reshape(d_val, [1]))
    return i + 1, c_prime, d_prime, n, a, b, c, d


def _tridiagonal_back_cond_tf(i, _x, _c_prime, _d_prime):
    return i >= 0


def _tridiagonal_back_body_tf(i, x, c_prime, d_prime):
    x_next = x[i + 1]
    x_val = d_prime[i] - c_prime[i] * x_next
    x = tf.tensor_scatter_nd_update(x, tf.reshape(i, [1, 1]), tf.reshape(x_val, [1]))
    return i - 1, x, c_prime, d_prime


def tridiagonal_solve_tf(a, b, c, d):
    n = tf.shape(b)[0]
    dtype = b.dtype

    c_prime = tf.zeros([n - 1], dtype=dtype)
    d_prime = tf.zeros([n], dtype=dtype)

    c_prime = tf.tensor_scatter_nd_update(c_prime, [[0]], tf.reshape(c[0] / b[0], [1]))
    d_prime = tf.tensor_scatter_nd_update(d_prime, [[0]], tf.reshape(d[0] / b[0], [1]))

    _, c_prime, d_prime, _, _, _, _, _ = tf.while_loop(
        _tridiagonal_forward_cond_tf,
        _tridiagonal_forward_body_tf,
        [1, c_prime, d_prime, n, a, b, c, d]
    )

    c_last = c_prime[n - 2]
    d_prev = d_prime[n - 2]
    denom = b[n - 1] - a[n - 2] * c_last
    d_last = (d[n - 1] - a[n - 2] * d_prev) / denom
    d_prime = tf.tensor_scatter_nd_update(d_prime, tf.reshape(n - 1, [1, 1]), tf.reshape(d_last, [1]))

    x = tf.zeros([n], dtype=dtype)
    x = tf.tensor_scatter_nd_update(x, tf.reshape(n - 1, [1, 1]), tf.reshape(d_prime[n - 1], [1]))

    _, x, _, _ = tf.while_loop(
        _tridiagonal_back_cond_tf,
        _tridiagonal_back_body_tf,
        [n - 2, x, c_prime, d_prime]
    )

    return x


def _leapfrog_compute_accelerations_tf(pos, masses, softening, G):
    diff = tf.expand_dims(pos, 0) - tf.expand_dims(pos, 1)

    dist_sq = tf.reduce_sum(diff ** 2, axis=-1) + softening ** 2
    dist = tf.sqrt(dist_sq)
    dist_cubed = dist_sq * dist

    dist_cubed = tf.where(dist_cubed == 0, tf.ones_like(dist_cubed), dist_cubed)

    force_factor = G * tf.expand_dims(masses, 0) / dist_cubed

    acc = tf.reduce_sum(tf.expand_dims(force_factor, -1) * diff, axis=1)
    return acc


def _leapfrog_step_body_tf(i, pos, vel, masses, softening, dt, n_steps):
    G = 1.0
    acc = _leapfrog_compute_accelerations_tf(pos, masses, softening, G)

    vel = vel + 0.5 * dt * acc
    pos = pos + dt * vel
    vel = vel + 0.5 * dt * acc

    return i + 1, pos, vel, masses, softening, dt, n_steps


def _leapfrog_step_cond_tf(i, _pos, _vel, _masses, _softening, _dt, n_steps):
    return i < n_steps


def leapfrog_integration_tf(
    positions,
    velocities,
    masses,
    dt: float,
    n_steps: int,
    softening: float = 0.01
):
    dt = tf.constant(dt, dtype=positions.dtype)
    softening = tf.constant(softening, dtype=positions.dtype)

    _, final_pos, final_vel, _, _, _, _ = tf.while_loop(
        _leapfrog_step_cond_tf,
        _leapfrog_step_body_tf,
        [0, positions, velocities, masses, softening, dt, n_steps]
    )

    return final_pos, final_vel


def _lis_inner_body_tf(j, dp_inner, arr, i):
    condition = tf.logical_and(arr[j] < arr[i], dp_inner[j] + 1 > dp_inner[i])
    new_val = tf.where(condition, dp_inner[j] + 1, dp_inner[i])
    indices = tf.reshape(i, [1, 1])
    updates = tf.reshape(new_val, [1])
    dp_updated = tf.tensor_scatter_nd_update(dp_inner, indices, updates)
    return j + 1, dp_updated, arr, i


def _lis_inner_cond_tf(j, _dp_inner, _arr, i):
    return j < i


def _lis_outer_body_tf(i, dp, arr, n):
    _, dp, _, _ = tf.while_loop(
        _lis_inner_cond_tf,
        _lis_inner_body_tf,
        [0, dp, arr, i]
    )
    return i + 1, dp, arr, n


def _lis_outer_cond_tf(i, _dp, _arr, n):
    return i < n


def longest_increasing_subsequence_length_tf(arr):
    n = tf.shape(arr)[0]

    if n == 0:
        return 0

    dp = tf.ones(n, dtype=tf.int32)

    _, dp, _, _ = tf.while_loop(
        _lis_outer_cond_tf,
        _lis_outer_body_tf,
        [1, dp, arr, n]
    )

    return int(tf.reduce_max(dp))

def njit(*args, **kwargs):
    # simple pass-through decorator when Numba isn't installed
    def _decorator(func):
        return func
    if args and callable(args[0]):
        return args[0]
    return _decorator


@njit(cache=True)
def _leapfrog_core(
    pos: np.ndarray,
    vel: np.ndarray,
    masses: np.ndarray,
    dt: float,
    n_steps: int,
    softening: float
) -> None:
    n_particles = masses.shape[0]
    acc = np.zeros_like(pos)

    G = 1.0

    soft2 = softening * softening
    half_dt = 0.5 * dt

    for step in range(n_steps):
        # reset accelerations
        for ii in range(n_particles):
            acc[ii, 0] = 0.0
            acc[ii, 1] = 0.0
            acc[ii, 2] = 0.0

        # pairwise interactions (symmetric updates)
        for i in range(n_particles):
            mi = masses[i]
            pos_i0 = pos[i, 0]
            pos_i1 = pos[i, 1]
            pos_i2 = pos[i, 2]
            for j in range(i + 1, n_particles):
                dx = pos[j, 0] - pos_i0
                dy = pos[j, 1] - pos_i1
                dz = pos[j, 2] - pos_i2

                dist_sq = dx * dx + dy * dy + dz * dz + soft2
                dist = math.sqrt(dist_sq)
                dist_cubed = dist_sq * dist

                force_over_dist = G / dist_cubed

                mj = masses[j]

                ax = mj * force_over_dist * dx
                ay = mj * force_over_dist * dy
                az = mj * force_over_dist * dz

                acc[i, 0] += ax
                acc[i, 1] += ay
                acc[i, 2] += az

                acc[j, 0] -= mi * force_over_dist * dx
                acc[j, 1] -= mi * force_over_dist * dy
                acc[j, 2] -= mi * force_over_dist * dz

        # velocity half-step
        for i in range(n_particles):
            vel[i, 0] += half_dt * acc[i, 0]
            vel[i, 1] += half_dt * acc[i, 1]
            vel[i, 2] += half_dt * acc[i, 2]

        # position full-step
        for i in range(n_particles):
            pos[i, 0] += dt * vel[i, 0]
            pos[i, 1] += dt * vel[i, 1]
            pos[i, 2] += dt * vel[i, 2]

        # velocity half-step
        for i in range(n_particles):
            vel[i, 0] += half_dt * acc[i, 0]
            vel[i, 1] += half_dt * acc[i, 1]
            vel[i, 2] += half_dt * acc[i, 2]
