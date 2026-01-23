from functools import partial
import torch

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

