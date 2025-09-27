import jax.numpy as jnp


def sum_hand(hand):
    """Returns the total points in a hand."""
    return sum(hand) + (10 * usable_ace(hand))


def usable_ace(hand):
    """Checks to se if a hand has a usable ace."""
    return jnp.logical_and((jnp.count_nonzero(hand == 1) > 0), (sum(hand) + 10 <= 21))
