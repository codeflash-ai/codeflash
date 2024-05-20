def is_prime(n):
    """Check if a number is prime."""
    if n <= 1:
        return False
    if n <= 3:
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False
    i = 5
    while i * i <= n:
        if n % i == 0 or n % (i + 2) == 0:
            return False
        i += 6
    return True


def generate_primes(limit):
    """Generate primes up to a limit using the Sieve of Eratosthenes."""
    sieve = [True] * (limit + 1)
    p = 2
    while p * p <= limit:
        if sieve[p] is True:
            for i in range(p * p, limit + 1, p):
                sieve[i] = False
        p += 1
    primes = [p for p in range(2, limit + 1) if sieve[p]]
    return primes
