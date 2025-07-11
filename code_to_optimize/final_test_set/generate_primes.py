def is_prime(n):
    # No changes: utility function, keep as is for identical behavior.
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
    # Use Sieve of Eratosthenes for optimal speed and memory
    if limit < 2:
        return []
    sieve = [True] * (limit + 1)
    sieve[0:2] = [False, False]
    # No need to go beyond sqrt(limit)
    for num in range(2, int(limit**0.5) + 1):
        if sieve[num]:
            sieve[num * num : limit + 1 : num] = [False] * ((limit - num * num) // num + 1)
    primes = [num for num, is_p in enumerate(sieve) if is_p]
    return primes
