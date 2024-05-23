from code_to_optimize.final_test_set.generate_primes import generate_primes


def test_generate_primes():
    primes = generate_primes(100)
    assert len(primes) == 25

    primes = generate_primes(10000)
    assert len(primes) == 1229

    primes = generate_primes(100000)
    assert len(primes) == 9592
