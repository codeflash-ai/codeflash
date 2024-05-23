from code_to_optimize.final_test_set.exponentiation import exponentiation


def test_exponentiation():
    res = exponentiation(2, 10)
    assert res == 1024

    res = exponentiation(2, 100)
    assert res == 1267650600228229401496703205376
