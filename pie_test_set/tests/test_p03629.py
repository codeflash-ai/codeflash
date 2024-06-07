from pie_test_set.p03629 import problem_p03629


def test_problem_p03629_0():
    actual_output = problem_p03629("atcoderregularcontest")
    expected_output = "b"
    assert str(actual_output) == expected_output


def test_problem_p03629_1():
    actual_output = problem_p03629(
        "frqnvhydscshfcgdemurlfrutcpzhopfotpifgepnqjxupnskapziurswqazdwnwbgdhyktfyhqqxpoidfhjdakoxraiedxskywuepzfniuyskxiyjpjlxuqnfgmnjcvtlpnclfkpervxmdbvrbrdn"
    )
    expected_output = "aca"
    assert str(actual_output) == expected_output


def test_problem_p03629_2():
    actual_output = problem_p03629("abcdefghijklmnopqrstuvwxyz")
    expected_output = "aa"
    assert str(actual_output) == expected_output


def test_problem_p03629_3():
    actual_output = problem_p03629("atcoderregularcontest")
    expected_output = "b"
    assert str(actual_output) == expected_output
