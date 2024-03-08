from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03629_0():
    input_content = "atcoderregularcontest"
    expected_output = "b"
    run_pie_test_case("../p03629.py", input_content, expected_output)


def test_problem_p03629_1():
    input_content = "frqnvhydscshfcgdemurlfrutcpzhopfotpifgepnqjxupnskapziurswqazdwnwbgdhyktfyhqqxpoidfhjdakoxraiedxskywuepzfniuyskxiyjpjlxuqnfgmnjcvtlpnclfkpervxmdbvrbrdn"
    expected_output = "aca"
    run_pie_test_case("../p03629.py", input_content, expected_output)


def test_problem_p03629_2():
    input_content = "abcdefghijklmnopqrstuvwxyz"
    expected_output = "aa"
    run_pie_test_case("../p03629.py", input_content, expected_output)


def test_problem_p03629_3():
    input_content = "atcoderregularcontest"
    expected_output = "b"
    run_pie_test_case("../p03629.py", input_content, expected_output)
