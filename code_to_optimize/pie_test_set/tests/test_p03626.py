from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03626_0():
    input_content = "3\naab\nccb"
    expected_output = "6"
    run_pie_test_case("../p03626.py", input_content, expected_output)


def test_problem_p03626_1():
    input_content = "1\nZ\nZ"
    expected_output = "3"
    run_pie_test_case("../p03626.py", input_content, expected_output)


def test_problem_p03626_2():
    input_content = "52\nRvvttdWIyyPPQFFZZssffEEkkaSSDKqcibbeYrhAljCCGGJppHHn\nRLLwwdWIxxNNQUUXXVVMMooBBaggDKqcimmeYrhAljOOTTJuuzzn"
    expected_output = "958681902"
    run_pie_test_case("../p03626.py", input_content, expected_output)


def test_problem_p03626_3():
    input_content = "3\naab\nccb"
    expected_output = "6"
    run_pie_test_case("../p03626.py", input_content, expected_output)
