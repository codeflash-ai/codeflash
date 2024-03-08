from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03760_0():
    input_content = "xyz\nabc"
    expected_output = "xaybzc"
    run_pie_test_case("../p03760.py", input_content, expected_output)


def test_problem_p03760_1():
    input_content = "atcoderbeginnercontest\natcoderregularcontest"
    expected_output = "aattccooddeerrbreeggiunlnaerrccoonntteesstt"
    run_pie_test_case("../p03760.py", input_content, expected_output)


def test_problem_p03760_2():
    input_content = "xyz\nabc"
    expected_output = "xaybzc"
    run_pie_test_case("../p03760.py", input_content, expected_output)
