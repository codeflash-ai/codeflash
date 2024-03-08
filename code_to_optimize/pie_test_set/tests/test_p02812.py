from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p02812_0():
    input_content = "10\nZABCDBABCQ"
    expected_output = "2"
    run_pie_test_case("../p02812.py", input_content, expected_output)


def test_problem_p02812_1():
    input_content = "19\nTHREEONEFOURONEFIVE"
    expected_output = "0"
    run_pie_test_case("../p02812.py", input_content, expected_output)


def test_problem_p02812_2():
    input_content = "33\nABCCABCBABCCABACBCBBABCBCBCBCABCB"
    expected_output = "5"
    run_pie_test_case("../p02812.py", input_content, expected_output)


def test_problem_p02812_3():
    input_content = "10\nZABCDBABCQ"
    expected_output = "2"
    run_pie_test_case("../p02812.py", input_content, expected_output)
