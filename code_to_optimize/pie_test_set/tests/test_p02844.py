from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p02844_0():
    input_content = "4\n0224"
    expected_output = "3"
    run_pie_test_case("../p02844.py", input_content, expected_output)


def test_problem_p02844_1():
    input_content = "4\n0224"
    expected_output = "3"
    run_pie_test_case("../p02844.py", input_content, expected_output)


def test_problem_p02844_2():
    input_content = "6\n123123"
    expected_output = "17"
    run_pie_test_case("../p02844.py", input_content, expected_output)


def test_problem_p02844_3():
    input_content = "19\n3141592653589793238"
    expected_output = "329"
    run_pie_test_case("../p02844.py", input_content, expected_output)
