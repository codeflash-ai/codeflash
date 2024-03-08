from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p02892_0():
    input_content = "2\n01\n10"
    expected_output = "2"
    run_pie_test_case("../p02892.py", input_content, expected_output)


def test_problem_p02892_1():
    input_content = "2\n01\n10"
    expected_output = "2"
    run_pie_test_case("../p02892.py", input_content, expected_output)


def test_problem_p02892_2():
    input_content = "6\n010110\n101001\n010100\n101000\n100000\n010000"
    expected_output = "4"
    run_pie_test_case("../p02892.py", input_content, expected_output)


def test_problem_p02892_3():
    input_content = "3\n011\n101\n110"
    expected_output = "-1"
    run_pie_test_case("../p02892.py", input_content, expected_output)
