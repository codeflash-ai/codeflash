from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p02660_0():
    input_content = "24"
    expected_output = "3"
    run_pie_test_case("../p02660.py", input_content, expected_output)


def test_problem_p02660_1():
    input_content = "997764507000"
    expected_output = "7"
    run_pie_test_case("../p02660.py", input_content, expected_output)


def test_problem_p02660_2():
    input_content = "1"
    expected_output = "0"
    run_pie_test_case("../p02660.py", input_content, expected_output)


def test_problem_p02660_3():
    input_content = "1000000007"
    expected_output = "1"
    run_pie_test_case("../p02660.py", input_content, expected_output)


def test_problem_p02660_4():
    input_content = "64"
    expected_output = "3"
    run_pie_test_case("../p02660.py", input_content, expected_output)


def test_problem_p02660_5():
    input_content = "24"
    expected_output = "3"
    run_pie_test_case("../p02660.py", input_content, expected_output)
