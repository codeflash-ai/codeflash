from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p02929_0():
    input_content = "2\nBWWB"
    expected_output = "4"
    run_pie_test_case("../p02929.py", input_content, expected_output)


def test_problem_p02929_1():
    input_content = "2\nBWWB"
    expected_output = "4"
    run_pie_test_case("../p02929.py", input_content, expected_output)


def test_problem_p02929_2():
    input_content = "5\nWWWWWWWWWW"
    expected_output = "0"
    run_pie_test_case("../p02929.py", input_content, expected_output)


def test_problem_p02929_3():
    input_content = "4\nBWBBWWWB"
    expected_output = "288"
    run_pie_test_case("../p02929.py", input_content, expected_output)
