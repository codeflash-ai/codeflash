from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p02693_0():
    input_content = "7\n500 600"
    expected_output = "OK"
    run_pie_test_case("../p02693.py", input_content, expected_output)


def test_problem_p02693_1():
    input_content = "4\n5 7"
    expected_output = "NG"
    run_pie_test_case("../p02693.py", input_content, expected_output)


def test_problem_p02693_2():
    input_content = "7\n500 600"
    expected_output = "OK"
    run_pie_test_case("../p02693.py", input_content, expected_output)


def test_problem_p02693_3():
    input_content = "1\n11 11"
    expected_output = "OK"
    run_pie_test_case("../p02693.py", input_content, expected_output)
