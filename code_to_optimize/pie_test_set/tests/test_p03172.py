from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03172_0():
    input_content = "3 4\n1 2 3"
    expected_output = "5"
    run_pie_test_case("../p03172.py", input_content, expected_output)


def test_problem_p03172_1():
    input_content = "2 0\n0 0"
    expected_output = "1"
    run_pie_test_case("../p03172.py", input_content, expected_output)


def test_problem_p03172_2():
    input_content = "1 10\n9"
    expected_output = "0"
    run_pie_test_case("../p03172.py", input_content, expected_output)


def test_problem_p03172_3():
    input_content = "4 100000\n100000 100000 100000 100000"
    expected_output = "665683269"
    run_pie_test_case("../p03172.py", input_content, expected_output)


def test_problem_p03172_4():
    input_content = "3 4\n1 2 3"
    expected_output = "5"
    run_pie_test_case("../p03172.py", input_content, expected_output)
