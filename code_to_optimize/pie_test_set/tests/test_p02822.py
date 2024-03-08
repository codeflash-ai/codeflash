from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p02822_0():
    input_content = "3\n1 2\n2 3"
    expected_output = "125000001"
    run_pie_test_case("../p02822.py", input_content, expected_output)


def test_problem_p02822_1():
    input_content = "4\n1 2\n2 3\n3 4"
    expected_output = "375000003"
    run_pie_test_case("../p02822.py", input_content, expected_output)


def test_problem_p02822_2():
    input_content = "4\n1 2\n1 3\n1 4"
    expected_output = "250000002"
    run_pie_test_case("../p02822.py", input_content, expected_output)


def test_problem_p02822_3():
    input_content = "7\n4 7\n3 1\n2 6\n5 2\n7 1\n2 7"
    expected_output = "570312505"
    run_pie_test_case("../p02822.py", input_content, expected_output)


def test_problem_p02822_4():
    input_content = "3\n1 2\n2 3"
    expected_output = "125000001"
    run_pie_test_case("../p02822.py", input_content, expected_output)
