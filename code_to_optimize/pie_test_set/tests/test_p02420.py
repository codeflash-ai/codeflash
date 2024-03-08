from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p02420_0():
    input_content = "aabc\n3\n1\n2\n1\nvwxyz\n2\n3\n4\n-"
    expected_output = "aabc\nxyzvw"
    run_pie_test_case("../p02420.py", input_content, expected_output)


def test_problem_p02420_1():
    input_content = "aabc\n3\n1\n2\n1\nvwxyz\n2\n3\n4\n-"
    expected_output = "aabc\nxyzvw"
    run_pie_test_case("../p02420.py", input_content, expected_output)
