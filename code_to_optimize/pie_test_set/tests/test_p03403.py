from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03403_0():
    input_content = "3\n3 5 -1"
    expected_output = "12\n8\n10"
    run_pie_test_case("../p03403.py", input_content, expected_output)


def test_problem_p03403_1():
    input_content = "6\n-679 -2409 -3258 3095 -3291 -4462"
    expected_output = "21630\n21630\n19932\n8924\n21630\n19288"
    run_pie_test_case("../p03403.py", input_content, expected_output)


def test_problem_p03403_2():
    input_content = "5\n1 1 1 2 0"
    expected_output = "4\n4\n4\n2\n4"
    run_pie_test_case("../p03403.py", input_content, expected_output)


def test_problem_p03403_3():
    input_content = "3\n3 5 -1"
    expected_output = "12\n8\n10"
    run_pie_test_case("../p03403.py", input_content, expected_output)
