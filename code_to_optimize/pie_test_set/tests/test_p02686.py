from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p02686_0():
    input_content = "2\n)\n(()"
    expected_output = "Yes"
    run_pie_test_case("../p02686.py", input_content, expected_output)


def test_problem_p02686_1():
    input_content = "3\n(((\n)\n)"
    expected_output = "No"
    run_pie_test_case("../p02686.py", input_content, expected_output)


def test_problem_p02686_2():
    input_content = "2\n)\n(()"
    expected_output = "Yes"
    run_pie_test_case("../p02686.py", input_content, expected_output)


def test_problem_p02686_3():
    input_content = "4\n((()))\n((((((\n))))))\n()()()"
    expected_output = "Yes"
    run_pie_test_case("../p02686.py", input_content, expected_output)


def test_problem_p02686_4():
    input_content = "2\n)(\n()"
    expected_output = "No"
    run_pie_test_case("../p02686.py", input_content, expected_output)
