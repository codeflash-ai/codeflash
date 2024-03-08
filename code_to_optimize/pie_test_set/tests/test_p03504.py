from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03504_0():
    input_content = "3 2\n1 7 2\n7 8 1\n8 12 1"
    expected_output = "2"
    run_pie_test_case("../p03504.py", input_content, expected_output)


def test_problem_p03504_1():
    input_content = (
        "9 4\n56 60 4\n33 37 2\n89 90 3\n32 43 1\n67 68 3\n49 51 3\n31 32 3\n70 71 1\n11 12 3"
    )
    expected_output = "2"
    run_pie_test_case("../p03504.py", input_content, expected_output)


def test_problem_p03504_2():
    input_content = "3 2\n1 7 2\n7 8 1\n8 12 1"
    expected_output = "2"
    run_pie_test_case("../p03504.py", input_content, expected_output)


def test_problem_p03504_3():
    input_content = "3 4\n1 3 2\n3 4 4\n1 4 3"
    expected_output = "3"
    run_pie_test_case("../p03504.py", input_content, expected_output)
