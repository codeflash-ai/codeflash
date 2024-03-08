from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p02901_0():
    input_content = "2 3\n10 1\n1\n15 1\n2\n30 2\n1 2"
    expected_output = "25"
    run_pie_test_case("../p02901.py", input_content, expected_output)


def test_problem_p02901_1():
    input_content = (
        "4 6\n67786 3\n1 3 4\n3497 1\n2\n44908 3\n2 3 4\n2156 3\n2 3 4\n26230 1\n2\n86918 1\n3"
    )
    expected_output = "69942"
    run_pie_test_case("../p02901.py", input_content, expected_output)


def test_problem_p02901_2():
    input_content = "12 1\n100000 1\n2"
    expected_output = "-1"
    run_pie_test_case("../p02901.py", input_content, expected_output)


def test_problem_p02901_3():
    input_content = "2 3\n10 1\n1\n15 1\n2\n30 2\n1 2"
    expected_output = "25"
    run_pie_test_case("../p02901.py", input_content, expected_output)
