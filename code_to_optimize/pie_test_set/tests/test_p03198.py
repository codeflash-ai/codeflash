from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03198_0():
    input_content = "4\n3 1 4 1"
    expected_output = "3"
    run_pie_test_case("../p03198.py", input_content, expected_output)


def test_problem_p03198_1():
    input_content = "4\n3 1 4 1"
    expected_output = "3"
    run_pie_test_case("../p03198.py", input_content, expected_output)


def test_problem_p03198_2():
    input_content = (
        "8\n657312726 129662684 181537270 324043958 468214806 916875077 825989291 319670097"
    )
    expected_output = "7"
    run_pie_test_case("../p03198.py", input_content, expected_output)


def test_problem_p03198_3():
    input_content = "5\n1 2 3 4 5"
    expected_output = "0"
    run_pie_test_case("../p03198.py", input_content, expected_output)
