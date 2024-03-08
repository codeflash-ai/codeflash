from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03422_0():
    input_content = "2\n5 2\n3 3"
    expected_output = "Aoki"
    run_pie_test_case("../p03422.py", input_content, expected_output)


def test_problem_p03422_1():
    input_content = "4\n3141 59\n26535 897\n93 23\n8462 64"
    expected_output = "Takahashi"
    run_pie_test_case("../p03422.py", input_content, expected_output)


def test_problem_p03422_2():
    input_content = "3\n28 3\n16 4\n19 2"
    expected_output = "Aoki"
    run_pie_test_case("../p03422.py", input_content, expected_output)


def test_problem_p03422_3():
    input_content = "3\n3 2\n4 3\n5 1"
    expected_output = "Takahashi"
    run_pie_test_case("../p03422.py", input_content, expected_output)


def test_problem_p03422_4():
    input_content = "2\n5 2\n3 3"
    expected_output = "Aoki"
    run_pie_test_case("../p03422.py", input_content, expected_output)
