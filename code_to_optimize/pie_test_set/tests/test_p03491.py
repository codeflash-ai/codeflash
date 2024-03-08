from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03491_0():
    input_content = "2 2\n00\n01"
    expected_output = "Alice"
    run_pie_test_case("../p03491.py", input_content, expected_output)


def test_problem_p03491_1():
    input_content = "3 3\n0\n10\n110"
    expected_output = "Alice"
    run_pie_test_case("../p03491.py", input_content, expected_output)


def test_problem_p03491_2():
    input_content = "2 3\n101\n11"
    expected_output = "Bob"
    run_pie_test_case("../p03491.py", input_content, expected_output)


def test_problem_p03491_3():
    input_content = "1 2\n11"
    expected_output = "Alice"
    run_pie_test_case("../p03491.py", input_content, expected_output)


def test_problem_p03491_4():
    input_content = "2 2\n00\n11"
    expected_output = "Bob"
    run_pie_test_case("../p03491.py", input_content, expected_output)


def test_problem_p03491_5():
    input_content = "2 1\n0\n1"
    expected_output = "Bob"
    run_pie_test_case("../p03491.py", input_content, expected_output)


def test_problem_p03491_6():
    input_content = "2 2\n00\n01"
    expected_output = "Alice"
    run_pie_test_case("../p03491.py", input_content, expected_output)
