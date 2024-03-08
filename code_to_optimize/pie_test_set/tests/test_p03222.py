from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03222_0():
    input_content = "1 3 2"
    expected_output = "1"
    run_pie_test_case("../p03222.py", input_content, expected_output)


def test_problem_p03222_1():
    input_content = "2 3 1"
    expected_output = "5"
    run_pie_test_case("../p03222.py", input_content, expected_output)


def test_problem_p03222_2():
    input_content = "1 3 2"
    expected_output = "1"
    run_pie_test_case("../p03222.py", input_content, expected_output)


def test_problem_p03222_3():
    input_content = "1 3 1"
    expected_output = "2"
    run_pie_test_case("../p03222.py", input_content, expected_output)


def test_problem_p03222_4():
    input_content = "7 1 1"
    expected_output = "1"
    run_pie_test_case("../p03222.py", input_content, expected_output)


def test_problem_p03222_5():
    input_content = "2 3 3"
    expected_output = "1"
    run_pie_test_case("../p03222.py", input_content, expected_output)


def test_problem_p03222_6():
    input_content = "15 8 5"
    expected_output = "437760187"
    run_pie_test_case("../p03222.py", input_content, expected_output)
