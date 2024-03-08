from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03150_0():
    input_content = "keyofscience"
    expected_output = "YES"
    run_pie_test_case("../p03150.py", input_content, expected_output)


def test_problem_p03150_1():
    input_content = "keyence"
    expected_output = "YES"
    run_pie_test_case("../p03150.py", input_content, expected_output)


def test_problem_p03150_2():
    input_content = "ashlfyha"
    expected_output = "NO"
    run_pie_test_case("../p03150.py", input_content, expected_output)


def test_problem_p03150_3():
    input_content = "keyofscience"
    expected_output = "YES"
    run_pie_test_case("../p03150.py", input_content, expected_output)


def test_problem_p03150_4():
    input_content = "mpyszsbznf"
    expected_output = "NO"
    run_pie_test_case("../p03150.py", input_content, expected_output)
