from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03054_0():
    input_content = "2 3 3\n2 2\nRRL\nLUD"
    expected_output = "YES"
    run_pie_test_case("../p03054.py", input_content, expected_output)


def test_problem_p03054_1():
    input_content = "2 3 3\n2 2\nRRL\nLUD"
    expected_output = "YES"
    run_pie_test_case("../p03054.py", input_content, expected_output)


def test_problem_p03054_2():
    input_content = "4 3 5\n2 2\nUDRRR\nLLDUD"
    expected_output = "NO"
    run_pie_test_case("../p03054.py", input_content, expected_output)


def test_problem_p03054_3():
    input_content = "5 6 11\n2 1\nRLDRRUDDLRL\nURRDRLLDLRD"
    expected_output = "NO"
    run_pie_test_case("../p03054.py", input_content, expected_output)
