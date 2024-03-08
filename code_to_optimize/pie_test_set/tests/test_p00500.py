from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p00500_0():
    input_content = "5\n100 99 98\n100 97 92\n63 89 63\n99 99 99\n89 97 98"
    expected_output = "0\n92\n215\n198\n89"
    run_pie_test_case("../p00500.py", input_content, expected_output)


def test_problem_p00500_1():
    input_content = "5\n100 99 98\n100 97 92\n63 89 63\n99 99 99\n89 97 98"
    expected_output = "0\n92\n215\n198\n89"
    run_pie_test_case("../p00500.py", input_content, expected_output)
