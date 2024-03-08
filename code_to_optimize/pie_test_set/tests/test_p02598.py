from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p02598_0():
    input_content = "2 3\n7 9"
    expected_output = "4"
    run_pie_test_case("../p02598.py", input_content, expected_output)


def test_problem_p02598_1():
    input_content = "3 0\n3 4 5"
    expected_output = "5"
    run_pie_test_case("../p02598.py", input_content, expected_output)


def test_problem_p02598_2():
    input_content = "2 3\n7 9"
    expected_output = "4"
    run_pie_test_case("../p02598.py", input_content, expected_output)


def test_problem_p02598_3():
    input_content = "10 10\n158260522 877914575 602436426 24979445 861648772 623690081 433933447 476190629 262703497 211047202"
    expected_output = "292638192"
    run_pie_test_case("../p02598.py", input_content, expected_output)
