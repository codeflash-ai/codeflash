from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p00693_0():
    input_content = "2 5\npermit 192168?? ?12??34?\ndeny 19216899 012343?5\n19216711 11233340 HiIamACracker\n19216891 01234345 Hello\n19216899 01234345 HiIamAlsoACracker\n19216809 11200340 World\n00000000 99999999 TheEndOfTheWorld\n1 2\npermit 12345678 23456789\n19216891 01234345 Hello\n12345678 23456789 Hello\n0 0"
    expected_output = (
        "2\n19216891 01234345 Hello\n19216809 11200340 World\n1\n12345678 23456789 Hello"
    )
    run_pie_test_case("../p00693.py", input_content, expected_output)


def test_problem_p00693_1():
    input_content = "2 5\npermit 192168?? ?12??34?\ndeny 19216899 012343?5\n19216711 11233340 HiIamACracker\n19216891 01234345 Hello\n19216899 01234345 HiIamAlsoACracker\n19216809 11200340 World\n00000000 99999999 TheEndOfTheWorld\n1 2\npermit 12345678 23456789\n19216891 01234345 Hello\n12345678 23456789 Hello\n0 0"
    expected_output = (
        "2\n19216891 01234345 Hello\n19216809 11200340 World\n1\n12345678 23456789 Hello"
    )
    run_pie_test_case("../p00693.py", input_content, expected_output)
