from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p00663_0():
    input_content = "(B&B&f)|(~d&~i&i)|(~v&i&~V)|(~g&~e&o)|(~f&d&~v)|(d&~i&o)|(g&i&~B)|(~i&f&d)|(e&~i&~V)|(~v&f&~d)\n(S&X&~X)\n#"
    expected_output = "yes\nno"
    run_pie_test_case("../p00663.py", input_content, expected_output)


def test_problem_p00663_1():
    input_content = "(B&B&f)|(~d&~i&i)|(~v&i&~V)|(~g&~e&o)|(~f&d&~v)|(d&~i&o)|(g&i&~B)|(~i&f&d)|(e&~i&~V)|(~v&f&~d)\n(S&X&~X)\n#"
    expected_output = "yes\nno"
    run_pie_test_case("../p00663.py", input_content, expected_output)


def test_problem_p00663_2():
    input_content = "(B&B&f)|(~d&~i&i)|(~v&i&~V)|(~g&~e&o)|(~f&d&~v)|(d&~i&o)|(g&i&~B)|(~i&f&d)|(e&~i&~V)|(~v&f&~d)\n(S&X&~X)"
    expected_output = "yes\nno"
    run_pie_test_case("../p00663.py", input_content, expected_output)
