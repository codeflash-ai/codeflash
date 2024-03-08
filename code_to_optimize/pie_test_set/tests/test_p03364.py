from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03364_0():
    input_content = "2\nab\nca"
    expected_output = "2"
    run_pie_test_case("../p03364.py", input_content, expected_output)


def test_problem_p03364_1():
    input_content = "2\nab\nca"
    expected_output = "2"
    run_pie_test_case("../p03364.py", input_content, expected_output)


def test_problem_p03364_2():
    input_content = "5\nabcde\nfghij\nklmno\npqrst\nuvwxy"
    expected_output = "0"
    run_pie_test_case("../p03364.py", input_content, expected_output)


def test_problem_p03364_3():
    input_content = "4\naaaa\naaaa\naaaa\naaaa"
    expected_output = "16"
    run_pie_test_case("../p03364.py", input_content, expected_output)
