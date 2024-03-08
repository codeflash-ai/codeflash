from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p02830_0():
    input_content = "2\nip cc"
    expected_output = "icpc"
    run_pie_test_case("../p02830.py", input_content, expected_output)


def test_problem_p02830_1():
    input_content = "2\nip cc"
    expected_output = "icpc"
    run_pie_test_case("../p02830.py", input_content, expected_output)


def test_problem_p02830_2():
    input_content = "8\nhmhmnknk uuuuuuuu"
    expected_output = "humuhumunukunuku"
    run_pie_test_case("../p02830.py", input_content, expected_output)


def test_problem_p02830_3():
    input_content = "5\naaaaa aaaaa"
    expected_output = "aaaaaaaaaa"
    run_pie_test_case("../p02830.py", input_content, expected_output)
