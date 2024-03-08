from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p02538_0():
    input_content = "8 5\n3 6 2\n1 4 7\n3 8 3\n2 2 2\n4 5 1"
    expected_output = "11222211\n77772211\n77333333\n72333333\n72311333"
    run_pie_test_case("../p02538.py", input_content, expected_output)


def test_problem_p02538_1():
    input_content = "8 5\n3 6 2\n1 4 7\n3 8 3\n2 2 2\n4 5 1"
    expected_output = "11222211\n77772211\n77333333\n72333333\n72311333"
    run_pie_test_case("../p02538.py", input_content, expected_output)


def test_problem_p02538_2():
    input_content = "200000 1\n123 456 7"
    expected_output = "641437905"
    run_pie_test_case("../p02538.py", input_content, expected_output)
