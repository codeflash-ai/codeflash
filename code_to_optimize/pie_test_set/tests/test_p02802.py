from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p02802_0():
    input_content = "2 5\n1 WA\n1 AC\n2 WA\n2 AC\n2 WA"
    expected_output = "2 2"
    run_pie_test_case("../p02802.py", input_content, expected_output)


def test_problem_p02802_1():
    input_content = "6 0"
    expected_output = "0 0"
    run_pie_test_case("../p02802.py", input_content, expected_output)


def test_problem_p02802_2():
    input_content = "2 5\n1 WA\n1 AC\n2 WA\n2 AC\n2 WA"
    expected_output = "2 2"
    run_pie_test_case("../p02802.py", input_content, expected_output)


def test_problem_p02802_3():
    input_content = "100000 3\n7777 AC\n7777 AC\n7777 AC"
    expected_output = "1 0"
    run_pie_test_case("../p02802.py", input_content, expected_output)
