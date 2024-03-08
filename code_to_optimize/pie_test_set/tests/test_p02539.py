from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p02539_0():
    input_content = "2\n1\n1\n2\n3"
    expected_output = "2"
    run_pie_test_case("../p02539.py", input_content, expected_output)


def test_problem_p02539_1():
    input_content = "2\n1\n1\n2\n3"
    expected_output = "2"
    run_pie_test_case("../p02539.py", input_content, expected_output)


def test_problem_p02539_2():
    input_content = "5\n30\n10\n20\n40\n20\n10\n10\n30\n50\n60"
    expected_output = "516"
    run_pie_test_case("../p02539.py", input_content, expected_output)
