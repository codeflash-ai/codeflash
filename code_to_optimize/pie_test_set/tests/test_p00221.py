from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p00221_0():
    input_content = "5 7\n1\n2\nFizz\n4\nBuzz\n6\n7\n3 5\n1\n2\n3\n4\n5\n0 0"
    expected_output = "2 3 4 5\n1"
    run_pie_test_case("../p00221.py", input_content, expected_output)


def test_problem_p00221_1():
    input_content = "5 7\n1\n2\nFizz\n4\nBuzz\n6\n7\n3 5\n1\n2\n3\n4\n5\n0 0"
    expected_output = "2 3 4 5\n1"
    run_pie_test_case("../p00221.py", input_content, expected_output)
