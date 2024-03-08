from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p00043_0():
    input_content = (
        "3649596966777\n6358665788577\n9118992346175\n9643871425498\n7755542764533\n1133557799246"
    )
    expected_output = "2 3 5 8\n3 4\n1 2 3 4 5 6 7 8 9\n7 8 9\n1 2 3 4 6 7 8\n0"
    run_pie_test_case("../p00043.py", input_content, expected_output)


def test_problem_p00043_1():
    input_content = (
        "3649596966777\n6358665788577\n9118992346175\n9643871425498\n7755542764533\n1133557799246"
    )
    expected_output = "2 3 5 8\n3 4\n1 2 3 4 5 6 7 8 9\n7 8 9\n1 2 3 4 6 7 8\n0"
    run_pie_test_case("../p00043.py", input_content, expected_output)
