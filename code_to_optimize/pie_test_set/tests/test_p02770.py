from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p02770_0():
    input_content = "3 1\n3 1 4\n5 3 2"
    expected_output = "1"
    run_pie_test_case("../p02770.py", input_content, expected_output)


def test_problem_p02770_1():
    input_content = "3 1\n3 1 4\n5 3 2"
    expected_output = "1"
    run_pie_test_case("../p02770.py", input_content, expected_output)


def test_problem_p02770_2():
    input_content = (
        "7 3\n27 18 28 18 28 46 1000000000\n1000000000 1 7\n1000000000 2 10\n1000000000 3 12"
    )
    expected_output = "224489796\n214285714\n559523809"
    run_pie_test_case("../p02770.py", input_content, expected_output)
