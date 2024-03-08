from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p02381_0():
    input_content = "5\n70 80 100 90 20\n3\n80 80 80\n0"
    expected_output = "27.85677655\n0.00000000"
    run_pie_test_case("../p02381.py", input_content, expected_output)


def test_problem_p02381_1():
    input_content = "5\n70 80 100 90 20\n3\n80 80 80\n0"
    expected_output = "27.85677655\n0.00000000"
    run_pie_test_case("../p02381.py", input_content, expected_output)
