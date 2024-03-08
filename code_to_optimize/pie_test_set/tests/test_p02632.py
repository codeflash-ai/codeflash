from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p02632_0():
    input_content = "5\noof"
    expected_output = "575111451"
    run_pie_test_case("../p02632.py", input_content, expected_output)


def test_problem_p02632_1():
    input_content = "5\noof"
    expected_output = "575111451"
    run_pie_test_case("../p02632.py", input_content, expected_output)


def test_problem_p02632_2():
    input_content = "37564\nwhydidyoudesertme"
    expected_output = "318008117"
    run_pie_test_case("../p02632.py", input_content, expected_output)
