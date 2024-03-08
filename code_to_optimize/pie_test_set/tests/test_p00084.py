from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p00084_0():
    input_content = "Rain, rain, go to Spain."
    expected_output = "Rain rain Spain"
    run_pie_test_case("../p00084.py", input_content, expected_output)


def test_problem_p00084_1():
    input_content = "Win today's preliminary contest and be qualified to visit University of Aizu."
    expected_output = "Win and visit Aizu"
    run_pie_test_case("../p00084.py", input_content, expected_output)


def test_problem_p00084_2():
    input_content = "Rain, rain, go to Spain."
    expected_output = "Rain rain Spain"
    run_pie_test_case("../p00084.py", input_content, expected_output)
