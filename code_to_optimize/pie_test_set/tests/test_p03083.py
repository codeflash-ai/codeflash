from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03083_0():
    input_content = "2 1"
    expected_output = "500000004\n750000006\n750000006"
    run_pie_test_case("../p03083.py", input_content, expected_output)


def test_problem_p03083_1():
    input_content = "2 1"
    expected_output = "500000004\n750000006\n750000006"
    run_pie_test_case("../p03083.py", input_content, expected_output)


def test_problem_p03083_2():
    input_content = "3 2"
    expected_output = "500000004\n500000004\n625000005\n187500002\n187500002"
    run_pie_test_case("../p03083.py", input_content, expected_output)


def test_problem_p03083_3():
    input_content = "6 9"
    expected_output = "500000004\n500000004\n500000004\n500000004\n500000004\n500000004\n929687507\n218750002\n224609377\n303710940\n633300786\n694091802\n172485353\n411682132\n411682132"
    run_pie_test_case("../p03083.py", input_content, expected_output)
