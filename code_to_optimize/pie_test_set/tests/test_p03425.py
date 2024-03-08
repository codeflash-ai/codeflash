from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03425_0():
    input_content = "5\nMASHIKE\nRUMOI\nOBIRA\nHABORO\nHOROKANAI"
    expected_output = "2"
    run_pie_test_case("../p03425.py", input_content, expected_output)


def test_problem_p03425_1():
    input_content = "5\nMASHIKE\nRUMOI\nOBIRA\nHABORO\nHOROKANAI"
    expected_output = "2"
    run_pie_test_case("../p03425.py", input_content, expected_output)


def test_problem_p03425_2():
    input_content = "5\nCHOKUDAI\nRNG\nMAKOTO\nAOKI\nRINGO"
    expected_output = "7"
    run_pie_test_case("../p03425.py", input_content, expected_output)


def test_problem_p03425_3():
    input_content = "4\nZZ\nZZZ\nZ\nZZZZZZZZZZ"
    expected_output = "0"
    run_pie_test_case("../p03425.py", input_content, expected_output)
