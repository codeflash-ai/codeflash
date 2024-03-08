from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03111_0():
    input_content = "5 100 90 80\n98\n40\n30\n21\n80"
    expected_output = "23"
    run_pie_test_case("../p03111.py", input_content, expected_output)


def test_problem_p03111_1():
    input_content = "8 1000 800 100\n300\n333\n400\n444\n500\n555\n600\n666"
    expected_output = "243"
    run_pie_test_case("../p03111.py", input_content, expected_output)


def test_problem_p03111_2():
    input_content = "5 100 90 80\n98\n40\n30\n21\n80"
    expected_output = "23"
    run_pie_test_case("../p03111.py", input_content, expected_output)


def test_problem_p03111_3():
    input_content = "8 100 90 80\n100\n100\n90\n90\n90\n80\n80\n80"
    expected_output = "0"
    run_pie_test_case("../p03111.py", input_content, expected_output)
