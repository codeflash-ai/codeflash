from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p02914_0():
    input_content = "3\n3 6 5"
    expected_output = "12"
    run_pie_test_case("../p02914.py", input_content, expected_output)


def test_problem_p02914_1():
    input_content = "20\n1008288677408720767 539403903321871999 1044301017184589821 215886900497862655 504277496111605629 972104334925272829 792625803473366909 972333547668684797 467386965442856573 755861732751878143 1151846447448561405 467257771752201853 683930041385277311 432010719984459389 319104378117934975 611451291444233983 647509226592964607 251832107792119421 827811265410084479 864032478037725181"
    expected_output = "2012721721873704572"
    run_pie_test_case("../p02914.py", input_content, expected_output)


def test_problem_p02914_2():
    input_content = "3\n3 6 5"
    expected_output = "12"
    run_pie_test_case("../p02914.py", input_content, expected_output)


def test_problem_p02914_3():
    input_content = "4\n23 36 66 65"
    expected_output = "188"
    run_pie_test_case("../p02914.py", input_content, expected_output)
