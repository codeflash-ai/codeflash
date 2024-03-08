from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p01336_0():
    input_content = "3 10\nDobkeradops\n7 5 23 10\nPataPata\n1 1 2 1\ndop\n5 3 11 14\n2 300\nBydo System Alpha\n7 11 4 7\nGreen Inferno\n300 300 300 300"
    expected_output = "29\n462"
    run_pie_test_case("../p01336.py", input_content, expected_output)


def test_problem_p01336_1():
    input_content = "3 10\nDobkeradops\n7 5 23 10\nPataPata\n1 1 2 1\ndop\n5 3 11 14\n2 300\nBydo System Alpha\n7 11 4 7\nGreen Inferno\n300 300 300 300"
    expected_output = "29\n462"
    run_pie_test_case("../p01336.py", input_content, expected_output)
