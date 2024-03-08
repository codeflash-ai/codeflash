from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p01359_0():
    input_content = "4 3\nmeiji 10 1877\ntaisho 6 1917\nshowa 62 1987\nheisei 22 2010\n1868\n1917\n1988\n1 1\nuniversalcentury 123 2168\n2010\n0 0"
    expected_output = "meiji 1\ntaisho 6\nUnknown\nUnknown"
    run_pie_test_case("../p01359.py", input_content, expected_output)


def test_problem_p01359_1():
    input_content = "4 3\nmeiji 10 1877\ntaisho 6 1917\nshowa 62 1987\nheisei 22 2010\n1868\n1917\n1988\n1 1\nuniversalcentury 123 2168\n2010\n0 0"
    expected_output = "meiji 1\ntaisho 6\nUnknown\nUnknown"
    run_pie_test_case("../p01359.py", input_content, expected_output)
