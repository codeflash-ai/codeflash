from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03388_0():
    input_content = "8\n1 4\n10 5\n3 3\n4 11\n8 9\n22 40\n8 36\n314159265 358979323"
    expected_output = "1\n12\n4\n11\n14\n57\n31\n671644785"
    run_pie_test_case("../p03388.py", input_content, expected_output)


def test_problem_p03388_1():
    input_content = "8\n1 4\n10 5\n3 3\n4 11\n8 9\n22 40\n8 36\n314159265 358979323"
    expected_output = "1\n12\n4\n11\n14\n57\n31\n671644785"
    run_pie_test_case("../p03388.py", input_content, expected_output)
