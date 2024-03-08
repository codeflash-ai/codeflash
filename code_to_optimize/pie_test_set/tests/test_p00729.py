from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p00729_0():
    input_content = "4 2\n10\n775 1 1 1\n780 4 2 1\n790 2 1 1\n800 2 1 0\n810 3 1 1\n820 1 1 0\n825 3 1 0\n860 1 1 1\n870 4 2 0\n880 1 1 0\n1\n780 870 1\n13 15\n12\n540 12 13 1\n600 12 13 0\n650 13 15 1\n660 12 15 1\n665 11 13 1\n670 13 15 0\n675 11 13 0\n680 12 15 0\n1000 11 14 1\n1060 12 14 1\n1060 11 14 0\n1080 12 14 0\n3\n540 700 13\n600 1000 15\n1000 1200 11\n1 1\n2\n600 1 1 1\n700 1 1 0\n5\n540 600 1\n550 650 1\n610 620 1\n650 750 1\n700 800 1\n0 0"
    expected_output = "55\n70\n30\n0\n0\n50\n10\n50\n0"
    run_pie_test_case("../p00729.py", input_content, expected_output)


def test_problem_p00729_1():
    input_content = "4 2\n10\n775 1 1 1\n780 4 2 1\n790 2 1 1\n800 2 1 0\n810 3 1 1\n820 1 1 0\n825 3 1 0\n860 1 1 1\n870 4 2 0\n880 1 1 0\n1\n780 870 1\n13 15\n12\n540 12 13 1\n600 12 13 0\n650 13 15 1\n660 12 15 1\n665 11 13 1\n670 13 15 0\n675 11 13 0\n680 12 15 0\n1000 11 14 1\n1060 12 14 1\n1060 11 14 0\n1080 12 14 0\n3\n540 700 13\n600 1000 15\n1000 1200 11\n1 1\n2\n600 1 1 1\n700 1 1 0\n5\n540 600 1\n550 650 1\n610 620 1\n650 750 1\n700 800 1\n0 0"
    expected_output = "55\n70\n30\n0\n0\n50\n10\n50\n0"
    run_pie_test_case("../p00729.py", input_content, expected_output)
