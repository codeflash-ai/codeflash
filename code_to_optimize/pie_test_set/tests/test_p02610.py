from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p02610_0():
    input_content = "3\n2\n1 5 10\n2 15 5\n3\n2 93 78\n1 71 59\n3 57 96\n19\n19 23 16\n5 90 13\n12 85 70\n19 67 78\n12 16 60\n18 48 28\n5 4 24\n12 97 97\n4 57 87\n19 91 74\n18 100 76\n7 86 46\n9 100 57\n3 76 73\n6 84 93\n1 6 84\n11 75 94\n19 15 3\n12 11 34"
    expected_output = "25\n221\n1354"
    run_pie_test_case("../p02610.py", input_content, expected_output)


def test_problem_p02610_1():
    input_content = "3\n2\n1 5 10\n2 15 5\n3\n2 93 78\n1 71 59\n3 57 96\n19\n19 23 16\n5 90 13\n12 85 70\n19 67 78\n12 16 60\n18 48 28\n5 4 24\n12 97 97\n4 57 87\n19 91 74\n18 100 76\n7 86 46\n9 100 57\n3 76 73\n6 84 93\n1 6 84\n11 75 94\n19 15 3\n12 11 34"
    expected_output = "25\n221\n1354"
    run_pie_test_case("../p02610.py", input_content, expected_output)
