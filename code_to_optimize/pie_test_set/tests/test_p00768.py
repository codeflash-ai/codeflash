from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p00768_0():
    input_content = "300 10 8 5\n50 5 2 1\n70 5 2 0\n75 1 1 0\n100 3 1 0\n150 3 2 0\n240 5 5 7\n50 1 1 0\n60 2 2 0\n70 2 3 0\n90 1 3 0\n120 3 5 0\n140 4 1 0\n150 2 4 1\n180 3 5 4\n15 2 2 1\n20 2 2 1\n25 2 2 0\n60 1 1 0\n120 5 5 4\n15 5 4 1\n20 5 4 0\n40 1 1 0\n40 2 2 0\n120 2 3 4\n30 1 1 0\n40 2 1 0\n50 2 2 0\n60 1 2 0\n120 3 3 2\n0 1 1 0\n1 2 2 0\n300 5 8 0\n0 0 0 0"
    expected_output = "3,1,5,10=9=8=7=6=4=2\n2,1,3,4,5\n1,2,3\n5=2=1,4=3\n2=1\n1,2,3\n5=4=3=2=1"
    run_pie_test_case("../p00768.py", input_content, expected_output)


def test_problem_p00768_1():
    input_content = "300 10 8 5\n50 5 2 1\n70 5 2 0\n75 1 1 0\n100 3 1 0\n150 3 2 0\n240 5 5 7\n50 1 1 0\n60 2 2 0\n70 2 3 0\n90 1 3 0\n120 3 5 0\n140 4 1 0\n150 2 4 1\n180 3 5 4\n15 2 2 1\n20 2 2 1\n25 2 2 0\n60 1 1 0\n120 5 5 4\n15 5 4 1\n20 5 4 0\n40 1 1 0\n40 2 2 0\n120 2 3 4\n30 1 1 0\n40 2 1 0\n50 2 2 0\n60 1 2 0\n120 3 3 2\n0 1 1 0\n1 2 2 0\n300 5 8 0\n0 0 0 0"
    expected_output = "3,1,5,10=9=8=7=6=4=2\n2,1,3,4,5\n1,2,3\n5=2=1,4=3\n2=1\n1,2,3\n5=4=3=2=1"
    run_pie_test_case("../p00768.py", input_content, expected_output)
