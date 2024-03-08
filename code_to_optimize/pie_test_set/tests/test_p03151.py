from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03151_0():
    input_content = "3\n2 3 5\n3 4 1"
    expected_output = "3"
    run_pie_test_case("../p03151.py", input_content, expected_output)


def test_problem_p03151_1():
    input_content = "12\n757232153 372327760 440075441 195848680 354974235 458054863 463477172 740174259 615762794 632963102 529866931 64991604\n74164189 98239366 465611891 362739947 147060907 118867039 63189252 78303147 501410831 110823640 122948912 572905212"
    expected_output = "5"
    run_pie_test_case("../p03151.py", input_content, expected_output)


def test_problem_p03151_2():
    input_content = "3\n2 3 5\n3 4 1"
    expected_output = "3"
    run_pie_test_case("../p03151.py", input_content, expected_output)


def test_problem_p03151_3():
    input_content = "3\n17 7 1\n25 6 14"
    expected_output = "-1"
    run_pie_test_case("../p03151.py", input_content, expected_output)


def test_problem_p03151_4():
    input_content = "3\n2 3 3\n2 2 1"
    expected_output = "0"
    run_pie_test_case("../p03151.py", input_content, expected_output)
