from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p02874_0():
    input_content = "4\n4 7\n1 4\n5 8\n2 5"
    expected_output = "6"
    run_pie_test_case("../p02874.py", input_content, expected_output)


def test_problem_p02874_1():
    input_content = "4\n1 20\n2 19\n3 18\n4 17"
    expected_output = "34"
    run_pie_test_case("../p02874.py", input_content, expected_output)


def test_problem_p02874_2():
    input_content = "4\n4 7\n1 4\n5 8\n2 5"
    expected_output = "6"
    run_pie_test_case("../p02874.py", input_content, expected_output)


def test_problem_p02874_3():
    input_content = "10\n457835016 996058008\n456475528 529149798\n455108441 512701454\n455817105 523506955\n457368248 814532746\n455073228 459494089\n456651538 774276744\n457667152 974637457\n457293701 800549465\n456580262 636471526"
    expected_output = "540049931"
    run_pie_test_case("../p02874.py", input_content, expected_output)
