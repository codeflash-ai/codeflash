from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03036_0():
    input_content = "2 10 20"
    expected_output = "30\n50\n90\n170\n330\n650\n1290\n2570\n5130\n10250"
    run_pie_test_case("../p03036.py", input_content, expected_output)


def test_problem_p03036_1():
    input_content = "2 10 20"
    expected_output = "30\n50\n90\n170\n330\n650\n1290\n2570\n5130\n10250"
    run_pie_test_case("../p03036.py", input_content, expected_output)


def test_problem_p03036_2():
    input_content = "4 40 60"
    expected_output = "200\n760\n3000\n11960\n47800\n191160\n764600\n3058360\n12233400\n48933560"
    run_pie_test_case("../p03036.py", input_content, expected_output)
