from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p02551_0():
    input_content = "5 5\n1 3\n2 3\n1 4\n2 2\n1 2"
    expected_output = "1"
    run_pie_test_case("../p02551.py", input_content, expected_output)


def test_problem_p02551_1():
    input_content = "5 5\n1 3\n2 3\n1 4\n2 2\n1 2"
    expected_output = "1"
    run_pie_test_case("../p02551.py", input_content, expected_output)


def test_problem_p02551_2():
    input_content = "200000 0"
    expected_output = "39999200004"
    run_pie_test_case("../p02551.py", input_content, expected_output)


def test_problem_p02551_3():
    input_content = "176527 15\n1 81279\n2 22308\n2 133061\n1 80744\n2 44603\n1 170938\n2 139754\n2 15220\n1 172794\n1 159290\n2 156968\n1 56426\n2 77429\n1 97459\n2 71282"
    expected_output = "31159505795"
    run_pie_test_case("../p02551.py", input_content, expected_output)
