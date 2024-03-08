from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03853_0():
    input_content = "2 2\n*.\n.*"
    expected_output = "*.\n*.\n.*\n.*"
    run_pie_test_case("../p03853.py", input_content, expected_output)


def test_problem_p03853_1():
    input_content = "9 20\n.....***....***.....\n....*...*..*...*....\n...*.....**.....*...\n...*.....*......*...\n....*.....*....*....\n.....**..*...**.....\n.......*..*.*.......\n........**.*........\n.........**........."
    expected_output = ".....***....***.....\n.....***....***.....\n....*...*..*...*....\n....*...*..*...*....\n...*.....**.....*...\n...*.....**.....*...\n...*.....*......*...\n...*.....*......*...\n....*.....*....*....\n....*.....*....*....\n.....**..*...**.....\n.....**..*...**.....\n.......*..*.*.......\n.......*..*.*.......\n........**.*........\n........**.*........\n.........**.........\n.........**........."
    run_pie_test_case("../p03853.py", input_content, expected_output)


def test_problem_p03853_2():
    input_content = "2 2\n*.\n.*"
    expected_output = "*.\n*.\n.*\n.*"
    run_pie_test_case("../p03853.py", input_content, expected_output)


def test_problem_p03853_3():
    input_content = "1 4\n***."
    expected_output = "***.\n***."
    run_pie_test_case("../p03853.py", input_content, expected_output)
