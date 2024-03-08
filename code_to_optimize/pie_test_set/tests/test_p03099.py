from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03099_0():
    input_content = (
        "7\n1 3 6\n1 5 9\n3 1 8\n4 3 8\n6 2 9\n5 4 11\n5 7 10\n4\nL 3 1\nR 2 3\nD 5 3\nU 4 2"
    )
    expected_output = "36"
    run_pie_test_case("../p03099.py", input_content, expected_output)


def test_problem_p03099_1():
    input_content = "10\n66 47 71040136000\n65 77 74799603000\n80 53 91192869000\n24 34 24931901000\n91 78 49867703000\n68 71 46108236000\n46 73 74799603000\n56 63 93122668000\n32 51 71030136000\n51 26 70912345000\n21\nL 51 1\nL 7 0\nU 47 4\nR 92 0\nR 91 1\nD 53 2\nR 65 3\nD 13 0\nU 63 3\nL 68 3\nD 47 1\nL 91 5\nR 32 4\nL 66 2\nL 80 4\nD 77 4\nU 73 1\nD 78 5\nU 26 5\nR 80 2\nR 24 5"
    expected_output = "305223377000"
    run_pie_test_case("../p03099.py", input_content, expected_output)


def test_problem_p03099_2():
    input_content = "3\n1 2 3\n4 5 6\n7 8 9\n1\nL 100 0"
    expected_output = "0"
    run_pie_test_case("../p03099.py", input_content, expected_output)


def test_problem_p03099_3():
    input_content = "4\n1 1 10\n1 2 11\n2 1 12\n2 2 13\n3\nL 8 3\nL 9 2\nL 10 1"
    expected_output = "13"
    run_pie_test_case("../p03099.py", input_content, expected_output)


def test_problem_p03099_4():
    input_content = (
        "7\n1 3 6\n1 5 9\n3 1 8\n4 3 8\n6 2 9\n5 4 11\n5 7 10\n4\nL 3 1\nR 2 3\nD 5 3\nU 4 2"
    )
    expected_output = "36"
    run_pie_test_case("../p03099.py", input_content, expected_output)
