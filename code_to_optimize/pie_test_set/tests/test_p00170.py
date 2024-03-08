from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p00170_0():
    input_content = "4\nsandwich 80 120\napple 50 200\ncheese 20 40\ncake 100 100\n9\nonigiri 80 300\nonigiri 80 300\nanpan 70 280\nmikan 50 80\nkanzume 100 500\nchocolate 50 350\ncookie 30 80\npurin 40 400\ncracker 40 160\n0"
    expected_output = "apple\ncake\nsandwich\ncheese\nkanzume\npurin\nchocolate\nonigiri\nonigiri\nanpan\nmikan\ncracker\ncookie"
    run_pie_test_case("../p00170.py", input_content, expected_output)


def test_problem_p00170_1():
    input_content = "4\nsandwich 80 120\napple 50 200\ncheese 20 40\ncake 100 100\n9\nonigiri 80 300\nonigiri 80 300\nanpan 70 280\nmikan 50 80\nkanzume 100 500\nchocolate 50 350\ncookie 30 80\npurin 40 400\ncracker 40 160\n0"
    expected_output = "apple\ncake\nsandwich\ncheese\nkanzume\npurin\nchocolate\nonigiri\nonigiri\nanpan\nmikan\ncracker\ncookie"
    run_pie_test_case("../p00170.py", input_content, expected_output)
