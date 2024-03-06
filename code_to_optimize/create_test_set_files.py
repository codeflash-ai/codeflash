import json

with open("test.jsonl", "r") as file:
    data = file.readlines()

for line in data:
    line = json.loads(line)
    problem_id = line["problem_id"].strip('"')
    code = line["input"].strip('"')
    file = open(f"{problem_id}.py", "w")
    file.write(code)
    file.close()
    print(f"Created file {problem_id}.py")
