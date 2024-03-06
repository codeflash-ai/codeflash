import sys



read = sys.stdin.buffer.read

readline = sys.stdin.buffer.readline

readlines = sys.stdin.buffer.readlines



N = int(read())

x = N + (-N) % 1000

print((x - N))