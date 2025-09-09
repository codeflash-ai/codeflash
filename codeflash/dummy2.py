# import random
# import time

# from line_profiler import LineProfiler

# codeflash_line_profile = LineProfiler()


# @codeflash_line_profile
# def complex_function(n: int) -> float:
#     """A function with nested loops, math operations, and simulated I/O."""

#     def helper_compute(x: float) -> float:
#         """Nested helper doing some CPU-heavy math."""
#         total = 0.0
#         for i in range(1, 5000):
#             total += math.sin(x * i) ** 2 + math.cos(x / (i + 1)) ** 3
#         return total

#     def helper_io_simulation():
#         """Simulate I/O-bound operations with tiny sleeps."""
#         for _ in range(5):
#             time.sleep(0.001 * random.random())  # sleep for 0-1ms

#     result = 0.0
#     for i in range(1, n + 1):
#         val = helper_compute(i)
#         helper_io_simulation()
#         result += val / (i + 1)

#     # some post-processing
#     result = math.sqrt(result) + math.log1p(result)
#     return result

# complex_function(10)
# with open("foo.lprof", "w") as f:
#     codeflash_line_profile.print_stats(f)

# import functools
# from pathlib import Path

# from line_profiler import LineProfiler


# class LineProfilerDecorator:
#     """Decorator class that profiles multiple functions and saves stats automatically."""
#     def __init__(self, output_file: str | Path):
#         self.output_file = Path(output_file)
#         self.profiler = LineProfiler()
#         # Ensure parent folder exists
#         self.output_file.parent.mkdir(parents=True, exist_ok=True)

#     def __call__(self, func):
#         """Decorate a function to profile it and save stats after execution."""
#         self.profiler.add_function(func)

#         @functools.wraps(func)
#         def wrapper(*args, **kwargs):
#             try:
#                 return self.profiler(func)(*args, **kwargs)
#             finally:
#                 # Save stats after each call
#                 with self.output_file.open("w") as f:
#                     self.profiler.print_stats(f)

#         return wrapper

# codeflash_line_profile = LineProfilerDecorator("baseline_lprof")

# @codeflash_line_profile
# def foo():
#     for i in range(100):
#         print("hello world")

# if __name__ == "__main__":
#     foo()


