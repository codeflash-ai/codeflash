---
sidebar_position: 4
---
# Using Benchmarks

Codeflash is able the determine the impact of an optimization on predefined benchmarks, when used in benchmark mode.

Benchmark mode is an easy way for users to define workflows that are performance-critical and need to be optimized.
For example, if a user has an important function that requires minimal latency, the user can define a benchmark for that function.
Codeflash will then calculate the impact (if any) of any optimization on the performance of that function.

## Using Codeflash in Benchmark Mode

1. **Create a benchmarks root** 

    Create a directory for benchmarks. This directory must be a sub directory of your tests directory.

   In your pyproject.toml, add the path to the 'benchmarks-root' section.
    ```yaml
    [tool.codeflash]
   # All paths are relative to this pyproject.toml's directory.
   module-root = "inference"
   tests-root = "tests"
   test-framework = "pytest"
   benchmarks-root = "tests/benchmarks" # add your benchmarks root dir here
   ignore-paths = []
   formatter-cmds = ["disabled"]
    ```
    
2. **Define your benchmarks**
    
   Currently, Codeflash only supports benchmarks written as pytest-benchmarks. Check out the [pytest-benchmark](https://pytest-benchmark.readthedocs.io/en/stable/index.html) documentation for more information on syntax.

   For example:
      
   ```python
   from core.bubble_sort import sorter

   def test_sort(benchmark):
       result = benchmark(sorter, list(reversed(range(500))))
       assert result == list(range(500))
   ```

   Note that these benchmarks should be defined in such a way that they don't take a long time to run.

   The pytest-benchmark format is simply used as an interface. The plugin is actually not used - Codeflash will run these benchmarks with its own pytest plugin


3. **Run Codeflash**

   Run Codeflash with the `--benchmark` flag. Note that benchmark mode cannot be used with `--all`. 

   ```bash
   codeflash --file test_file.py --benchmark
   ```
   
   If you did not define your benchmarks-root in your pyproject.toml, you can do:

   ```bash
   codeflash --file test_file.py --benchmark --benchmarks-root path/to/benchmarks
   ```
   

4. **Run Codeflash in CI**

   Benchmark mode is best used together with Codeflash as a Github Action. This way, with every PR, you will know the impact of Codeflash's optimizations on your benchmarks.
   
   Use `codeflash init` for an easy way to set up Codeflash as a Github Action (with the option to enable benchmark mode).



## How it works

1. Codeflash identifies benchmarks in the benchmarks-root directory.


2. The benchmarks are run so that runtime statistics and information can be recorded. 


3. Replay tests are generated so the performance of optimization candidates on the exact inputs used in the benchmarks can be measured.


4. If an optimization candidate is verified to be correct, the speedup of the optimization is calculated for each benchmark. 


5. Codeflash then reports the impact of the optimization on each benchmark. 


Using Codeflash with benchmarks is a great way to find optimizations that really matter.

Codeflash is actively working on this feature and will be adding new capabilities in the near future!