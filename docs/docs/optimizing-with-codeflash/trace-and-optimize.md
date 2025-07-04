---
sidebar_position: 4
---
# Optimize Workflows End-to-End.

Codeflash supports optimizing an entire Python script end-to-end by tracing the script's execution and generating Replay Tests. Tracing follows the execution of a script, profiles it and captures inputs to all called functions, allowing them to be replayed during optimization. Codeflash uses these Replay Tests to optimize all functions called in the script, starting from the most important ones.

To optimize a script, `python myscript.py`, replace `python` with `codeflash optimize` and run the following command:

```bash
codeflash optimize myscript.py
```

To optimize code within pytest tests that you could normally run like `python -m pytest tests/`, use this command:

```bash
codeflash optimize -m pytest tests/
```

This powerful command creates high-quality optimizations, making it ideal when you need to optimize a workflow or script. The initial tracing process can be slow, so try to limit your script's runtime to under 1 minute for best results. If your workflow is longer, consider tracing it into smaller sections by using the Codeflash tracer as a context manager (point 3 below). 

## What is the codeflash optimize command?

`codeflash optimize` tries to do everything that an expert engineer would do while optimizing a workflow. It profiles your code, traces the execution of your workflow and generates a set of test cases that are derived from how your code is actually run.
Codeflash Tracer works by recording the inputs of your functions as they are called in your codebase. These inputs are then used to generate test cases that are representative of the real-world usage of your functions.
We call these generated test cases "Replay Tests" because they replay the inputs that were recorded during the tracing phase.

Then, Codeflash Optimizer can use these replay tests to verify correctness and calculate accurate performance gains for the optimized functions.
Using Replay Tests, Codeflash can verify that the optimized functions produce the same output as the original function and also measure the performance gains of the optimized function on the real-world inputs.
This way you can be *sure* that the optimized function causes no changes of behavior for the traced workflow and also, that it is faster than the original function. To get more confidence on the correctness of the code, we also generate several LLM generated test cases and discover any existing unit cases you may have.

## Using codeflash optimize

Codeflash script optimizer can be used in three ways:

1. **As an integrated command** 

    If you run a Python script as follows
    
    ```bash
    python path/to/your/file.py --your_options
    ```
    
    You can start tracing and optimizing your code with the following command
    
    ```bash
    codeflash optimize path/to/your/file.py --your_options
    ```
    
    The above command should suffice in most situations. You can add a argument like `codeflash optimize -o trace_file_path.trace` if you want to customize the trace file location. Otherwise, it defaults to `codeflash.trace` in the current working directory. 
    
2. **Trace and optimize as two separate steps**
    
    If you want more control over the tracing and optimization process. You can trace first and then optimize with the replay tests later. Each replay test is associated with a trace file. 
    
    To first create just the trace file, run
    
    ```python
    codeflash optimize -o trace_file.trace --trace-only path/to/your/file.py --your_options
    ```
    
    This will create a replay test file. To optimize with the replay test, run the 
    
    ```bash
    codeflash --replay-test /path/to/test_replay_test_0.py
    ```
    
    More Options:
    - `--tracer-timeout`: The maximum time in seconds to trace the entire workflow. Default is indefinite. This is useful while tracing really long workflows.
3. **As a Context Manager -**
    
    To trace only very specific sections of your codeflash, You can also use the Codeflash Tracer as a context manager.
    You can wrap the code you want to trace in a `with` statement as follows -
    
    ```python
    from codeflash.tracer import Tracer
    
    with Tracer(output="codeflash.trace"):
        model.predict() # Your code here
    ```
    
    This is much faster than tracing the whole script. Sometimes, if tracing the whole script fails, then the Context Manager can also be used to trace the code sections. 
    
    After this finishes, you can optimize using the generated replay tests.
    
    ```bash
    codeflash --replay-test /path/to/test_replay_test_0.py
    ```
    
    More Options for the Tracer:
    
    - `disable`: If set to `True`, the tracer will not trace the code. Default is `False`.
    - `max_function_count`: The maximum number of times to trace a single function. More calls to a function will not be traced. Default is 100.
    - `timeout`: The maximum time in seconds to trace the entire workflow. Default is indefinite. This is useful while tracing really long workflows, to not wait indefinitely.
    - `output`: The file to save the trace to. Default is `codeflash.trace`.
    - `config_file_path`: The path to the `pyproject.toml` file which stores the Codeflash config. This is auto-discovered by default.
    You can also disable the tracer in the code by setting the `disable=True` option in the `Tracer` constructor.
