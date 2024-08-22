# Codeflash

Codeflash is an AI optimization tool that automatically improves the performance of your Python code while maintaining its correctness.

![img.png](https://tacitassets.blob.core.windows.net/website-assets/readme-img.png)

## Features

- Automatically optimizes your code using AI
- Maintains code correctness through extensive testing
- Opens a pull request for each optimization
- Continuously optimizes your codebase through CI
- Dynamically optimizes your real workflows through tracing

## Installation

To install Codeflash, run:

```
pip install codeflash
```

## Quick Start

1. Configure Codeflash for your project:
   ```
   codeflash init
   ```

2. Optimize a function:
   ```
   codeflash --file path/to/your/file.py --function function_name
   ```
   
3. Optimize your entire codebase:
   ```
   codeflash --all
   ```

## Getting the Best Results

To get the most out of Codeflash:

1. Install the Github App and actions workflow
2. Find optimizations on your whole codebase with codeflash --all
3. Find and optimize bottlenecks with the Codeflash Tracer
4. Review the PRs Codeflash opens


## Learn More

- [Codeflash Website](https://www.codeflash.ai)
- [Documentation](https://docs.codeflash.ai)

## License

Codeflash is licensed under the BSL-1.1 License. See the LICENSE file for details.
