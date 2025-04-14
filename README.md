![Codeflash-banner](https://i.postimg.cc/GmPRC52t/Codeflash-banner.png)
<p align="center">
   <a href="https://github.com/codeflash-ai/codeflash">
    <img src="https://img.shields.io/github/commit-activity/m/codeflash-ai/codeflash" alt="GitHub commit activity">
  </a>
  <a href="https://pypi.org/project/codeflash/">
    <img src="https://img.shields.io/pypi/dm/codeflash" alt="PyPI Downloads">
  </a>
  <a href="https://pypi.org/project/codeflash/">
    <img src="https://img.shields.io/pypi/v/codeflash?label=PyPI%20version" alt="PyPI Downloads">
  </a>
</p>

[Codeflash](https://www.codeflash.ai) is a general purpose optimizer for Python that automatically improves the performance of your Python code while maintaining its correctness. 

How Codeflash works:
1. LLMs generate multiple optimization candidates for your code
2. Codeflash tests the optimization candidates for correctness
3. Codeflash benchmarks the optimization candidates for performance

Should the optimization be valid and faster than the original code, Codeflash will create a pull request with the optimized code. You can now review and merge the code to make your codebase faster!

Ways to use Codeflash:
- Optimize an entire codebase by running `codeflash --all`
- Automatically optimize all __future__ code written by installing Codeflash as a GitHub action. Codeflash will try to optimize your new code before you merge it into the codebase.
- Optimize a Python workflow end-to-end by tracing the workflow.

Codeflash is used by top engineering teams at [Pydantic](https://github.com/pydantic/pydantic/pulls?q=is%3Apr+author%3Amisrasaurabh1+is%3Amerged), [Langflow](https://github.com/langflow-ai/langflow/issues?q=state%3Aclosed%20is%3Apr%20author%3Amisrasaurabh1), [Albumentations](https://github.com/albumentations-team/albumentations/issues?q=state%3Amerged%20is%3Apr%20author%3Akrrt7%20OR%20state%3Amerged%20is%3Apr%20author%3Aaseembits93%20) and many others to ship performant, expert level code.

Codeflash is great at optimizing AI Agents, Computer Vision algorithms, numerical code, backend code or anything else you might write with Python.


## Installation

To install Codeflash, run:

```
pip install codeflash
```
Add codeflash as a development time dependency if you are using package managers like uv or poetry.
## Quick Start


1. Run the following command at the root directory of your project where the pyproject.toml file is located
   ```
   codeflash init
   ```
   This will set up basic configurations for your project, eg:
   - Input a Codeflash API key (for access to LLMs)
   - Enable a [GitHub app](https://github.com/apps/codeflash-ai/installations/select_target) to open Pull Requests on the repo
   - [Optional] Setup a GitHub actions which will optimize all your future code.



2. Optimize a file:

   ```   
   codeflash --file <path/to/file.py>
   ```
3. Optimize your entire codebase: (This will run for a while and open PRs as it finds optimizations)
   ```   
   codeflash --all
   ```

## Documentation
For detailed installation and usage instructions, visit our documentation at [docs.codeflash.ai](https://docs.codeflash.ai)

## Demo


- Optimizing the performance of new code for a Pull Request through GitHub Actions. This lets you ship code quickly while ensuring it remains performant.

https://github.com/user-attachments/assets/38f44f4e-be1c-4f84-8db9-63d5ee3e61e5

## Support

Join our community for support and discussions. If you have any questions, feel free to reach out to us using one of the following methods:

- [Join our Discord](https://www.codeflash.ai/discord)
- [Follow us on Twitter](https://x.com/codeflashAI)
- [Follow us on Linkedin](https://www.linkedin.com/in/saurabh-misra/)
- [Email founders](mailto:saurabh@codeflash.ai)

## License

Codeflash is licensed under the BSL-1.1 License. See the LICENSE file for details.
