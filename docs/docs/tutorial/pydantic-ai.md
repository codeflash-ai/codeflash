---
sidebar_position: 1
---

## Optimizing Pydantic AI

In this how-to guide, we'll walk through optimizing the Pydantic AI framework
for performance using Codeflash. After following these steps, you will have found tens of 
optmizations in the Pydantic AI framework like this one: [Pydantic AI - Optimized](https://github.com/misrasaurabh1/pydantic-ai/pulls)

Estimated time: 5 minutes

### Step 1: Fork the Pydantic AI Repository
Fork the [Pydantic AI repository](https://github.com/pydantic/pydantic-ai) on GitHub.

### Step 2: Clone the Forked Repository
Clone the forked repository to your local machine or a development server.

using https - `git clone https://github.com/your-username/pydantic-ai.git`

using ssh - `git clone git@github.com:your-username/pydantic-ai.git`

### Step 3: Install Pydantic AI Python Environment

The instructions below follow the [official Contributing instructions](https://ai.pydantic.dev/contributing/)

Pydantic uses uv package manager. If it is not installed, install it from the [offical instructions](https://docs.astral.sh/uv/getting-started/installation/)

Pydantic also requires deno to be installed. If it is not installed, install it from the [offical instructions](https://docs.deno.com/runtime/getting_started/installation/)

After this run the following command to install the dependencies.
```bash
make install
```

To ensure that Pydantic is installed correctly, run the following command to run the tests:
```bash
make test
```
You should expect to see most of the tests pass.

### Step 4: Install Codeflash

```bash
uv pip install codeflash
```

### Step 5: Configure Codeflash

Run Codeflash init to configure Codeflash for this project.

```bash
uv run codeflash init
```

You will be prompted to enter your Codeflash API key that you can create on the [Codeflash Web App](https://app.codeflash.ai/app/apikeys).
Also ensure that you have installed the Codeflash Github app on your forked repository. You can install it [from here](https://github.com/apps/codeflash-ai/installations/select_target).
Since the Pydantic AI code is located in pydantic_ai_slim/pydantic_ai directory, when codeflash asks you to enter the module name, select "Enter a custom directory" and type `pydantic_ai_slim/pydantic_ai`

For the location of the tests, select `tests`
For the test framework, select `pytest`
For the formatter, select `ruff`
Installing the GitHub action for continuous optimization is optional, but you can skip it for this tutorial.

### Step 6: Run Codeflash Optimization

Now that you have setup Pydantic_ai and codeflash, finding optimizations for it is as simple as running the following command:

```bash
codeflash --all
```
This will find all the functions in the module, find all the existing tests available for it and start optimizing all the code function by function.

As it finds optimizations, it will keep opening Pull Requests for your review.

