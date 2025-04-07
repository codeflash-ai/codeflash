---
sidebar_position: 2
---

# Automate Optimization of Pull Requests
<!--- TODO: Add more pictures to guide better --->

Codeflash can automatically optimize your code when new pull requests are opened.

To be able to scan new code for performance optimizations, Codeflash requires a GitHub action workflow to 
be installed which runs the Codeflash optimization logic on every new pull request.
If the action workflow finds an optimization, it communicates with the Codeflash GitHub 
App through our secure servers and asks it to suggest new changes to the pull request.

This is the most useful way of using Codeflash, where you set it up once and all your new code gets optimized.
So setting this up is highly recommended.

## Prerequisites
- You have a Codeflash API key. If you don't have one, you can generate one from the [Codeflash Webapp](https://app.codeflash.ai/). Make sure you generate the API key with the right GitHub account that has access to the repository you want to optimize.
- You have completed [local installation](/docs/docs/getting-started/local-installation.md) steps and have a Python project with a `pyproject.toml` file that is configured with Codeflash. If you haven't configured Codeflash for your project yet, you can do so by running `codeflash init` in the root directory of your project.

## Add the Codeflash GitHub Actions workflow

### Guided setup

To add the Codeflash GitHub Actions workflow to your repository, you can run the following command in your project directory:

```bash
codeflash init-actions
```

This will walk you through the process of adding the Codeflash GitHub Actions workflow to your repository.

### All Set up!

Open a new PR to your GitHub project, and you will now see a new actions workflow for Codeflash run. If it finds an optimization,
codeflash-ai bot will comment on your repo with the optimization suggestions.

### Manual Installation (optional)
If you prefer to install the GitHub actions manually, follow the steps below -

#### Add the workflow file
Create a new file in your repository at `.github/workflows/codeflash-optimize.yaml` with the following contents:


```yaml title=".github/workflows/codeflash-optimize.yaml"
name: Codeflash

on:
  pull_request:
  workflow_dispatch:

jobs:
  optimize:
    name: Optimize new code in this PR
    if: ${{ github.actor != 'codeflash-ai[bot]' }}
    runs-on: ubuntu-latest
    env:
      CODEFLASH_API_KEY: ${{ secrets.CODEFLASH_API_KEY }}
      CODEFLASH_PR_NUMBER: ${{ github.event.number }}
    steps:
      - uses: actions/checkout@v4
        with:
          fetch-depth: 0
          token: ${{ secrets.GITHUB_TOKEN }}
      # TODO: Replace the following with your project's Python installation method
      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.11'
      # TODO: Replace the following with your project's dependency installation method
      - name: Install Project Dependencies
        run: |
          python -m pip install --upgrade pip
        # TODO: Replace the following with your project setup method
          pip install -r requirements.txt
          pip install codeflash
      - name: Run Codeflash to optimize code
        id: optimize_code
        run: |
          codeflash
```
You would need to fill in the `#TODO`s in the file above to make it work. Please commit this file to your repository.
If you use a particular Python package manager like Poetry or uv, some helpful configurations are provided below.

#### Config with different Python package managers

The yaml config above is a basic template. Here is how you can run Codeflash with the different Python package managers:

1. Poetry

```yaml
      - name: Install Project Dependencies
        run: |
          python -m pip install --upgrade pip
          pip install poetry
          poetry install --with dev
      - name: Run Codeflash to optimize code
        id: optimize_code
        run: |
          poetry env use python 
          poetry run codeflash
```
This assumes that you install poetry with pip and have Codeflash dependency in the `dev` section of your `pyproject.toml` file.

2. uv

```yaml
      - uses: astral-sh/setup-uv@v4
        with:
          enable-cache: true
      - run: uv sync --group=dev
      - name: Run Codeflash to optimize code
        run: uv run codeflash
```

#### Add your API key to your repository secrets

Go to your GitHub repository, click **Settings**, and click on **Secrets and
Variables** -> **Actions** on the left sidebar.

Add the following secret:

- `CODEFLASH_API_KEY`: The API key you got from https://app.codeflash.ai/app/apikeys

