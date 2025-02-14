---
sidebar_position: 1
---

# Local Installation

Codeflash is installed and configured on a per-project basis.

You can install Codeflash locally for a project by running the following command in the project's virtual environment:

```bash
pip install codeflash
```

:::tip[Codeflash is a Development Dependency]
We recommend installing Codeflash as a development dependency.
It doesn't need to be installed as part of your package requirements. 
Codeflash is intended to be used locally and as part of development workflows such as CI.
:::

## Generate a Codeflash API Key

Since Codeflash uses advanced Large Language Models (LLMs) that are hosted in the cloud, you will need to generate an API key to use Codeflash.

To generate an API key, visit the [Codeflash Web App](https://app.codeflash.ai/) and sign up for an account with GitHub login.
<!--- TODO: Do we ask for access to specific repositories here? --->
Once you have signed up, you will be able to generate an API key from the [API Key](https://app.codeflash.ai/app/apikeys) page.
You will need the API key in the next step.

## Automatic Configuration

To configure Codeflash for a project, in the root directory of your project where your `pyproject.toml` file is located, run the following command :

```bash
codeflash init
```

If you don't have a pyproject.toml file yet, the codeflash init command will ask you to create one

:::tip[What's pyproject.toml?]
`pyproject.toml` is a configuration file that is used to specify build tool settings for Python projects. 
pyproject.toml is the modern replacement for setup.py and requirements.txt files.
It's the new standard for Python package metadata.
:::

When running `codeflash init`, you will see the following prompts:

```text
1. Enter your Codeflash API key: 
2. Which Python module do you want me to optimize going forward? (e.g. my_module)
3. Where are your tests located? (e.g. tests/)
4. Which test framework do you use? (pytest/unittest)
```

After you have answered these questions, Codeflash will be configured for your project.
The configuration will be saved in the `pyproject.toml` file in the root directory of your project.
To understand the configuration options, and set more advanced options, see the [Configuration](/configuration) page.

## Install the Codeflash GitHub App

Finally, if you have not done so already, Codeflash will ask you to install the Github App in your repository.  The Codeflash GitHub App allows access to your repository to the codeflash-ai bot to open PRs, review code, and provide optimization suggestions.

Please [install the Codeflash GitHub
app](https://github.com/apps/codeflash-ai/installations/select_target) by choosing the repository you want to install
Codeflash on.
## 
