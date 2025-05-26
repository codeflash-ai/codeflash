---
sidebar_position: 1
---

# Local Installation

Codeflash is installed and configured on a per-project basis.

You can install Codeflash locally for a project by running the following command in the project's virtual environment:

### Prerequisites

Before installing Codeflash, ensure you have:

1. **Python 3.8 or higher** installed
2. **A Python project** with a virtual environment
3. **Project dependencies installed** in your virtual environment
4. **Tests** (optional) for your code (Codeflash uses tests to verify optimizations)

:::important[Virtual Environment Required]
Always install Codeflash in your project's virtual environment, not globally. Make sure your virtual environment is activated before proceeding.

```bash
# Example: Activate your virtual environment
source venv/bin/activate  # On Linux/Mac
# or
venv\Scripts\activate     # On Windows
```
:::
### Step 1: Install Codeflash
```bash
pip install codeflash
```

:::tip[Codeflash is a Development Dependency]
We recommend installing Codeflash as a development dependency.
It doesn't need to be installed as part of your package requirements. 
Codeflash is intended to be used locally and as part of development workflows such as CI.
If using pyproject.toml:
```toml
[tool.poetry.dependencies.dev]
codeflash = "^latest"
```
Or with pip:
```bash
pip install --dev codeflash
````
:::

### Step 2: Generate a Codeflash API Key

Codeflash uses cloud-hosted AI models to optimize your code. You'll need an API key to use it.

1. Visit the [Codeflash Web App](https://app.codeflash.ai/) 
2. Sign up with your GitHub account (free)
3. Navigate to the [API Key](https://app.codeflash.ai/app/apikeys) page to generate your API key
<!--- TODO: Do we ask for access to specific repositories here? --->

:::note[Free Tier Available]
Codeflash offers a **free tier** with a limited number of optimizations per month. Perfect for trying it out or small projects!
:::

### Step 3: Automatic Configuration

Navigate to your project's root directory (where your `pyproject.toml` file is or should be) and run:

```bash
# Make sure you're in your project root
cd /path/to/your/project

# Run the initialization
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

### Step 4: Install the Codeflash GitHub App

<!--- TODO: Justify to users Why we need the user to install Github App even in local Installation or local optimization? --->
Finally, if you have not done so already, Codeflash will ask you to install the Github App in your repository.  The Codeflash GitHub App allows access to your repository to the codeflash-ai bot to open PRs, review code, and provide optimization suggestions.

Please [install the Codeflash GitHub
app](https://github.com/apps/codeflash-ai/installations/select_target) by choosing the repository you want to install
Codeflash on.
## 

## Try It Out!

Once configured, you can start optimizing your code:

```bash
# Optimize a specific function
codeflash --file path/to/your/file.py --function function_name

# Or if want to optimize only locally without creating a PR
codeflash --file path/to/your/file.py --function function_name --no-pr
```

### Example Project

Want to see Codeflash in action? Check out our example repository:

🔗 [github.com/codeflash-ai/optimize-me](https://github.com/codeflash-ai/optimize-me)

This repo includes:
- Sample Python code with performance issues
- Tests for verification
- Pre-configured `pyproject.toml`
- Before/after optimization examples in PRs

Clone it and try running:
```bash
git clone https://github.com/codeflash-ai/optimize-me.git
cd optimize-me
python -m venv .venv
source .venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt
pip install codeflash
codeflash init  # Use your own API key
codeflash --all # optimize the entire repo
```

### 🔧 Troubleshooting

#### 📦 "Module not found" errors
Make sure:
- ✅ Your virtual environment is activated
- ✅ All project dependencies are installed
- ✅ You're running `codeflash` from your project root

#### 🧪 "No optimizations found" or debugging issues
Use the `--verbose` flag for detailed output:
```bash
codeflash optimize --verbose
```

This will show:
- 🔍 Which functions are being analyzed
- 🚫 Why certain functions were skipped
- ⚠️ Detailed error messages
- 📊 Performance analysis results

#### 🔍 "No tests found" errors
Verify:
- 📁 Your test directory path is correct in `pyproject.toml`
- 🔍 Tests are discoverable by your test framework
- 📝 Test files follow naming conventions (`test_*.py` for pytest)

#### ⚙️ Configuration issues
Check your `pyproject.toml`:
```toml
[tool.codeflash]
module = "my_package"
test-framework = "pytest"
tests = "tests/"
```

### Next Steps

- Learn about [Codeflash Concepts](/codeflash-concepts/how-codeflash-works)
- Explore [optimization workflows](/optimizing-with-codeflash/one-function)
- Set up [GitHub Actions integration](/getting-started/codeflash-github-actions)
- Read [configuration options](/configuration) for advanced setups
