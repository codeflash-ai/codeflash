# Authentication

UV provides secure authentication management for PyPI and private package indexes, supporting API tokens, username/password authentication, and keyring integration with encrypted credential storage.

## Capabilities

### Login Management

Authenticate with package indexes and store credentials securely for future operations.

```bash { .api }
uv auth login [SERVICE]
# Authenticates with a package index service
# Stores credentials securely for future use

# Services:
# pypi                    # PyPI (default)
# testpypi               # TestPyPI
# URL                    # Custom index URL

# Options:
# --username USERNAME    # Username for authentication
# --password PASSWORD    # Password for authentication
# --token TOKEN         # API token for authentication
# --keyring-provider    # Keyring provider to use
```

Usage examples:

```bash
# Login to PyPI with interactive prompts
uv auth login

# Login to PyPI with username/password
uv auth login --username myuser --password mypass

# Login with API token
uv auth login --token pypi-token-here

# Login to TestPyPI
uv auth login testpypi

# Login to private index
uv auth login https://private.pypi.org/simple/
```

### Logout Management

Remove stored credentials and invalidate authentication for package indexes.

```bash { .api }
uv auth logout [SERVICE]
# Removes stored credentials for service
# Invalidates authentication tokens where possible

# Services:
# pypi                   # PyPI (default)
# testpypi              # TestPyPI
# URL                   # Custom index URL
# --all                 # All configured services
```

Usage examples:

```bash
# Logout from PyPI
uv auth logout

# Logout from TestPyPI
uv auth logout testpypi

# Logout from private index
uv auth logout https://private.pypi.org/simple/

# Logout from all services
uv auth logout --all
```

### Token Management

Display and manage authentication tokens for troubleshooting and integration purposes.

```bash { .api }
uv auth token [SERVICE]
# Shows authentication token for service
# Useful for troubleshooting and CI/CD integration

# Options:
# --format FORMAT       # Output format (text/json)
```

Usage examples:

```bash
# Show PyPI token
uv auth token

# Show TestPyPI token
uv auth token testpypi

# Show token for private index
uv auth token https://private.pypi.org/simple/

# Get machine-readable output
uv auth token --format json
```

### Credential Directory

Manage authentication credential storage location and configuration.

```bash { .api }
uv auth dir
# Shows path to UV credentials directory
# Location where authentication data is stored
```

Usage examples:

```bash
# Show credentials directory
uv auth dir

# List credential files
ls "$(uv auth dir)"

# Backup credentials
cp -r "$(uv auth dir)" ~/uv-credentials-backup
```

## Authentication Methods

### API Token Authentication

Most secure method for PyPI and compatible indexes:

```bash { .api }
# Using API token (recommended)
uv auth login --token pypi-AgENdGVzdC5weXBpLm9yZw...

# Token format for PyPI:
# pypi-<token-data>

# Token format for TestPyPI:
# testpypi-<token-data>
```

### Username/Password Authentication

Traditional authentication method:

```bash { .api }
# Interactive prompts
uv auth login --username myuser
# Password: [hidden input]

# Non-interactive
uv auth login --username myuser --password mypassword
```

### Environment Variable Authentication

Configure authentication through environment variables:

```bash { .api }
# PyPI token
UV_PUBLISH_TOKEN=pypi-token-here

# Index-specific tokens
UV_INDEX_TOKEN_PYPI=pypi-token
UV_INDEX_TOKEN_TESTPYPI=testpypi-token

# Username/password
UV_PUBLISH_USERNAME=username
UV_PUBLISH_PASSWORD=password

# Index URL
UV_PUBLISH_URL=https://upload.pypi.org/legacy/
```

## Keyring Integration

UV integrates with system keyring services for secure credential storage:

### Supported Keyring Providers

```bash { .api }
# System keyring (default)
uv auth login --keyring-provider keyring

# Subprocess keyring
uv auth login --keyring-provider subprocess

# Disabled keyring
uv auth login --keyring-provider disabled
```

### Platform-Specific Keyring Support

- **macOS**: Keychain integration
- **Windows**: Windows Credential Store
- **Linux**: libsecret, KWallet, or encrypted files

### Keyring Configuration

Configure keyring behavior in UV settings:

```toml { .api }
[tool.uv]
keyring-provider = "keyring"        # keyring, subprocess, disabled
auth-cache-dir = "~/.uv/auth"      # Custom auth cache directory
```

## Index Configuration

Configure authentication for multiple package indexes:

### Global Index Authentication

```toml { .api }
# .pypirc configuration
[distutils]
index-servers =
    pypi
    testpypi
    private

[pypi]
repository = https://upload.pypi.org/legacy/
username = __token__
password = pypi-token

[testpypi]
repository = https://test.pypi.org/legacy/
username = __token__
password = testpypi-token

[private]
repository = https://private-pypi.company.com/simple/
username = company-user
password = company-token
```

### Per-Project Index Authentication

```toml { .api }
# pyproject.toml
[tool.uv]
index = "https://pypi.org/simple/"
extra-index-urls = [
    "https://test.pypi.org/simple/",
    "https://private.company.com/simple/",
]

# Authentication will use stored credentials
```

## Authentication Workflow

### Initial Setup

```bash
# 1. Generate API token on PyPI
# Visit: https://pypi.org/manage/account/token/

# 2. Login with token
uv auth login --token pypi-your-token-here

# 3. Test authentication
uv publish --repository testpypi dist/test-package-1.0.0.tar.gz
```

### CI/CD Integration

```bash { .api }
# GitHub Actions
env:
  UV_PUBLISH_TOKEN: ${{ secrets.PYPI_TOKEN }}

# GitLab CI
variables:
  UV_PUBLISH_TOKEN: $PYPI_TOKEN

# Jenkins
withCredentials([string(credentialsId: 'pypi-token', variable: 'UV_PUBLISH_TOKEN')]) {
    sh 'uv publish'
}
```

### Multiple Index Management

```bash
# Setup multiple indexes
uv auth login pypi --token pypi-token
uv auth login testpypi --token testpypi-token
uv auth login https://private.company.com --username user --password pass

# Publish to specific index
uv publish --repository pypi
uv publish --repository testpypi
uv publish --repository-url https://private.company.com/simple/
```

## Security Best Practices

### Token Management
- Use scoped tokens with minimal necessary permissions
- Rotate tokens regularly
- Store tokens in secure credential managers
- Never commit tokens to version control

### Access Control
- Use separate tokens for different purposes
- Limit token scope to specific projects when possible
- Monitor token usage for unauthorized access
- Revoke compromised tokens immediately

### Environment Security
- Use environment variables in CI/CD
- Encrypt sensitive configuration files
- Limit credential file permissions (600)
- Regular security audits of stored credentials

## Troubleshooting Authentication

### Common Issues

#### Authentication failures:
```bash
# Check stored credentials
uv auth token

# Re-authenticate
uv auth logout
uv auth login --token new-token

# Test with verbose output
uv publish --verbose
```

#### Keyring issues:
```bash
# Disable keyring temporarily
uv auth login --keyring-provider disabled

# Check keyring availability
python -c "import keyring; print(keyring.get_keyring())"
```

#### Token format errors:
```bash
# Verify token format
echo $UV_PUBLISH_TOKEN | head -c 20

# Use correct token prefix
# PyPI: pypi-<token>
# TestPyPI: testpypi-<token>
```

#### Permission errors:
```bash
# Check credential directory permissions
ls -la "$(uv auth dir)"

# Fix permissions
chmod 700 "$(uv auth dir)"
chmod 600 "$(uv auth dir)"/*
```