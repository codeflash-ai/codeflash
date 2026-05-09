# Self Management

UV provides built-in self-management capabilities for updating to the latest versions, checking version information, and maintaining the UV installation itself. These features ensure you always have access to the latest performance improvements and features.

## Capabilities

### Self Update

Update UV to the latest available version with automatic detection and installation.

```bash { .api }
uv self update
# Updates uv to the latest version
# Downloads and installs the latest release

# Options:
# --prerelease           # Include pre-release versions
# --no-modify-path      # Don't update PATH during installation
# --token TOKEN         # GitHub token for API access
```

Usage examples:

```bash
# Update to latest stable version
uv self update

# Update to latest including pre-releases
uv self update --prerelease

# Update with custom GitHub token
uv self update --token ghp_xxxxxxxxxxxx
```

### Version Information

Display detailed version and build information for UV installation.

```bash { .api }
uv self version
# Shows detailed version information
# Includes build date, commit hash, and platform info

# Options:
# --format FORMAT       # Output format (text/json)
```

Usage examples:

```bash
# Show version information
uv self version

# Get machine-readable version info
uv self version --format json

# Quick version check
uv --version
```

## Update Mechanisms

### Standalone Installer Updates

For UV installed via the standalone installer:

```bash { .api }
# Self-update is available and recommended
uv self update

# Update checks GitHub releases automatically
# Downloads platform-specific binary
# Replaces current installation atomically
```

### Package Manager Updates

For UV installed via package managers:

```bash { .api }
# pip installation
pip install --upgrade uv

# pipx installation
pipx upgrade uv

# Homebrew (macOS)
brew upgrade uv

# Cargo installation
cargo install uv --force
```

Note: `uv self update` may not be available for package manager installations.

## Version Detection

UV automatically detects the installation method and provides appropriate update mechanisms:

### Installation Methods

```bash { .api }
# Check installation method
uv self version

# Standalone installer: Shows self-update capability
# Package manager: Shows package manager info
# Development build: Shows build information
```

### Update Notifications

UV may display update notifications when newer versions are available:

```text { .api }
A newer version of uv is available: 0.8.19 (current: 0.8.18)
Run `uv self update` to update to the latest version.
```

Disable update notifications:

```bash { .api }
UV_NO_UPDATE_CHECK=1
```

## Release Information

UV follows semantic versioning and provides detailed release information:

### Version Format

```text { .api }
MAJOR.MINOR.PATCH[-PRERELEASE][+BUILD]

Examples:
0.8.18              # Stable release
0.9.0-rc.1          # Release candidate
0.8.19-dev+abc123   # Development build
```

### Release Channels

- **Stable**: Fully tested releases for production use
- **Pre-release**: Beta and release candidate versions
- **Development**: Nightly and feature branch builds

### Release Notes

Access release information:

```bash
# Check latest releases on GitHub
open https://github.com/astral-sh/uv/releases

# View changelog
open https://github.com/astral-sh/uv/blob/main/CHANGELOG.md
```

## Configuration

Configure self-management behavior through UV settings:

```toml { .api }
[tool.uv]
# Self-update settings
self-update-check = true            # Check for updates automatically
self-update-prerelease = false      # Include pre-releases in updates
self-update-github-token = "token"  # GitHub token for API access
```

Environment variables:

```bash { .api }
UV_NO_UPDATE_CHECK=1               # Disable update checks
UV_GITHUB_TOKEN=ghp_xxxxxxx        # GitHub token for API access
UV_PRERELEASE=1                    # Include pre-releases
```

## Build Information

UV provides detailed build and runtime information:

### Build Details

```bash { .api }
uv self version --format json
```

Returns information including:
- Version number and build date
- Git commit hash and branch
- Rust compiler version
- Target platform and architecture
- Feature flags and compilation options

### Runtime Information

```bash { .api }
# Check UV executable location
which uv

# Check installation directory
dirname "$(which uv)"

# Check file information
ls -la "$(which uv)"
file "$(which uv)"
```

## Installation Verification

Verify UV installation integrity and functionality:

### Basic Verification

```bash { .api }
# Test basic functionality
uv --version
uv --help

# Test core commands
uv python list
uv cache dir

# Test package operations
uv pip list
```

### Advanced Verification

```bash { .api }
# Check all entry points
ls -la "$(dirname "$(which uv)")"

# Verify checksums (for standalone installations)
# Check against published checksums on GitHub releases

# Test critical functionality
uv venv test-env
uv pip install -e . --target test-env
rm -rf test-env
```

## Troubleshooting Self-Management

### Update Failures

```bash { .api }
# Check network connectivity
curl -s https://api.github.com/repos/astral-sh/uv/releases/latest

# Check file permissions
ls -la "$(which uv)"

# Manual update fallback
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Version Conflicts

```bash { .api }
# Check for multiple UV installations
which -a uv
find /usr -name "uv" 2>/dev/null

# Check PATH for conflicts
echo $PATH | tr ':' '\n' | grep -E "(uv|\.local|\.cargo)"

# Resolve conflicts by updating PATH or removing duplicates
```

### Permission Issues

```bash { .api }
# Check installation permissions
ls -la "$(dirname "$(which uv)")"

# For user installations, ensure proper permissions
chmod +x "$(which uv)"

# For system installations, may require sudo
sudo uv self update
```

## Integration with Package Managers

### Homebrew (macOS)

```bash { .api }
# Install
brew install uv

# Update
brew upgrade uv

# Check version
brew list uv --versions
```

### APT (Ubuntu/Debian)

```bash { .api }
# Add repository
curl -LsSf https://astral.sh/uv/install.sh | sh

# Or use system package if available
sudo apt update && sudo apt install uv
```

### Scoop (Windows)

```powershell { .api }
# Install
scoop install uv

# Update
scoop update uv

# Check version
scoop info uv
```

### Chocolatey (Windows)

```powershell { .api }
# Install
choco install uv

# Update
choco upgrade uv

# Check version
choco info uv
```

## Development Builds

For development and testing with unreleased features:

```bash { .api }
# Install from main branch
cargo install uv --git https://github.com/astral-sh/uv.git

# Install specific commit
cargo install uv --git https://github.com/astral-sh/uv.git --rev abc123

# Build from source
git clone https://github.com/astral-sh/uv.git
cd uv
cargo build --release
```

## Security Considerations

### Update Security

- UV updates are signed and verified
- Downloads use HTTPS with certificate verification
- GitHub releases include checksums for verification
- Self-update process is atomic (complete or rollback)

### Best Practices

```bash { .api }
# Regular updates for security patches
uv self update

# Verify installation integrity
uv self version

# Use stable releases for production
uv self update  # (avoids pre-releases by default)

# Monitor security advisories
# https://github.com/astral-sh/uv/security/advisories
```