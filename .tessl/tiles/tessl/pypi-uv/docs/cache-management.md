# Cache Management

UV maintains a global cache for packages, metadata, and build artifacts to provide fast installations and reduce network usage. The cache system supports deduplication, automatic pruning, and manual management for optimal performance and disk usage.

## Capabilities

### Cache Cleaning

Remove cache entries to free disk space and resolve cache-related issues.

```bash { .api }
uv cache clean
uv clean                        # Legacy alias for cache clean
# Removes cache entries for packages and metadata
# Clears both package and metadata caches

# Options:
# PACKAGE...             # Clean specific packages only
# --all                  # Clean all cache entries (default)
# --dry-run             # Show what would be removed without removing
```

Usage examples:

```bash
# Clean entire cache
uv cache clean

# Clean specific packages
uv cache clean requests numpy

# Show what would be cleaned
uv cache clean --dry-run

# Clean with confirmation
uv cache clean --all
```

### Cache Pruning

Remove unreachable and orphaned cache objects while preserving recently used entries.

```bash { .api }
uv cache prune
# Removes unreachable objects from cache
# Preserves recently used packages and metadata

# Options:
# --ci                  # Use CI-appropriate pruning strategy
# --dry-run            # Show what would be pruned without removing
```

Usage examples:

```bash
# Prune unreachable cache objects
uv cache prune

# Show what would be pruned
uv cache prune --dry-run

# Use CI-friendly pruning
uv cache prune --ci
```

### Cache Directory Information

Display cache location and usage statistics for monitoring and troubleshooting.

```bash { .api }
uv cache dir
# Shows the cache directory path
# Location where UV stores cached data
```

Usage examples:

```bash
# Show cache directory
uv cache dir

# Show cache size
du -sh "$(uv cache dir)"

# List cache contents
ls -la "$(uv cache dir)"
```

## Cache Structure

UV organizes the cache into logical sections for efficient access and management:

```text { .api }
cache/
├── wheels/              # Built wheels for packages
│   ├── pypi/           # PyPI package wheels
│   ├── simple/         # Simple index wheels
│   └── git/            # Git repository wheels
├── built-wheels/       # Locally built wheels
├── downloads/          # Downloaded source distributions
├── metadata/           # Package metadata cache
│   ├── pypi/          # PyPI metadata
│   └── simple/        # Simple index metadata
├── builds/             # Build environments and artifacts
└── git/               # Git repository clones
    ├── db/            # Git database objects
    └── checkouts/     # Git working directories
```

## Cache Configuration

Configure cache behavior through UV settings and environment variables.

### Global Cache Configuration

```toml { .api }
[tool.uv]
cache-dir = "~/.cache/uv"           # Custom cache directory
no-cache = false                    # Disable cache usage
cache-keys = ["platform", "python"] # Cache key components

# Cache size limits
max-cache-size = "10GB"             # Maximum cache size
cache-retention = "30d"             # Cache retention period
```

### Environment Variables

```bash { .api }
UV_CACHE_DIR=/custom/cache/path     # Custom cache directory
UV_NO_CACHE=1                      # Disable cache usage
UV_LINK_MODE=copy                  # Cache linking mode (copy/hardlink/symlink)
```

### Per-Command Cache Control

```bash { .api }
# Global cache options (available for most commands):
--cache-dir DIR                     # Override cache directory
--no-cache                         # Disable cache for this operation
--refresh                          # Refresh cached data
--refresh-package PACKAGE          # Refresh specific package cache
```

## Cache Benefits

### Performance Improvements
- **Fast installations**: Reuse downloaded and built packages
- **Reduced network usage**: Cache packages and metadata locally
- **Parallel builds**: Cache enables concurrent package processing
- **Incremental updates**: Only download changed components

### Disk Space Efficiency
- **Deduplication**: Share identical files across cache entries
- **Compression**: Store cached data in compressed format when beneficial
- **Intelligent linking**: Use hardlinks and symlinks to save space
- **Automatic cleanup**: Remove unreachable cache entries

## Cache Strategies

### Linking Modes

UV supports different cache linking strategies:

```bash { .api }
# Hardlink mode (default, most efficient)
UV_LINK_MODE=hardlink

# Copy mode (safer, uses more space)
UV_LINK_MODE=copy

# Symlink mode (efficient, may have compatibility issues)
UV_LINK_MODE=symlink
```

### Cache Keys

UV generates cache keys based on multiple factors:

- **Platform**: Operating system and architecture
- **Python version**: Major.minor Python version
- **Package version**: Exact package version and hash
- **Dependencies**: Dependency tree hash
- **Build configuration**: Build settings and environment

## Cache Maintenance

### Automatic Maintenance

UV performs automatic cache maintenance:

- **Size-based cleanup**: Remove old entries when cache exceeds limits
- **Time-based cleanup**: Remove entries older than retention period
- **Access-based cleanup**: Remove least recently used entries
- **Integrity checks**: Verify cache entry integrity and remove corrupted data

### Manual Maintenance

Regular maintenance commands:

```bash
# Check cache size and usage
du -sh "$(uv cache dir)"
uv cache dir && find "$(uv cache dir)" -type f | wc -l

# Clean specific package types
uv cache clean --dry-run | grep wheels
uv cache clean requests urllib3 certifi

# Prune old cache entries
uv cache prune --dry-run
uv cache prune

# Complete cache rebuild
uv cache clean --all
```

### Cache Monitoring

Monitor cache performance and usage:

```bash { .api }
# Cache statistics
ls -la "$(uv cache dir)" | head -20

# Package-specific cache usage
find "$(uv cache dir)" -name "*requests*" -type f

# Cache age analysis
find "$(uv cache dir)" -type f -mtime +7  # Files older than 7 days

# Disk usage by cache component
du -sh "$(uv cache dir)"/*
```

## Multi-User Cache

UV supports shared cache configurations for teams and CI environments:

### Shared Cache Setup

```bash { .api }
# Set shared cache directory
export UV_CACHE_DIR=/shared/uv-cache

# Set appropriate permissions
mkdir -p /shared/uv-cache
chmod 775 /shared/uv-cache
chgrp developers /shared/uv-cache
```

### Cache Isolation

```bash { .api }
# User-specific cache
UV_CACHE_DIR=~/.cache/uv-$USER

# Project-specific cache
UV_CACHE_DIR=./.uv-cache

# Environment-specific cache
UV_CACHE_DIR=/tmp/uv-cache-$$
```

## CI/CD Cache Integration

Optimize CI/CD pipelines with cache management:

### GitHub Actions

```yaml { .api }
- name: Cache UV
  uses: actions/cache@v3
  with:
    path: ~/.cache/uv
    key: uv-${{ runner.os }}-${{ hashFiles('**/pyproject.toml') }}
    restore-keys: |
      uv-${{ runner.os }}-
```

### GitLab CI

```yaml { .api }
cache:
  key: uv-$CI_COMMIT_REF_SLUG
  paths:
    - .cache/uv/

before_script:
  - export UV_CACHE_DIR="$CI_PROJECT_DIR/.cache/uv"
```

### Docker Builds

```dockerfile { .api }
# Multi-stage build with cache
FROM python:3.12 as builder
ENV UV_CACHE_DIR=/opt/uv-cache
RUN pip install uv
COPY . /app
WORKDIR /app
RUN uv sync

# Production stage
FROM python:3.12-slim
COPY --from=builder /opt/uv-cache /opt/uv-cache
COPY --from=builder /app/.venv /app/.venv
```

## Troubleshooting Cache Issues

### Cache Corruption

```bash
# Verify cache integrity
uv cache prune --dry-run

# Clean corrupted entries
uv cache clean --dry-run
uv cache clean

# Complete cache reset
rm -rf "$(uv cache dir)"
uv cache dir  # Recreates cache directory
```

### Disk Space Issues

```bash
# Check cache size
du -sh "$(uv cache dir)"

# Clean large packages
du -sh "$(uv cache dir)"/* | sort -rh | head -10
uv cache clean large-package

# Aggressive cleanup
uv cache prune
uv cache clean --all
```

### Permission Issues

```bash
# Check cache permissions
ls -la "$(uv cache dir)"

# Fix permissions
chmod -R u+rw "$(uv cache dir)"

# Reset cache with correct permissions
rm -rf "$(uv cache dir)"
uv cache dir
```

### Network vs Cache Issues

```bash
# Force network refresh
uv pip install --refresh requests

# Disable cache temporarily
uv pip install --no-cache requests

# Compare with/without cache
time uv pip install --no-cache requests
time uv pip install requests
```

## Cache Security

### Security Considerations

- Cache contents are not encrypted by default
- Shared caches should use appropriate file permissions
- Network-sourced cache entries should be verified
- Regular cache cleanup prevents information leakage

### Best Practices

```bash
# Secure cache permissions
chmod 700 "$(uv cache dir)"

# Regular cleanup
uv cache prune --ci

# Cache isolation for sensitive projects
UV_CACHE_DIR=/secure/project-cache

# Audit cache contents
find "$(uv cache dir)" -type f -name "*.whl" | head -10
```