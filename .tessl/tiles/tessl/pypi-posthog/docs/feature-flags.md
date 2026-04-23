# Feature Flags

Comprehensive feature flag system supporting boolean flags, multivariate testing, remote configuration, and both local and remote evaluation. PostHog's feature flags enable controlled feature rollouts, A/B testing, and dynamic configuration management with built-in caching and fallback mechanisms.

## Capabilities

### Boolean Feature Flags

Check if a feature flag is enabled for a specific user with support for targeting and percentage rollouts.

```python { .api }
def feature_enabled(
    key: str,
    distinct_id: str,
    groups: Optional[dict] = None,
    person_properties: Optional[dict] = None,
    group_properties: Optional[dict] = None,
    only_evaluate_locally: bool = False,
    send_feature_flag_events: bool = True,
    disable_geoip: Optional[bool] = None
) -> bool:
    """
    Use feature flags to enable or disable features for users.

    Parameters:
    - key: str - The feature flag key
    - distinct_id: str - The user's distinct ID
    - groups: Optional[dict] - Groups mapping from group type to group key
    - person_properties: Optional[dict] - Person properties for evaluation
    - group_properties: Optional[dict] - Group properties in format { group_type_name: { group_properties } }
    - only_evaluate_locally: bool - Whether to evaluate only locally (default: False)
    - send_feature_flag_events: bool - Whether to send feature flag events (default: True)
    - disable_geoip: Optional[bool] - Whether to disable GeoIP lookup

    Returns:
    bool - True if the feature flag is enabled, False otherwise

    Notes:
    - Call load_feature_flags() before to avoid unexpected requests
    - Automatically sends $feature_flag_called events unless disabled
    """
```

### Multivariate Feature Flags

Get feature flag variants for A/B testing and experiments with multiple treatment groups.

```python { .api }
def get_feature_flag(
    key: str,
    distinct_id: str,
    groups: Optional[dict] = None,
    person_properties: Optional[dict] = None,
    group_properties: Optional[dict] = None,
    only_evaluate_locally: bool = False,
    send_feature_flag_events: bool = True,
    disable_geoip: Optional[bool] = None
) -> Optional[FeatureFlag]:
    """
    Get feature flag variant for users. Used with experiments.

    Parameters:
    - Same as feature_enabled()

    Returns:
    Optional[FeatureFlag] - FeatureFlag object with variant information, or None if not enabled

    Notes:
    - Returns variant string for multivariate flags
    - Returns True for simple boolean flags
    - Groups format: {"organization": "5"}
    - Group properties format: {"organization": {"name": "PostHog", "employees": 11}}
    """
```

### Bulk Flag Operations

Retrieve all flags and their values for a user in a single operation for efficient bulk evaluation.

```python { .api }
def get_all_flags(
    distinct_id: str,
    groups: Optional[dict] = None,
    person_properties: Optional[dict] = None,
    group_properties: Optional[dict] = None,
    only_evaluate_locally: bool = False,
    disable_geoip: Optional[bool] = None
) -> Optional[dict[str, FeatureFlag]]:
    """
    Get all flags for a given user.

    Parameters:
    - distinct_id: str - The user's distinct ID
    - groups: Optional[dict] - Groups mapping
    - person_properties: Optional[dict] - Person properties
    - group_properties: Optional[dict] - Group properties
    - only_evaluate_locally: bool - Whether to evaluate only locally
    - disable_geoip: Optional[bool] - Whether to disable GeoIP lookup

    Returns:
    Optional[dict[str, FeatureFlag]] - Dictionary mapping flag keys to FeatureFlag objects

    Notes:
    - More efficient than individual flag calls
    - Does not send feature flag events
    - Flags are key-value pairs where value is variant, True, or False
    """

def get_all_flags_and_payloads(
    distinct_id: str,
    groups: Optional[dict] = None,
    person_properties: Optional[dict] = None,
    group_properties: Optional[dict] = None,
    only_evaluate_locally: bool = False,
    disable_geoip: Optional[bool] = None
) -> FlagsAndPayloads:
    """
    Get all flags and their payloads for a user.

    Parameters:
    - Same as get_all_flags()

    Returns:
    FlagsAndPayloads - Object with featureFlags and featureFlagPayloads dictionaries
    """
```

### Feature Flag Results with Payloads

Get complete feature flag information including variants, payloads, and evaluation reasons.

```python { .api }
def get_feature_flag_result(
    key: str,
    distinct_id: str,
    groups: Optional[dict] = None,
    person_properties: Optional[dict] = None,
    group_properties: Optional[dict] = None,
    only_evaluate_locally: bool = False,
    send_feature_flag_events: bool = True,
    disable_geoip: Optional[bool] = None
) -> Optional[FeatureFlagResult]:
    """
    Get a FeatureFlagResult object which contains the flag result and payload.

    Parameters:
    - Same as feature_enabled()

    Returns:
    Optional[FeatureFlagResult] - Complete flag result with enabled, variant, payload, key, and reason

    Notes:
    - Most comprehensive flag evaluation method
    - Includes automatic JSON deserialization of payloads
    - Provides evaluation reason for debugging
    """

def get_feature_flag_payload(
    key: str,
    distinct_id: str,
    match_value: Optional[str] = None,
    groups: Optional[dict] = None,
    person_properties: Optional[dict] = None,
    group_properties: Optional[dict] = None,
    only_evaluate_locally: bool = False,
    send_feature_flag_events: bool = True,
    disable_geoip: Optional[bool] = None
) -> Optional[str]:
    """
    Get the payload for a feature flag.

    Parameters:
    - key: str - The feature flag key
    - distinct_id: str - The user's distinct ID
    - match_value: Optional[str] - Expected flag value for payload retrieval
    - Other parameters same as feature_enabled()

    Returns:
    Optional[str] - The payload string, or None if flag not enabled or no payload
    """
```

### Remote Configuration

Access remote configuration flags that don't require user evaluation, useful for application-wide settings.

```python { .api }
def get_remote_config_payload(key: str) -> Optional[str]:
    """
    Get the payload for a remote config feature flag.

    Parameters:
    - key: str - The key of the feature flag

    Returns:
    Optional[str] - The payload associated with the feature flag, decrypted if encrypted

    Notes:
    - Requires personal_api_key to be set for authentication
    - Used for application-wide configuration
    - Does not require user evaluation
    """
```

### Flag Management

Load and inspect feature flag definitions for debugging and management.

```python { .api }
def load_feature_flags():
    """
    Load feature flag definitions from PostHog.

    Notes:
    - Fetches latest flag definitions from server
    - Updates local cache for faster evaluation
    - Should be called on application startup
    - Enables local evaluation for better performance
    """

def feature_flag_definitions():
    """
    Returns loaded feature flags.

    Returns:
    dict - Currently loaded feature flag definitions

    Notes:
    - Helpful for debugging what flag information is loaded
    - Shows flag keys, conditions, and rollout percentages
    - Returns empty dict if no flags loaded
    """
```

## Usage Examples

### Basic Feature Flag Usage

```python
import posthog

# Configure PostHog
posthog.api_key = 'phc_your_project_api_key'
posthog.personal_api_key = 'phc_your_personal_api_key'  # For remote config

# Load flags on startup
posthog.load_feature_flags()

# Simple boolean flag check
if posthog.feature_enabled('new-checkout', 'user123'):
    # Show new checkout flow
    render_new_checkout()
else:
    # Show existing checkout
    render_old_checkout()

# Check flag with user properties
enabled = posthog.feature_enabled(
    'premium-features',
    'user123',
    person_properties={'plan': 'premium', 'region': 'us'}
)

if enabled:
    show_premium_features()
```

### Multivariate Flag Testing

```python
import posthog

# Get flag variant for A/B testing
variant = posthog.get_feature_flag('checkout-design', 'user123')

if variant == 'variant-a':
    render_checkout_design_a()
elif variant == 'variant-b':
    render_checkout_design_b()
elif variant == 'control':
    render_original_checkout()
else:
    # Flag not enabled or no variant matched
    render_original_checkout()

# Get variant with payload
result = posthog.get_feature_flag_result('pricing-test', 'user123')

if result and result.enabled:
    variant = result.variant  # 'control', 'test-a', 'test-b'
    config = result.payload   # {'discount': 10, 'button_color': 'red'}
    
    render_pricing_page(variant, config)
```

### Group-Based Feature Flags

```python
import posthog

# Feature flag with company-level targeting
enabled = posthog.feature_enabled(
    'enterprise-features',
    'user123',
    groups={'company': 'acme_corp'},
    person_properties={'role': 'admin'},
    group_properties={
        'company': {
            'plan': 'enterprise',
            'employees': 500,
            'industry': 'technology'
        }
    }
)

if enabled:
    show_enterprise_dashboard()

# Multi-level group targeting
variant = posthog.get_feature_flag(
    'ui-redesign',
    'user123',
    groups={
        'company': 'acme_corp',
        'team': 'engineering',
        'region': 'us-west'
    },
    group_properties={
        'company': {'size': 'large'},
        'team': {'department': 'product'},
        'region': {'timezone': 'PST'}
    }
)
```

### Bulk Flag Evaluation

```python
import posthog

# Get all flags for efficient evaluation
all_flags = posthog.get_all_flags(
    'user123',
    groups={'company': 'acme_corp'},
    person_properties={'plan': 'premium'}
)

if all_flags:
    # Check multiple flags efficiently
    features = {
        'new_ui': all_flags.get('new-ui', False),
        'beta_features': all_flags.get('beta-features', False),
        'advanced_analytics': all_flags.get('advanced-analytics', False)
    }
    
    configure_user_interface(features)

# Get flags with payloads
flags_and_payloads = posthog.get_all_flags_and_payloads(
    'user123',
    person_properties={'segment': 'power_user'}
)

feature_flags = flags_and_payloads['featureFlags']
payloads = flags_and_payloads['featureFlagPayloads']

# Configure features with their payloads
for flag_key, enabled in feature_flags.items():
    if enabled and flag_key in payloads:
        configure_feature(flag_key, payloads[flag_key])
```

### Remote Configuration

```python
import posthog

# Application-wide configuration
api_rate_limit_config = posthog.get_remote_config_payload('api-rate-limits')
if api_rate_limit_config:
    import json
    config = json.loads(api_rate_limit_config)
    set_rate_limits(config['requests_per_minute'])

# Feature configuration without user context
maintenance_mode = posthog.get_remote_config_payload('maintenance-mode')
if maintenance_mode == 'enabled':
    show_maintenance_page()
```

### Local vs Remote Evaluation

```python
import posthog

# Force local evaluation only (faster, but may be stale)
local_result = posthog.feature_enabled(
    'new-feature',
    'user123',
    only_evaluate_locally=True
)

# Allow remote evaluation (slower, but always current)
remote_result = posthog.feature_enabled(
    'new-feature',
    'user123',
    only_evaluate_locally=False  # Default behavior
)

# Disable automatic event tracking
silent_check = posthog.feature_enabled(
    'internal-flag',
    'user123',
    send_feature_flag_events=False
)
```

### Advanced Configuration

```python
import posthog

# Configure flag polling
posthog.poll_interval = 60  # Check for flag updates every 60 seconds
posthog.enable_local_evaluation = True  # Enable local flag evaluation

# Custom error handling
def flag_error_handler(error):
    print(f"Feature flag error: {error}")
    # Log to monitoring service
    log_error("feature_flag_error", str(error))

posthog.on_error = flag_error_handler

# Load flags with error handling
try:
    posthog.load_feature_flags()
except Exception as e:
    print(f"Failed to load feature flags: {e}")
    # Continue with default behavior
```

## Flag Types and Data Structures

### FeatureFlag Object

```python
from posthog.types import FeatureFlag, FlagReason, FlagMetadata

# FeatureFlag structure
flag = FeatureFlag(
    key='test-flag',
    enabled=True,
    variant='test-variant',
    reason=FlagReason(
        code='CONDITION_MATCH',
        condition_index=0,
        description='User matched condition 1'
    ),
    metadata=FlagMetadata(
        id=123,
        payload='{"config": "value"}',
        version=1,
        description='Test flag for A/B testing'
    )
)

# Access flag properties
flag_value = flag.get_value()  # Returns variant or enabled boolean
```

### FeatureFlagResult Object

```python
from posthog.types import FeatureFlagResult

# FeatureFlagResult structure (most comprehensive)
result = FeatureFlagResult(
    key='pricing-test',
    enabled=True,
    variant='test-variant',
    payload={'discount': 15, 'color': 'blue'},
    reason='User in test group A'
)

# Access result properties
value = result.get_value()  # 'test-variant'
config = result.payload     # {'discount': 15, 'color': 'blue'}
```

## Best Practices

### Flag Naming and Organization

```python
# Good - descriptive, hierarchical naming
posthog.feature_enabled('checkout-redesign-v2', 'user123')
posthog.feature_enabled('billing-monthly-invoicing', 'user123')
posthog.feature_enabled('analytics-real-time-data', 'user123')

# Avoid - vague or inconsistent naming
posthog.feature_enabled('flag1', 'user123')
posthog.feature_enabled('NewFeature', 'user123')  # Inconsistent case
```

### Performance Optimization

```python
# Load flags once on application startup
posthog.load_feature_flags()

# Use bulk evaluation for multiple flags
all_flags = posthog.get_all_flags('user123')

# Enable local evaluation for better performance
posthog.enable_local_evaluation = True

# Use appropriate evaluation mode
# Local: faster, may be slightly stale
local_check = posthog.feature_enabled('flag', 'user', only_evaluate_locally=True)

# Remote: slower, always current
remote_check = posthog.feature_enabled('flag', 'user', only_evaluate_locally=False)
```

### Error Handling and Fallbacks

```python
def check_feature_with_fallback(flag_key, user_id, default=False):
    try:
        return posthog.feature_enabled(flag_key, user_id)
    except Exception as e:
        print(f"Feature flag check failed: {e}")
        return default

# Use safe defaults
new_ui_enabled = check_feature_with_fallback('new-ui', 'user123', default=False)
```

### Testing and Debugging

```python
# Check loaded flag definitions
definitions = posthog.feature_flag_definitions()
print(f"Loaded {len(definitions)} feature flags")

# Debug flag evaluation
result = posthog.get_feature_flag_result('debug-flag', 'user123')
if result:
    print(f"Flag: {result.key}")
    print(f"Enabled: {result.enabled}")
    print(f"Variant: {result.variant}")
    print(f"Reason: {result.reason}")
    print(f"Payload: {result.payload}")
```