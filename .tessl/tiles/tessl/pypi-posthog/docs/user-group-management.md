# User and Group Management

User identification and property management system for tracking user attributes, behavioral data, and organizational groupings. PostHog's user management supports both individual user properties and group-level data for multi-tenant applications.

## Capabilities

### User Property Management

Set and manage user properties with support for both overwriting and append-only operations.

```python { .api }
def set(**kwargs: OptionalSetArgs) -> Optional[str]:
    """
    Set properties on a user record.

    Parameters:
    - distinct_id: Optional[ID_TYPES] - Unique identifier for the user (defaults to context user)
    - properties: Optional[Dict[str, Any]] - Dictionary of properties to set on the user
    - timestamp: Optional[Union[datetime, str]] - When the properties were set
    - uuid: Optional[str] - Unique identifier for this operation
    - disable_geoip: Optional[bool] - Whether to disable GeoIP lookup

    Returns:
    Optional[str] - The operation UUID if successful

    Notes:
    - Overwrites existing property values
    - Context tags are folded into properties
    - No-op if no distinct_id available
    """

def set_once(**kwargs: OptionalSetArgs) -> Optional[str]:
    """
    Set properties on a user record, only if they do not yet exist.

    Parameters:
    - Same as set() method

    Returns:
    Optional[str] - The operation UUID if successful

    Notes:
    - Does not overwrite existing property values
    - Otherwise behaves identically to set()
    """
```

### Group Management

Manage group properties and associations for organizational or team-level data tracking.

```python { .api }
def group_identify(
    group_type: str,
    group_key: str,
    properties: Optional[Dict] = None,
    timestamp: Optional[datetime] = None,
    uuid: Optional[str] = None,
    disable_geoip: Optional[bool] = None
) -> Optional[str]:
    """
    Set properties on a group.

    Parameters:
    - group_type: str - Type of your group (e.g., 'company', 'team', 'organization')
    - group_key: str - Unique identifier of the group
    - properties: Optional[Dict] - Properties to set on the group
    - timestamp: Optional[datetime] - When the group was identified
    - uuid: Optional[str] - Unique identifier for this operation
    - disable_geoip: Optional[bool] - Whether to disable GeoIP lookup

    Returns:
    Optional[str] - The operation UUID if successful
    """
```

### User Identity Management

Link user identities across different stages of their lifecycle and associate anonymous behavior with identified users.

```python { .api }
def alias(
    previous_id: str,
    distinct_id: str,
    timestamp: Optional[datetime] = None,
    uuid: Optional[str] = None,
    disable_geoip: Optional[bool] = None
) -> Optional[str]:
    """
    Associate user behaviour before and after they e.g. register, login, or perform some other identifying action.

    Parameters:
    - previous_id: str - The unique ID of the user before identification
    - distinct_id: str - The current unique id after identification
    - timestamp: Optional[datetime] - When the alias was created
    - uuid: Optional[str] - Unique identifier for this operation
    - disable_geoip: Optional[bool] - Whether to disable GeoIP lookup

    Returns:
    Optional[str] - The operation UUID if successful

    Notes:
    - Links anonymous behavior to identified users
    - Enables cohort analysis across user lifecycle
    - Should be called when user identity becomes known
    """
```

## Usage Examples

### User Property Management

```python
import posthog

# Configure PostHog
posthog.api_key = 'phc_your_project_api_key'

# Set user properties (overwrites existing)
posthog.set('user123', {
    'email': 'user@example.com',
    'name': 'John Doe',
    'plan': 'premium',
    'signup_date': '2024-01-15',
    'trial_end': '2024-02-15'
})

# Set properties only if they don't exist
posthog.set_once('user123', {
    'first_visit': '2024-01-10',
    'initial_referrer': 'google.com',
    'signup_source': 'landing_page'
})

# Using context for automatic user identification
with posthog.new_context():
    posthog.identify_context('user123')
    
    # User ID automatically applied from context
    posthog.set({
        'last_active': '2024-09-07',
        'feature_usage_count': 42
    })
```

### Group Management

```python
import posthog

# Identify a company/organization
posthog.group_identify('company', 'acme_corp', {
    'name': 'Acme Corporation',
    'industry': 'Technology',
    'size': 'Enterprise',
    'plan': 'Business',
    'mrr': 5000,
    'employees': 250
})

# Identify a team within organization
posthog.group_identify('team', 'engineering', {
    'name': 'Engineering Team',
    'department': 'Product',
    'lead': 'jane.doe@acme.com',
    'members_count': 12
})

# Update group properties
posthog.group_identify('company', 'acme_corp', {
    'mrr': 5500,  # Updated MRR
    'employees': 275  # Updated employee count
})
```

### User Identity Linking

```python
import posthog

# User starts as anonymous visitor
anonymous_id = 'anonymous_user_abc123'
posthog.capture(anonymous_id, 'page_viewed', {'page': 'landing'})
posthog.capture(anonymous_id, 'signup_started')

# User completes registration
identified_id = 'user_456'
posthog.capture(identified_id, 'signup_completed', {
    'email': 'user@example.com'
})

# Link anonymous behavior to identified user
posthog.alias(anonymous_id, identified_id)

# Set user properties after identification
posthog.set(identified_id, {
    'email': 'user@example.com',
    'plan': 'free',
    'verified': True
})
```

### Combined User and Group Tracking

```python
import posthog

# Set up user with group associations
user_id = 'user_789'
company_id = 'company_xyz'
team_id = 'team_frontend'

# Identify user properties
posthog.set(user_id, {
    'name': 'Alice Smith',
    'role': 'Senior Developer',
    'department': 'Engineering',
    'hire_date': '2023-06-01'
})

# Identify company
posthog.group_identify('company', company_id, {
    'name': 'XYZ Startup',
    'industry': 'SaaS',
    'size': 'Series A',
    'location': 'San Francisco'
})

# Identify team
posthog.group_identify('team', team_id, {
    'name': 'Frontend Team',
    'tech_stack': 'React',
    'team_lead': 'bob.johnson@xyz.com'
})

# Track events with group context
posthog.capture(user_id, 'feature_used', {
    'feature': 'advanced_dashboard'
}, groups={
    'company': company_id,
    'team': team_id
})
```

### Context-Based Property Management

```python
import posthog

with posthog.new_context():
    posthog.identify_context('user_456')
    posthog.tag('session_type', 'premium')
    posthog.tag('ab_test_group', 'variant_b')
    
    # Tags are automatically included in user properties
    posthog.set({
        'last_login': '2024-09-07T10:30:00Z',
        'subscription_status': 'active'
    })
    
    # Context tags become part of the user profile
    posthog.capture('dashboard_viewed')
```

## Property Types and Best Practices

### Supported Property Types

```python
# Strings
posthog.set('user123', {'name': 'John Doe', 'plan': 'premium'})

# Numbers
posthog.set('user123', {'age': 28, 'score': 95.5})

# Booleans
posthog.set('user123', {'verified': True, 'trial_expired': False})

# Dates (as ISO strings)
posthog.set('user123', {
    'signup_date': '2024-01-15T10:30:00Z',
    'last_active': '2024-09-07'
})

# Arrays
posthog.set('user123', {
    'interests': ['technology', 'sports', 'music'],
    'visited_pages': ['/home', '/about', '/contact']
})
```

### Property Naming Conventions

```python
# Good - clear, consistent naming
posthog.set('user123', {
    'email': 'user@example.com',
    'first_name': 'John',
    'last_name': 'Doe',
    'signup_date': '2024-01-15',
    'subscription_tier': 'premium',
    'feature_flags_enabled': ['new_ui', 'beta_features']
})

# Avoid - inconsistent or unclear names
posthog.set('user123', {
    'Email': 'user@example.com',  # Inconsistent case
    'fName': 'John',  # Abbreviated
    'user_registered': '2024-01-15',  # Inconsistent naming
    'tier': 'premium'  # Could be ambiguous
})
```

### Group Association Patterns

```python
# Multi-level organization structure
posthog.capture('user123', 'report_generated', {
    'report_type': 'monthly_summary'
}, groups={
    'company': 'acme_corp',
    'division': 'north_america',
    'team': 'sales_team_west'
})

# Feature flag evaluation with groups
enabled = posthog.feature_enabled(
    'new_dashboard',
    'user123',
    groups={'company': 'acme_corp'},
    person_properties={'plan': 'enterprise'},
    group_properties={
        'company': {'size': 'large', 'industry': 'tech'}
    }
)
```

## Error Handling

### Validation and Fallbacks

```python
# Properties are automatically validated
posthog.set('user123', {
    'valid_string': 'hello',
    'valid_number': 42,
    'invalid_function': lambda x: x,  # Dropped - not serializable
    'nested_object': {'key': 'value'}  # Flattened automatically
})

# Automatic fallback for missing distinct_id
with posthog.new_context():
    # No context user set - operation is no-op
    posthog.set({'property': 'value'})  # Does nothing
    
    posthog.identify_context('user123')
    posthog.set({'property': 'value'})  # Works correctly
```

### Retry and Reliability

User and group operations support the same retry mechanisms as events:

- Automatic retries for network failures
- Exponential backoff for rate limits
- Queueing for offline scenarios
- Error logging for debugging