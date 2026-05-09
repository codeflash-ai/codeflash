# AI Monitoring

AI-native performance tracking and observability for artificial intelligence workflows, including LLM calls, AI pipeline execution, and token usage monitoring.

## Capabilities

### AI Pipeline Tracking

Track AI workflows and pipelines with automatic exception capture and performance monitoring.

```python { .api }
def ai_track(description: str, **span_kwargs) -> Callable[[F], F]:
    """
    Decorator for tracking AI operations and pipelines.
    
    Parameters:
    - description: Name/description of the AI operation
    - **span_kwargs: Additional span configuration (op, tags, data, etc.)
    
    Returns:
    Decorated function with automatic AI monitoring
    """
```

**Usage Examples:**

```python
from sentry_sdk.ai import ai_track

# Track an AI pipeline
@ai_track("user-query-processing")
def process_user_query(query, context):
    # AI processing logic
    response = llm_call(query, context)
    return response

# Track with custom metadata
@ai_track(
    "document-analysis", 
    op="ai.analysis",
    sentry_tags={"model": "gpt-4", "type": "document"},
    sentry_data={"version": "v2.1"}
)
async def analyze_document(doc_id):
    # Document analysis logic
    return results
```

### Pipeline Name Management

Set and retrieve AI pipeline names for hierarchical tracking and organization.

```python { .api }
def set_ai_pipeline_name(name: Optional[str]) -> None:
    """
    Set the current AI pipeline name.
    
    Parameters:
    - name: Pipeline name (None to clear)
    """

def get_ai_pipeline_name() -> Optional[str]:
    """
    Get the current AI pipeline name.
    
    Returns:
    str: Current pipeline name or None if not set
    """
```

**Usage Examples:**

```python
from sentry_sdk.ai import set_ai_pipeline_name, get_ai_pipeline_name

# Set pipeline context manually
set_ai_pipeline_name("customer-support-bot")

# Operations will be tagged with pipeline name
process_message(user_input)

# Check current pipeline
current_pipeline = get_ai_pipeline_name()
print(f"Running in pipeline: {current_pipeline}")
```

### Token Usage Recording

Record AI model token consumption for cost tracking and performance analysis.

```python { .api }
def record_token_usage(
    span: Span,
    input_tokens: Optional[int] = None,
    input_tokens_cached: Optional[int] = None,
    output_tokens: Optional[int] = None,
    output_tokens_reasoning: Optional[int] = None,
    total_tokens: Optional[int] = None
) -> None:
    """
    Record token usage metrics for AI operations.
    
    Parameters:
    - span: Active span to attach token data
    - input_tokens: Number of input tokens consumed
    - input_tokens_cached: Number of cached input tokens
    - output_tokens: Number of output tokens generated
    - output_tokens_reasoning: Number of reasoning tokens (e.g., o1 models)
    - total_tokens: Total tokens (auto-calculated if not provided)
    """
```

**Usage Examples:**

```python
import sentry_sdk
from sentry_sdk.ai import record_token_usage

with sentry_sdk.start_span(op="ai.chat.completions", name="openai-completion") as span:
    response = openai_client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": "Hello!"}]
    )
    
    # Record token usage from response
    record_token_usage(
        span,
        input_tokens=response.usage.prompt_tokens,
        output_tokens=response.usage.completion_tokens,
        total_tokens=response.usage.total_tokens
    )
```

### Data Normalization Utilities

Utilities for normalizing AI model data (e.g., Pydantic models) for Sentry ingestion.

```python { .api }
def set_data_normalized(
    span: Span, 
    key: str, 
    value: Any, 
    unpack: bool = True
) -> None:
    """
    Set span data with automatic normalization for complex AI objects.
    
    Parameters:
    - span: Target span for data attachment
    - key: Data key name
    - value: Value to normalize and attach (supports Pydantic models)
    - unpack: Whether to unpack single-item lists
    """
```

## Integration Patterns

### Manual Pipeline Tracking

```python
import sentry_sdk
from sentry_sdk.ai import set_ai_pipeline_name, record_token_usage

# Set pipeline name for all subsequent operations
set_ai_pipeline_name("rag-document-qa")

with sentry_sdk.start_span(op="ai.retrieval", name="vector-search") as span:
    documents = vector_db.similarity_search(query)
    span.set_data("documents_found", len(documents))

with sentry_sdk.start_span(op="ai.generation", name="answer-generation") as span:
    response = llm.generate(query, documents)
    
    # Record token usage
    record_token_usage(
        span,
        input_tokens=response.usage.prompt_tokens,
        output_tokens=response.usage.completion_tokens
    )
```

### Automatic Pipeline Tracking

```python
from sentry_sdk.ai import ai_track

@ai_track("intelligent-document-processor")
def process_documents(files):
    results = []
    for file in files:
        # Each operation is tracked under the pipeline
        content = extract_text(file)
        summary = summarize_content(content)
        insights = analyze_sentiment(content)
        results.append({
            'summary': summary,
            'insights': insights
        })
    return results

# Usage - automatically creates pipeline context
processed = process_documents(uploaded_files)
```

### Error Tracking in AI Operations

The `ai_track` decorator automatically captures exceptions with AI-specific context:

```python
@ai_track("model-inference")
def run_inference(model_input):
    try:
        return model.predict(model_input)
    except ModelTimeoutError as e:
        # Exception automatically captured with AI context
        # including pipeline name and operation metadata
        raise
```

## AI-Specific Span Data Constants

The AI monitoring module uses specialized span data constants from `SPANDATA`:

- `GEN_AI_PIPELINE_NAME` - AI pipeline identifier
- `GEN_AI_USAGE_INPUT_TOKENS` - Input token count
- `GEN_AI_USAGE_INPUT_TOKENS_CACHED` - Cached input token count
- `GEN_AI_USAGE_OUTPUT_TOKENS` - Output token count
- `GEN_AI_USAGE_OUTPUT_TOKENS_REASONING` - Reasoning token count
- `GEN_AI_USAGE_TOTAL_TOKENS` - Total token consumption

These constants ensure consistent tagging across AI integrations and custom instrumentation.