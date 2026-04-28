# AI Integrations

LLM provider integrations with automatic usage tracking, cost monitoring, and performance analytics. PostHog's AI integrations provide drop-in replacements for popular LLM clients that automatically capture usage metrics, costs, and performance data while maintaining full API compatibility.

## Capabilities

### OpenAI Integration

Drop-in replacement for OpenAI clients with automatic tracking of usage, costs, and performance metrics.

```python { .api }
class OpenAI:
    """
    PostHog-wrapped OpenAI client with automatic usage tracking.
    
    Provides identical API to openai.OpenAI with added PostHog telemetry.
    Automatically tracks token usage, costs, model performance, and errors.
    """

class AsyncOpenAI:
    """
    PostHog-wrapped async OpenAI client with automatic usage tracking.
    
    Async version of OpenAI wrapper with identical API to openai.AsyncOpenAI.
    """

class AzureOpenAI:
    """
    PostHog-wrapped Azure OpenAI client with automatic usage tracking.
    
    Provides identical API to openai.AzureOpenAI with added PostHog telemetry.
    """

class AsyncAzureOpenAI:
    """
    PostHog-wrapped async Azure OpenAI client with automatic usage tracking.
    
    Async version of Azure OpenAI wrapper.
    """

# Utility functions for OpenAI data formatting
def format_openai_response(response) -> dict:
    """Format OpenAI response for PostHog tracking"""

def format_openai_input(input) -> dict:
    """Format OpenAI input for PostHog tracking"""

def extract_openai_tools(tools) -> list:
    """Extract tools from OpenAI request for tracking"""

def format_openai_streaming_content(content) -> dict:
    """Format OpenAI streaming content for tracking"""
```

### Anthropic Integration

Drop-in replacement for Anthropic clients with comprehensive usage tracking and cost monitoring.

```python { .api }
class Anthropic:
    """
    PostHog-wrapped Anthropic client with automatic usage tracking.
    
    Provides identical API to anthropic.Anthropic with added PostHog telemetry.
    """

class AsyncAnthropic:
    """
    PostHog-wrapped async Anthropic client with automatic usage tracking.
    
    Async version of Anthropic wrapper with identical API to anthropic.AsyncAnthropic.
    """

class AnthropicBedrock:
    """
    PostHog-wrapped Anthropic Bedrock client with automatic usage tracking.
    
    For use with AWS Bedrock Anthropic models.
    """

class AsyncAnthropicBedrock:
    """
    PostHog-wrapped async Anthropic Bedrock client with automatic usage tracking.
    """

class AnthropicVertex:
    """
    PostHog-wrapped Anthropic Vertex AI client with automatic usage tracking.
    
    For use with Google Cloud Vertex AI Anthropic models.
    """

class AsyncAnthropicVertex:
    """
    PostHog-wrapped async Anthropic Vertex AI client with automatic usage tracking.
    """

# Utility functions for Anthropic data formatting
def format_anthropic_response(response) -> dict:
    """Format Anthropic response for PostHog tracking"""

def format_anthropic_input(input) -> dict:
    """Format Anthropic input for PostHog tracking"""

def extract_anthropic_tools(tools) -> list:
    """Extract tools from Anthropic request for tracking"""

def format_anthropic_streaming_content(content) -> dict:
    """Format Anthropic streaming content for tracking"""
```

### Gemini Integration

Google Gemini client wrapper with automatic usage tracking and performance monitoring.

```python { .api }
class Client:
    """
    PostHog-wrapped Gemini client with automatic usage tracking.
    
    Provides comprehensive tracking for Google Gemini model interactions.
    Drop-in replacement for Google's genai.Client.
    """

# genai module compatibility
genai = _GenAI()  # Contains Client for drop-in replacement

# Utility functions for Gemini data formatting
def format_gemini_response(response) -> dict:
    """Format Gemini response for PostHog tracking"""

def format_gemini_input(input) -> dict:
    """Format Gemini input for PostHog tracking"""

def extract_gemini_tools(tools) -> list:
    """Extract tools from Gemini request for tracking"""
```

## Usage Examples

### Langchain Integration

Langchain callback handler for comprehensive tracking of complex AI workflows and chains.

```python { .api }
class CallbackHandler:
    """
    PostHog callback handler for Langchain applications.
    
    Automatically tracks:
    - Chain executions and performance
    - LLM calls and token usage
    - Tool usage and results
    - Agent actions and decisions
    - Error handling and debugging info
    
    Compatible with all Langchain components including chains, agents, and tools.
    """
```

## Usage Examples

### OpenAI Integration

```python
from posthog.ai.openai import OpenAI
import posthog

# Configure PostHog
posthog.api_key = 'phc_your_project_api_key'

# Create OpenAI client with PostHog tracking
client = OpenAI(
    api_key="your-openai-api-key",
    # All standard OpenAI parameters supported
)

# Use exactly like standard OpenAI client
response = client.chat.completions.create(
    model="gpt-4",
    messages=[
        {"role": "user", "content": "Hello, how are you?"}
    ],
    temperature=0.7,
    max_tokens=150
)

print(response.choices[0].message.content)

# PostHog automatically tracks:
# - Token usage (prompt + completion tokens)
# - Model and parameters used
# - Response time and latency
# - Costs (when available)
# - Errors and failures
```

### Async OpenAI Usage

```python
from posthog.ai.openai import AsyncOpenAI
import asyncio
import posthog

posthog.api_key = 'phc_your_project_api_key'

async def main():
    client = AsyncOpenAI(api_key="your-openai-api-key")
    
    response = await client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "user", "content": "Write a haiku about programming"}
        ]
    )
    
    print(response.choices[0].message.content)

asyncio.run(main())
```

### Azure OpenAI Integration

```python
from posthog.ai.openai import AzureOpenAI
import posthog

posthog.api_key = 'phc_your_project_api_key'

client = AzureOpenAI(
    azure_endpoint="https://your-resource.openai.azure.com/",
    api_key="your-azure-openai-key",
    api_version="2024-02-01"
)

response = client.chat.completions.create(
    model="gpt-4",  # Your deployment name
    messages=[
        {"role": "user", "content": "Explain quantum computing"}
    ]
)

print(response.choices[0].message.content)
```

### Anthropic Integration

```python
from posthog.ai.anthropic import Anthropic
import posthog

posthog.api_key = 'phc_your_project_api_key'

# Create Anthropic client with PostHog tracking
client = Anthropic(
    api_key="your-anthropic-api-key"
)

# Use exactly like standard Anthropic client
response = client.messages.create(
    model="claude-3-opus-20240229",
    max_tokens=1000,
    messages=[
        {"role": "user", "content": "Explain the theory of relativity"}
    ]
)

print(response.content[0].text)

# Automatic tracking includes:
# - Token usage and costs
# - Model performance metrics
# - Response quality indicators
# - Error rates and types
```

### Anthropic Bedrock Integration

```python
from posthog.ai.anthropic import AnthropicBedrock
import posthog

posthog.api_key = 'phc_your_project_api_key'

client = AnthropicBedrock(
    aws_access_key="your-aws-access-key",
    aws_secret_key="your-aws-secret-key",
    aws_region="us-east-1"
)

response = client.messages.create(
    model="anthropic.claude-3-sonnet-20240229-v1:0",
    max_tokens=1000,
    messages=[
        {"role": "user", "content": "What is machine learning?"}
    ]
)

print(response.content[0].text)
```

### Gemini Integration

```python
from posthog.ai.gemini import Client, genai
import posthog

posthog.api_key = 'phc_your_project_api_key'

# Configure Gemini with PostHog tracking
genai.configure(api_key="your-gemini-api-key")

# Create client with PostHog tracking
client = genai.GenerativeModel("gemini-pro")

# Use exactly like standard Gemini client
response = client.generate_content("Explain quantum computing")
print(response.text)

# PostHog automatically tracks:
# - Token usage and costs
# - Model performance metrics
# - Response quality indicators
# - Error rates and types
```

### Streaming Responses

```python
from posthog.ai.openai import OpenAI
import posthog

posthog.api_key = 'phc_your_project_api_key'
client = OpenAI(api_key="your-openai-api-key")

# Streaming is fully supported with automatic tracking
stream = client.chat.completions.create(
    model="gpt-4",
    messages=[
        {"role": "user", "content": "Write a story about AI"}
    ],
    stream=True
)

full_response = ""
for chunk in stream:
    if chunk.choices[0].delta.content is not None:
        content = chunk.choices[0].delta.content
        print(content, end="")
        full_response += content

# PostHog tracks complete streaming session including:
# - Total tokens used
# - Streaming latency and throughput
# - Time to first token
# - Complete response assembly
```

### Langchain Integration

```python
from posthog.ai.langchain import CallbackHandler
from langchain.llms import OpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
import posthog

posthog.api_key = 'phc_your_project_api_key'

# Create PostHog callback handler
posthog_handler = CallbackHandler()

# Set up Langchain components
llm = OpenAI(temperature=0.7)
prompt = PromptTemplate(
    input_variables=["topic"],
    template="Write a short article about {topic}"
)

chain = LLMChain(llm=llm, prompt=prompt)

# Run chain with PostHog tracking
result = chain.run(
    topic="artificial intelligence",
    callbacks=[posthog_handler]
)

print(result)

# PostHog automatically tracks:
# - Chain execution time and success
# - LLM calls and token usage
# - Prompt templates and variables
# - Output quality and length
# - Error handling and retries
```

### Advanced Langchain Usage

```python
from posthog.ai.langchain import CallbackHandler
from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType
from langchain.llms import OpenAI
import posthog

posthog.api_key = 'phc_your_project_api_key'

# Custom tools
def calculator(expression):
    """Calculate mathematical expressions"""
    try:
        return str(eval(expression))
    except:
        return "Error in calculation"

def search_tool(query):
    """Mock search tool"""
    return f"Search results for: {query}"

tools = [
    Tool(
        name="Calculator",
        func=calculator,
        description="Use for mathematical calculations"
    ),
    Tool(
        name="Search",
        func=search_tool,
        description="Use for web searches"
    )
]

# Create agent with PostHog tracking
llm = OpenAI(temperature=0)
agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

# Create callback handler
posthog_handler = CallbackHandler()

# Run agent with comprehensive tracking
result = agent.run(
    "What is 15 * 27, and then search for information about that number?",
    callbacks=[posthog_handler]
)

print(result)

# Tracks agent reasoning, tool usage, and decision making
```

### Context and User Tracking

```python
from posthog.ai.openai import OpenAI
import posthog

posthog.api_key = 'phc_your_project_api_key'

# Use context for user-specific AI tracking
with posthog.new_context():
    posthog.identify_context('user123')
    posthog.tag('session_type', 'premium')
    posthog.tag('use_case', 'content_generation')
    
    client = OpenAI(api_key="your-openai-api-key")
    
    # AI usage automatically associated with user and context
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "user", "content": "Help me write a blog post"}
        ]
    )
    
    # Additional manual tracking
    posthog.capture('ai_content_generated', {
        'content_type': 'blog_post',
        'word_count': len(response.choices[0].message.content.split()),
        'satisfaction': 'high'
    })
```

### Error Handling and Monitoring

```python
from posthog.ai.openai import OpenAI
import posthog

posthog.api_key = 'phc_your_project_api_key'
client = OpenAI(api_key="your-openai-api-key")

def safe_ai_call(messages, **kwargs):
    """Wrapper for AI calls with comprehensive error tracking"""
    try:
        with posthog.new_context():
            posthog.tag('ai_operation', 'chat_completion')
            posthog.tag('model', kwargs.get('model', 'gpt-3.5-turbo'))
            
            response = client.chat.completions.create(
                messages=messages,
                **kwargs
            )
            
            # Track successful usage
            posthog.capture('ai_call_success', {
                'tokens_used': response.usage.total_tokens,
                'model': response.model,
                'response_length': len(response.choices[0].message.content)
            })
            
            return response
            
    except Exception as e:
        # Error is automatically captured by context
        posthog.capture('ai_call_failed', {
            'error_type': type(e).__name__,
            'error_message': str(e),
            'model': kwargs.get('model', 'unknown')
        })
        raise

# Usage with error tracking
try:
    response = safe_ai_call(
        messages=[{"role": "user", "content": "Hello"}],
        model="gpt-4",
        max_tokens=100
    )
except Exception as e:
    print(f"AI call failed: {e}")
```

### Privacy Mode

```python
from posthog.ai.openai import OpenAI
import posthog

# Enable privacy mode to exclude prompts and responses from tracking
posthog.api_key = 'phc_your_project_api_key'
posthog.privacy_mode = True

client = OpenAI(api_key="your-openai-api-key")

# With privacy mode enabled, only usage metadata is tracked:
# - Token counts
# - Model used
# - Response times
# - Costs
# - Error rates
# 
# But NOT:
# - Prompt content
# - Response content
# - User data in messages

response = client.chat.completions.create(
    model="gpt-4",
    messages=[
        {"role": "user", "content": "Sensitive information here"}
    ]
)

# Only metadata tracked, content is not sent to PostHog
```

### Cost Tracking and Budgets

```python
from posthog.ai.openai import OpenAI
import posthog

posthog.api_key = 'phc_your_project_api_key'

class CostTracker:
    def __init__(self, budget_limit=100.0):
        self.total_cost = 0.0
        self.budget_limit = budget_limit
        self.client = OpenAI(api_key="your-openai-api-key")
    
    def track_usage(self, response):
        # Calculate approximate cost (varies by model)
        model_costs = {
            'gpt-4': {'input': 0.03, 'output': 0.06},  # per 1K tokens
            'gpt-3.5-turbo': {'input': 0.0015, 'output': 0.002}
        }
        
        model = response.model
        usage = response.usage
        
        if model in model_costs:
            input_cost = (usage.prompt_tokens / 1000) * model_costs[model]['input']
            output_cost = (usage.completion_tokens / 1000) * model_costs[model]['output']
            total_cost = input_cost + output_cost
            
            self.total_cost += total_cost
            
            # Track cost metrics
            posthog.capture('ai_cost_tracking', {
                'session_cost': total_cost,
                'total_cost': self.total_cost,
                'budget_remaining': self.budget_limit - self.total_cost,
                'model': model,
                'tokens_used': usage.total_tokens
            })
            
            # Budget alerts
            if self.total_cost > self.budget_limit * 0.8:
                posthog.capture('ai_budget_alert', {
                    'alert_type': 'approaching_limit',
                    'usage_percentage': (self.total_cost / self.budget_limit) * 100
                })
    
    def generate_text(self, messages, **kwargs):
        if self.total_cost >= self.budget_limit:
            raise Exception("Budget limit exceeded")
        
        response = self.client.chat.completions.create(
            messages=messages,
            **kwargs
        )
        
        self.track_usage(response)
        return response

# Usage
tracker = CostTracker(budget_limit=50.0)

response = tracker.generate_text(
    messages=[{"role": "user", "content": "Write a summary"}],
    model="gpt-4"
)
```

## Advanced Features

### Custom Metrics and Analysis

```python
from posthog.ai.openai import OpenAI
import posthog
import time

class AIAnalytics:
    def __init__(self):
        self.client = OpenAI(api_key="your-openai-api-key")
        
    def analyze_response_quality(self, prompt, response_text):
        """Analyze response quality metrics"""
        return {
            'length': len(response_text),
            'sentences': response_text.count('.'),
            'complexity_score': len(set(response_text.split())) / len(response_text.split()),
            'prompt_relevance': self._calculate_relevance(prompt, response_text)
        }
    
    def _calculate_relevance(self, prompt, response):
        # Simple relevance calculation
        prompt_words = set(prompt.lower().split())
        response_words = set(response.lower().split())
        overlap = len(prompt_words.intersection(response_words))
        return overlap / len(prompt_words) if prompt_words else 0
    
    def generate_with_analytics(self, messages, **kwargs):
        start_time = time.time()
        
        with posthog.new_context():
            posthog.tag('ai_analytics', True)
            
            response = self.client.chat.completions.create(
                messages=messages,
                **kwargs
            )
            
            end_time = time.time()
            response_time = end_time - start_time
            
            # Analyze response quality
            quality_metrics = self.analyze_response_quality(
                messages[-1]['content'],
                response.choices[0].message.content
            )
            
            # Track comprehensive analytics
            posthog.capture('ai_detailed_analytics', {
                'response_time': response_time,
                'model': response.model,
                'tokens_per_second': response.usage.total_tokens / response_time,
                **quality_metrics
            })
            
            return response

# Usage
analytics = AIAnalytics()
response = analytics.generate_with_analytics(
    messages=[{"role": "user", "content": "Explain machine learning"}],
    model="gpt-4"
)
```

### A/B Testing AI Models

```python
from posthog.ai.openai import OpenAI
import posthog
import random

class ModelABTester:
    def __init__(self):
        self.client = OpenAI(api_key="your-openai-api-key")
        self.models = {
            'gpt-4': {'weight': 0.3, 'cost_per_token': 0.00003},
            'gpt-3.5-turbo': {'weight': 0.7, 'cost_per_token': 0.000002}
        }
    
    def select_model(self, user_id):
        """Select model based on A/B test configuration"""
        # Use PostHog feature flags for A/B testing
        variant = posthog.get_feature_flag('ai_model_test', user_id)
        
        if variant == 'premium':
            return 'gpt-4'
        elif variant == 'standard':
            return 'gpt-3.5-turbo'
        else:
            # Fallback to weighted random selection
            return random.choices(
                list(self.models.keys()),
                weights=[m['weight'] for m in self.models.values()]
            )[0]
    
    def generate_with_testing(self, user_id, messages, **kwargs):
        model = self.select_model(user_id)
        
        with posthog.new_context():
            posthog.identify_context(user_id)
            posthog.tag('ab_test', 'ai_model_test')
            posthog.tag('model_variant', model)
            
            response = self.client.chat.completions.create(
                model=model,
                messages=messages,
                **kwargs
            )
            
            # Track A/B test metrics
            posthog.capture('ai_ab_test_result', {
                'model_used': model,
                'tokens_used': response.usage.total_tokens,
                'estimated_cost': response.usage.total_tokens * self.models[model]['cost_per_token'],
                'response_quality': len(response.choices[0].message.content)
            })
            
            return response

# Usage
tester = ModelABTester()
response = tester.generate_with_testing(
    'user123',
    messages=[{"role": "user", "content": "Help me write code"}]
)
```

## Best Practices

### Performance Optimization

```python
# Use appropriate models for different use cases
quick_client = OpenAI()  # For simple tasks
advanced_client = OpenAI()  # For complex reasoning

# Simple tasks
simple_response = quick_client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[{"role": "user", "content": "Summarize this text"}],
    max_tokens=100
)

# Complex tasks
complex_response = advanced_client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "Analyze this complex problem"}],
    max_tokens=1000
)
```

### Error Handling

```python
from posthog.ai.openai import OpenAI
import posthog

client = OpenAI(api_key="your-openai-api-key")

def robust_ai_call(messages, max_retries=3, **kwargs):
    for attempt in range(max_retries):
        try:
            with posthog.new_context():
                posthog.tag('attempt', attempt + 1)
                
                return client.chat.completions.create(
                    messages=messages,
                    **kwargs
                )
        except Exception as e:
            posthog.capture('ai_retry', {
                'attempt': attempt + 1,
                'error': str(e),
                'max_retries': max_retries
            })
            
            if attempt == max_retries - 1:
                raise
            
            time.sleep(2 ** attempt)  # Exponential backoff

# Usage with automatic retry and tracking
response = robust_ai_call(
    messages=[{"role": "user", "content": "Hello"}],
    model="gpt-4"
)
```

### Data Privacy and Security

```python
import posthog
from posthog.ai.openai import OpenAI

# Enable privacy mode globally
posthog.privacy_mode = True

# Or use environment-specific configuration
import os
if os.getenv('ENVIRONMENT') == 'production':
    posthog.privacy_mode = True

client = OpenAI(api_key="your-openai-api-key")

# With privacy mode, only metadata is tracked
response = client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "Confidential information"}]
)

# Manual tracking of non-sensitive metrics
posthog.capture('ai_usage_summary', {
    'model': 'gpt-4',
    'tokens_used': response.usage.total_tokens,
    'use_case': 'content_generation',
    'user_satisfaction': 'high'  # Can be collected separately
})
```