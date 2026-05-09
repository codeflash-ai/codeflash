# IPython Integration

Enhanced integration with IPython and Jupyter notebooks for interactive memory profiling workflows. Provides magic commands and seamless notebook integration for data science and development workflows.

## Capabilities

### Extension Loading

Load memray magic commands in IPython or Jupyter environments.

```python { .api }
def load_ipython_extension(ipython):
    """
    Load memray magic commands in IPython/Jupyter.
    
    Parameters:
    - ipython: IPython instance
    
    Provides:
    - %%memray_flamegraph magic command for notebook profiling
    """
```

Usage in IPython/Jupyter:

```python
# Load the extension
%load_ext memray

# Now memray magic commands are available
```

### Magic Commands

#### %%memray_flamegraph

Cell magic for profiling code cells and generating inline flame graphs.

```python
%%memray_flamegraph [options]
# Code to profile
```

Options:
- `--output FILE`: Output file for flame graph HTML
- `--native`: Enable native stack traces
- `--leaks`: Show only leaked allocations
- `--merge-threads`: Merge allocations across threads

Usage examples:

```python
# Basic cell profiling
%%memray_flamegraph
import numpy as np
data = np.random.random((1000, 1000))
result = np.sum(data ** 2)
```

```python
# Profile with native traces and custom output
%%memray_flamegraph --native --output my_analysis.html
import pandas as pd
df = pd.read_csv('large_dataset.csv')
processed = df.groupby('category').sum()
```

```python
# Focus on memory leaks
%%memray_flamegraph --leaks
def potentially_leaky_function():
    cache = {}
    for i in range(10000):
        cache[i] = [0] * 1000
    # Forgot to clear cache
    return len(cache)

result = potentially_leaky_function()
```

## Notebook Workflow Examples

### Data Science Profiling

```python
# Load extension
%load_ext memray

# Profile data loading
%%memray_flamegraph --output data_loading.html
import pandas as pd
import numpy as np

# Load large dataset
df = pd.read_csv('big_data.csv')
print(f"Loaded {len(df)} rows")
```

```python
# Profile data processing
%%memray_flamegraph --output processing.html
# Heavy computation
df['computed'] = df['value'].apply(lambda x: x ** 2 + np.sin(x))
df_grouped = df.groupby('category').agg({
    'value': ['mean', 'std', 'sum'],
    'computed': ['mean', 'max']
})
```

```python
# Profile model training
%%memray_flamegraph --native --output model_training.html
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

X = df[['feature1', 'feature2', 'feature3']]
y = df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = RandomForestRegressor(n_estimators=100)
model.fit(X_train, y_train)
```

### Algorithm Development

```python
# Profile algorithm implementations
%%memray_flamegraph --output algorithm_comparison.html

def bubble_sort(arr):
    n = len(arr)
    for i in range(n):
        for j in range(0, n-i-1):
            if arr[j] > arr[j+1]:
                arr[j], arr[j+1] = arr[j+1], arr[j]
    return arr

def quick_sort(arr):
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return quick_sort(left) + middle + quick_sort(right)

# Test both algorithms
test_data_1 = list(range(1000, 0, -1))  # Reverse sorted
test_data_2 = test_data_1.copy()

sorted_1 = bubble_sort(test_data_1)
sorted_2 = quick_sort(test_data_2)
```

### Memory Leak Detection

```python
# Profile for memory leaks
%%memray_flamegraph --leaks --output leak_detection.html

class PotentiallyLeakyClass:
    _instances = []  # Class variable that holds references
    
    def __init__(self, data):
        self.data = data
        PotentiallyLeakyClass._instances.append(self)  # Creates leak
    
    def process(self):
        return sum(self.data)

# Create many instances without cleanup
results = []
for i in range(100):
    instance = PotentiallyLeakyClass(list(range(1000)))
    results.append(instance.process())

print(f"Processed {len(results)} instances")
print(f"Leaked instances: {len(PotentiallyLeakyClass._instances)}")
```

### Interactive Analysis

```python
# Use programmatic API for custom analysis
import memray

# Profile a cell programmatically
with memray.Tracker("notebook_profile.bin"):
    # Expensive operation
    large_dict = {i: [j for j in range(100)] for i in range(1000)}
    
# Analyze results in next cell
with memray.FileReader("notebook_profile.bin") as reader:
    print(f"Peak memory: {reader.metadata.peak_memory:,} bytes")
    print(f"Total allocations: {reader.metadata.total_allocations:,}")
    
    # Find largest allocations
    records = list(reader.get_allocation_records())
    largest = sorted(records, key=lambda r: r.size, reverse=True)[:5]
    
    print("\nLargest allocations:")
    for record in largest:
        print(f"  {record.size:,} bytes in thread {record.thread_name}")
```

### Custom Reporting

```python
def analyze_memory_profile(filename):
    """Custom analysis function for notebook use."""
    with memray.FileReader(filename) as reader:
        metadata = reader.metadata
        
        # Collect statistics
        total_size = 0
        allocator_counts = {}
        
        for record in reader.get_allocation_records():
            total_size += record.size
            allocator = record.allocator.name
            allocator_counts[allocator] = allocator_counts.get(allocator, 0) + 1
        
        return {
            'duration': (metadata.end_time - metadata.start_time).total_seconds(),
            'peak_memory': metadata.peak_memory,
            'total_allocated': total_size,
            'allocator_breakdown': allocator_counts
        }

# Use custom analysis
%%memray_flamegraph --output custom_analysis.html
# Code to analyze
import json
data = json.loads('{"key": "value"}' * 10000)

# Analyze the results
stats = analyze_memory_profile("custom_analysis.html")
print(f"Duration: {stats['duration']:.2f}s")
print(f"Peak memory: {stats['peak_memory']:,} bytes") 
print(f"Allocators used: {list(stats['allocator_breakdown'].keys())}")
```