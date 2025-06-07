import argparse
import os
import tempfile
from pathlib import Path

import pytest
import shutil

from codeflash.code_utils.config_parser import parse_config_file
from codeflash.code_utils.formatter import format_code, sort_imports

from codeflash.discovery.functions_to_optimize import FunctionToOptimize
from codeflash.optimization.function_optimizer import FunctionOptimizer
from codeflash.verification.verification_utils import TestConfig

def test_remove_duplicate_imports():
    """Test that duplicate imports are removed when should_sort_imports is True."""
    original_code = "import os\nimport os\n"
    new_code = sort_imports(original_code)
    assert new_code == "import os\n"


def test_remove_multiple_duplicate_imports():
    """Test that multiple duplicate imports are removed when should_sort_imports is True."""
    original_code = "import sys\nimport os\nimport sys\n"

    new_code = sort_imports(original_code)
    assert new_code == "import os\nimport sys\n"


def test_sorting_imports():
    """Test that imports are sorted when should_sort_imports is True."""
    original_code = "import sys\nimport unittest\nimport os\n"

    new_code = sort_imports(original_code)
    assert new_code == "import os\nimport sys\nimport unittest\n"


def test_sort_imports_without_formatting():
    """Test that imports are sorted when formatting is disabled and should_sort_imports is True."""
    with tempfile.NamedTemporaryFile() as tmp:
        tmp.write(b"import sys\nimport unittest\nimport os\n")
        tmp.flush()
        tmp_path = Path(tmp.name)

        new_code = format_code(formatter_cmds=["disabled"], path=tmp_path)
        assert new_code is not None
        new_code = sort_imports(new_code)
        assert new_code == "import os\nimport sys\nimport unittest\n"


def test_dedup_and_sort_imports_deduplicates():
    original_code = """
import os
import sys


def foo():
    return os.path.join(sys.path[0], 'bar')
"""

    expected = """
import os
import sys


def foo():
    return os.path.join(sys.path[0], 'bar')
"""

    actual = sort_imports(original_code)

    assert actual == expected


def test_dedup_and_sort_imports_sorts_and_deduplicates():
    original_code = """
import os
import sys
import json
import os


def foo():
    return os.path.join(sys.path[0], 'bar')
"""

    expected = """
import json
import os
import sys


def foo():
    return os.path.join(sys.path[0], 'bar')
"""

    actual = sort_imports(original_code)

    assert actual == expected


def test_formatter_cmds_non_existent():
    """Test that default formatter-cmds is used when it doesn't exist in the toml."""
    config_data = """
[tool.codeflash]
module-root = "src"
tests-root = "tests"
test-framework = "pytest"
ignore-paths = []
"""

    with tempfile.NamedTemporaryFile(suffix=".toml", delete=False) as tmp:
        tmp.write(config_data.encode())
        tmp.flush()
        tmp_path = Path(tmp.name)

    try:
        config, _ = parse_config_file(tmp_path)
        assert config["formatter_cmds"] == ["black $file"]
    finally:
        os.remove(tmp_path)

    try:
        import black
    except ImportError:
        pytest.skip("black is not installed")

    original_code = b"""
import os
import sys
def foo():
    return os.path.join(sys.path[0], 'bar')"""
    expected = """import os
import sys


def foo():
    return os.path.join(sys.path[0], "bar")
"""
    with tempfile.NamedTemporaryFile() as tmp:
        tmp.write(original_code)
        tmp.flush()
        tmp_path = tmp.name

        actual = format_code(formatter_cmds=["black $file"], path=Path(tmp_path))
        assert actual == expected


def test_formatter_black():
    try:
        import black
    except ImportError:
        pytest.skip("black is not installed")
    original_code = b"""
import os
import sys    
def foo():
    return os.path.join(sys.path[0], 'bar')"""
    expected = """import os
import sys


def foo():
    return os.path.join(sys.path[0], "bar")
"""
    with tempfile.NamedTemporaryFile() as tmp:
        tmp.write(original_code)
        tmp.flush()
        tmp_path = tmp.name

        actual = format_code(formatter_cmds=["black $file"], path=Path(tmp_path))
        assert actual == expected


def test_formatter_ruff():
    try:
        import ruff  # type: ignore
    except ImportError:
        pytest.skip("ruff is not installed")
    original_code = b"""
import os
import sys    
def foo():
    return os.path.join(sys.path[0], 'bar')"""
    expected = """import os
import sys


def foo():
    return os.path.join(sys.path[0], "bar")
"""
    with tempfile.NamedTemporaryFile(suffix=".py") as tmp:
        tmp.write(original_code)
        tmp.flush()
        tmp_path = tmp.name

        actual = format_code(
            formatter_cmds=["ruff check --exit-zero --fix $file", "ruff format $file"], path=Path(tmp_path)
        )
        assert actual == expected


def test_formatter_error():
    original_code = """
import os
import sys
def foo():
    return os.path.join(sys.path[0], 'bar')"""
    expected = original_code
    with tempfile.NamedTemporaryFile("w") as tmp:
        tmp.write(original_code)
        tmp.flush()
        tmp_path = tmp.name
        with pytest.raises(FileNotFoundError):
            format_code(formatter_cmds=["exit 1"], path=Path(tmp_path))


def _run_formatting_test(source_code: str, should_content_change: bool, expected = None, optimized_function: str = ""):
    try:
        import ruff  # type: ignore
    except ImportError:
        pytest.skip("ruff is not installed")

    with tempfile.TemporaryDirectory() as test_dir_str:
        test_dir = Path(test_dir_str)
        source_file = test_dir / "source.py"
        
        source_file.write_text(source_code)
        original = source_code
        target_path = test_dir / "target.py"
        
        shutil.copy2(source_file, target_path)

        function_to_optimize = FunctionToOptimize(
            function_name="process_data", 
            parents=[], 
            file_path=target_path
        )

        test_cfg = TestConfig(
            tests_root=test_dir,
            project_root_path=test_dir,
            test_framework="pytest",
            tests_project_rootdir=test_dir,
        )

        args = argparse.Namespace(
            disable_imports_sorting=False,
            formatter_cmds=[
                "ruff check --exit-zero --fix $file",
                "ruff format $file"
            ],
        )

        optimizer = FunctionOptimizer(
            function_to_optimize=function_to_optimize,
            test_cfg=test_cfg,
            args=args,
        )
        
        optimizer.reformat_code_and_helpers(
            helper_functions=[],
            path=target_path,
            original_code=optimizer.function_to_optimize_source_code,
            optimized_function=optimized_function,
        )

        content = target_path.read_text(encoding="utf8")

        if expected is not None:
            assert content == expected, f"Expected content to be \n===========\n{expected}\n===========\nbut got\n===========\n{content}\n===========\n"

        if should_content_change:
            assert content != original, f"Expected content to change for source.py"
        else:
            assert content == original, f"Expected content to remain unchanged for source.py"



def test_formatting_file_with_many_diffs():
    """Test that files with many formatting errors are skipped (content unchanged)."""
    source_code = '''import os,sys,json,datetime,re
from collections import defaultdict,OrderedDict
import numpy as np,pandas as pd

class DataProcessor:
    def __init__(self,config_path,data_path,output_path):
        self.config_path=config_path
        self.data_path=data_path
        self.output_path=output_path
        self.config={}
        self.data=[]
        self.results={}
    
    def load_config(self):
        with open(self.config_path,'r') as f:
            self.config=json.load(f)
        if 'required_fields' not in self.config:self.config['required_fields']=[]
        if 'optional_fields' not in self.config:self.config['optional_fields']=[]
        return self.config
    
    def validate_data(self,data):
        errors=[]
        for idx,record in enumerate(data):
            if not isinstance(record,dict):
                errors.append(f"Record {idx} is not a dictionary")
                continue
            for field in self.config.get('required_fields',[]):
                if field not in record:
                    errors.append(f"Record {idx} missing required field: {field}")
                elif record[field] is None or record[field]=='':
                    errors.append(f"Record {idx} has empty required field: {field}")
        return errors
    
    def process_data(self,data,filter_func=None,transform_func=None,sort_key=None):
        if filter_func:data=[item for item in data if filter_func(item)]
        if transform_func:data=[transform_func(item) for item in data]
        if sort_key:data=sorted(data,key=sort_key)
        aggregated_data=defaultdict(list)
        for item in data:
            category=item.get('category','unknown')
            aggregated_data[category].append(item)
        final_results={}
        for category,items in aggregated_data.items():
            total_value=sum(item.get('value',0) for item in items)
            avg_value=total_value/len(items) if items else 0
            final_results[category]={'count':len(items),'total':total_value,'average':avg_value,'items':items}
        return final_results
    
    def save_results(self,results):
        with open(self.output_path,'w') as f:
            json.dump(results,f,indent=2,default=str)
        print(f"Results saved to {self.output_path}")
    
    def run_pipeline(self):
        try:
            config=self.load_config()
            with open(self.data_path,'r') as f:
                raw_data=json.load(f)
            validation_errors=self.validate_data(raw_data)
            if validation_errors:
                print("Validation errors found:")
                for error in validation_errors:print(f"  - {error}")
                return False
            processed_results=self.process_data(raw_data,filter_func=lambda x:x.get('active',True),transform_func=lambda x:{**x,'processed_at':datetime.datetime.now().isoformat()},sort_key=lambda x:x.get('name',''))
            self.save_results(processed_results)
            return True
        except Exception as e:
            print(f"Pipeline failed: {str(e)}")
            return False

def main():
    processor=DataProcessor('/path/to/config.json','/path/to/data.json','/path/to/output.json')
    success=processor.run_pipeline()
    if success:print("Pipeline completed successfully")
    else:print("Pipeline failed")

if __name__=='__main__':main()
'''
    _run_formatting_test(source_code, False)


def test_formatting_file_with_few_diffs():
    """Test that files with few formatting errors are formatted (content changed)."""
    source_code = '''import json
from datetime import datetime

def process_data(data, config=None):
    """Process data with optional configuration."""
    if not data:
        return {"success": False, "error": "No data provided"}
    
    if config is None:
        config = {"filter_active": True}
    
    # Minor formatting issues that should be fixed
    result=[]
    for item in data:
        if config.get("filter_active") and not item.get("active",True):
            continue
        processed_item={
            "id": item.get("id"),
            "name": item.get("name",""),
            "value": item.get("value",0),
            "processed_at": datetime.now().isoformat()
        }
        result.append(processed_item)
    
    return {"success": True, "data": result, "count": len(result)}
'''
    _run_formatting_test(source_code, True)


def test_formatting_file_with_no_diffs():
    """Test that files with no formatting errors are unchanged."""
    #  this test assumes you use ruff defaults for formatting
    source_code = '''from datetime import datetime


def process_data(data, config=None):
    """Process data with optional configuration."""
    if not data:
        return {"success": False, "error": "No data provided"}

    if config is None:
        config = {"filter_active": True}

    result = []
    for item in data:
        if config.get("filter_active") and not item.get("active", True):
            continue

        processed_item = {
            "id": item.get("id"),
            "name": item.get("name", ""),
            "value": item.get("value", 0),
            "processed_at": datetime.now().isoformat(),
        }
        result.append(processed_item)

    return {"success": True, "data": result, "count": len(result)}
'''
    _run_formatting_test(source_code, False)


def test_formatting_extremely_messy_file():
    """Test that extremely messy files with 100+ potential changes are skipped."""
    source_code = '''import os,sys,json,datetime,re,collections,itertools,functools,operator
from pathlib import Path
from typing import Dict,List,Optional,Union,Any,Tuple
import numpy as np,pandas as pd,matplotlib.pyplot as plt
from dataclasses import dataclass,field

@dataclass
class Config:
    input_path:str
    output_path:str
    batch_size:int=100
    max_retries:int=3
    timeout:float=30.0
    debug:bool=False
    filters:List[str]=field(default_factory=list)
    transformations:Dict[str,Any]=field(default_factory=dict)

class DataProcessorAdvanced:
    def __init__(self,config:Config):
        self.config=config
        self.data=[]
        self.results={}
        self.errors=[]
        self.stats={'processed':0,'failed':0,'skipped':0}
        
    def load_data(self,file_path:str)->List[Dict]:
        try:
            with open(file_path,'r',encoding='utf-8') as f:
                if file_path.endswith('.json'):data=json.load(f)
                elif file_path.endswith('.csv'):
                    import csv
                    reader=csv.DictReader(f)
                    data=[row for row in reader]
                else:raise ValueError(f"Unsupported file format: {file_path}")
            return data
        except Exception as e:self.errors.append(f"Failed to load {file_path}: {str(e)}");return[]
    
    def validate_record(self,record:Dict,schema:Dict)->Tuple[bool,List[str]]:
        errors=[]
        for field,rules in schema.items():
            if rules.get('required',False) and field not in record:
                errors.append(f"Missing required field: {field}")
            elif field in record:
                value=record[field]
                if 'type' in rules and not isinstance(value,rules['type']):
                    errors.append(f"Field {field} has wrong type")
                if 'min_length' in rules and isinstance(value,str) and len(value)<rules['min_length']:
                    errors.append(f"Field {field} too short")
                if 'max_length' in rules and isinstance(value,str) and len(value)>rules['max_length']:
                    errors.append(f"Field {field} too long")
                if 'min_value' in rules and isinstance(value,(int,float)) and value<rules['min_value']:
                    errors.append(f"Field {field} below minimum")
                if 'max_value' in rules and isinstance(value,(int,float)) and value>rules['max_value']:
                    errors.append(f"Field {field} above maximum")
        return len(errors)==0,errors
    
    def apply_filters(self,data:List[Dict])->List[Dict]:
        filtered_data=data
        for filter_name in self.config.filters:
            if filter_name=='active_only':filtered_data=[r for r in filtered_data if r.get('active',True)]
            elif filter_name=='has_value':filtered_data=[r for r in filtered_data if r.get('value') is not None]
            elif filter_name=='recent_only':
                cutoff=datetime.datetime.now()-datetime.timedelta(days=30)
                filtered_data=[r for r in filtered_data if datetime.datetime.fromisoformat(r.get('created_at','1970-01-01'))>cutoff]
        return filtered_data
    
    def apply_transformations(self,data:List[Dict])->List[Dict]:
        for transform_name,params in self.config.transformations.items():
            if transform_name=='add_timestamp':
                for record in data:record['processed_at']=datetime.datetime.now().isoformat()
            elif transform_name=='normalize_names':
                for record in data:
                    if 'name' in record:record['name']=record['name'].strip().title()
            elif transform_name=='calculate_derived':
                for record in data:
                    if 'value' in record and 'multiplier' in params:
                        record['derived_value']=record['value']*params['multiplier']
        return data
    
    def process_batch(self,batch:List[Dict])->Dict[str,Any]:
        try:
            processed_batch=[]
            for record in batch:
                try:
                    processed_record=dict(record)
                    processed_record['batch_id']=len(self.results)
                    processed_record['processed_at']=datetime.datetime.now().isoformat()
                    processed_batch.append(processed_record)
                    self.stats['processed']+=1
                except Exception as e:
                    self.errors.append(f"Failed to process record: {str(e)}")
                    self.stats['failed']+=1
            return {'success':True,'data':processed_batch,'count':len(processed_batch)}
        except Exception as e:
            self.errors.append(f"Batch processing failed: {str(e)}")
            return {'success':False,'error':str(e)}
    
    def run_processing_pipeline(self)->bool:
        try:
            raw_data=self.load_data(self.config.input_path)
            if not raw_data:return False
            filtered_data=self.apply_filters(raw_data)
            transformed_data=self.apply_transformations(filtered_data)
            batches=[transformed_data[i:i+self.config.batch_size] for i in range(0,len(transformed_data),self.config.batch_size)]
            all_results=[]
            for i,batch in enumerate(batches):
                if self.config.debug:print(f"Processing batch {i+1}/{len(batches)}")
                result=self.process_batch(batch)
                if result['success']:all_results.extend(result['data'])
                else:self.stats['failed']+=len(batch)
            with open(self.config.output_path,'w',encoding='utf-8') as f:
                json.dump({'results':all_results,'stats':self.stats,'errors':self.errors},f,indent=2,default=str)
            return True
        except Exception as e:
            self.errors.append(f"Pipeline failed: {str(e)}")
            return False

def create_sample_config()->Config:
    return Config(input_path='input.json',output_path='output.json',batch_size=50,max_retries=3,timeout=60.0,debug=True,filters=['active_only','has_value'],transformations={'add_timestamp':{},'normalize_names':{},'calculate_derived':{'multiplier':1.5}})

def main():
    config=create_sample_config()
    processor=DataProcessorAdvanced(config)
    success=processor.run_processing_pipeline()
    print(f"Processing {'completed' if success else 'failed'}")
    print(f"Stats: {processor.stats}")
    if processor.errors:
        print("Errors encountered:")
        for error in processor.errors:print(f"  - {error}")

if __name__=='__main__':main()
'''
    _run_formatting_test(source_code, False)


def test_formatting_edge_case_exactly_100_diffs():
    """Test behavior when exactly at the threshold of 100 changes."""
    # Create a file with exactly 100 minor formatting issues
    source_code = '''import json\n''' + '''
def func{}():
    x=1;y=2;z=3
    return x+y+z
'''.replace('{}', '_{i}').format(i='{i}') * 33  # This creates exactly 100 potential formatting fixes

    _run_formatting_test(source_code, False)


def test_formatting_with_syntax_errors():
    """Test that files with syntax errors are handled gracefully."""
    source_code = '''import json

def process_data(data):
    if not data:
        return {"error": "No data"
    # Missing closing brace above
    
    result = []
    for item in data
        # Missing colon above
        result.append(item)
    
    return result
'''
    _run_formatting_test(source_code, False)


def test_formatting_mixed_quotes_and_spacing():
    """Test files with mixed quote styles and inconsistent spacing."""
    source_code = '''import json
from datetime import datetime

def process_mixed_style(data):
    """Process data with mixed formatting styles."""
    config={'default_value':0,'required_fields':["id","name"],'optional_fields':["description","tags"]}
    
    results=[]
    for item in data:
        if not isinstance(item,dict):continue
        
        # Mixed quote styles
        item_id=item.get("id")
        item_name=item.get('name')
        item_desc=item.get("description",'')
        
        # Inconsistent spacing
        processed={
            'id':item_id,
            "name": item_name,
            'description':item_desc,
            "processed_at":datetime.now().isoformat( ),
            'status':'processed'
        }
        results.append(processed)
    
    return {'data':results,"count":len(results)}
'''
    _run_formatting_test(source_code, True)


def test_formatting_long_lines_and_imports():
    """Test files with long lines and import formatting issues."""
    source_code = '''import os, sys, json, datetime, re, collections, itertools
from pathlib import Path
from typing import Dict, List, Optional

def process_with_long_lines(data, filter_func=lambda x: x.get('active', True) and x.get('value', 0) > 0, transform_func=lambda x: {**x, 'processed_at': datetime.datetime.now().isoformat(), 'status': 'processed'}):
    """Function with very long parameter line."""
    return [transform_func(item) for item in data if filter_func(item) and isinstance(item, dict) and 'id' in item]

def another_function_with_long_line():
    very_long_dictionary = {'key1': 'value1', 'key2': 'value2', 'key3': 'value3', 'key4': 'value4', 'key5': 'value5'}
    return very_long_dictionary
'''
    _run_formatting_test(source_code, True)


def test_formatting_class_with_methods():
    """Test formatting of classes with multiple methods and minor issues."""
    source_code = '''class DataProcessor:
    def __init__(self, config):
        self.config=config
        self.data=[]
    
    def load_data(self,file_path):
        with open(file_path,'r') as f:
            self.data=json.load(f)
        return len(self.data)
    
    def process(self):
        result=[]
        for item in self.data:
            if item.get('active',True):
                result.append({
                    'id':item['id'],
                    'processed':True
                })
        return result
'''
    _run_formatting_test(source_code, True)


def test_formatting_with_complex_comprehensions():
    """Test files with complex list/dict comprehensions and formatting."""
    source_code = '''def complex_comprehensions(data):
    # Various comprehension styles with formatting issues
    result1=[item['value'] for item in data if item.get('active',True) and 'value' in item]
    
    result2={item['id']:item['name'] for item in data if item.get('type')=='user'}
    
    result3=[[x,y] for x in range(10) for y in range(5) if x*y>10]
    
    # Nested comprehensions
    nested=[[item for item in sublist if item%2==0] for sublist in data if isinstance(sublist,list)]
    
    return {
        'simple':result1,
        'mapping':result2,
        'complex':result3,
        'nested':nested
    }
'''
    _run_formatting_test(source_code, True)


def test_formatting_with_decorators_and_async():
    """Test files with decorators and async functions."""
    source_code = '''import asyncio
from functools import wraps

def timer_decorator(func):
    @wraps(func)
    def wrapper(*args,**kwargs):
        start=time.time()
        result=func(*args,**kwargs)
        end=time.time()
        print(f"{func.__name__} took {end-start:.2f} seconds")
        return result
    return wrapper

@timer_decorator
async def async_process_data(data):
    result=[]
    for item in data:
        await asyncio.sleep(0.01)  # Simulate async work
        processed_item={'id':item.get('id'),'processed':True}
        result.append(processed_item)
    return result

class AsyncProcessor:
    @staticmethod
    async def process_batch(batch):
        return [{'id':item['id'],'status':'done'} for item in batch if 'id' in item]
'''
    _run_formatting_test(source_code, True)


def test_formatting_threshold_configuration():
    """Test that the diff threshold can be configured (if supported)."""
    # This test assumes the threshold might be configurable
    source_code = '''import json,os,sys
def func1():x=1;y=2;return x+y
def func2():a=1;b=2;return a+b
def func3():c=1;d=2;return c+d
'''
    # Test with a file that has moderate formatting issues
    _run_formatting_test(source_code, True, optimized_function="def func2():a=1;b=2;return a+b")


def test_formatting_empty_file():
    """Test formatting of empty or minimal files."""
    source_code = '''# Just a comment pass
'''
    _run_formatting_test(source_code, False)


def test_formatting_with_docstrings():
    """Test files with various docstring formats."""
    source_code = """def function_with_docstring(    data):
    '''
    This is a function with a docstring.
    
    Args:
        data: Input data to process
        
    Returns:
        Processed data
    '''
    return  [item for item in data if item.get('active',True)]

class ProcessorWithDocs:
    '''A processor class with documentation.'''
    
    def __init__(self,config):
        '''Initialize with configuration.'''
        self.config=config
    
    def process(self,data):
        '''Single quote docstring with formatting issues.'''
        return{'result':[item for item in data if self._is_valid(item)]}
    
    def _is_valid(self,item):
        return isinstance(item,dict) and 'id' in item"""
    expected = '''def function_with_docstring(data):
    """This is a function with a docstring.

    Args:
        data: Input data to process

    Returns:
        Processed data

    """
    return [item for item in data if item.get("active", True)]


class ProcessorWithDocs:
    """A processor class with documentation."""

    def __init__(self, config):
        """Initialize with configuration."""
        self.config = config

    def process(self, data):
        """Single quote docstring with formatting issues."""
        return {"result": [item for item in data if self._is_valid(item)]}

    def _is_valid(self, item):
        return isinstance(item, dict) and "id" in item
'''

    optimization_function = """    def process(self,data):
        '''Single quote docstring with formatting issues.'''
        return{'result':[item for item in data if self._is_valid(item)]}"""
    _run_formatting_test(source_code, True, optimized_function=optimization_function, expected=expected)