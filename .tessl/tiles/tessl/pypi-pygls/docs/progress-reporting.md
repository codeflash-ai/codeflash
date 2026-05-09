# Progress Reporting

Built-in progress reporting system for long-running operations with client-side progress bar integration, cancellation support, and comprehensive progress lifecycle management.

## Capabilities

### Progress Class

Central progress management system that handles work-done progress reporting with client integration and cancellation support.

```python { .api }
class Progress:
    """
    Progress reporting system for long-running operations.
    
    Manages client-side progress bars, handles cancellation requests,
    and provides comprehensive progress lifecycle management for
    language server operations.
    """
    
    def __init__(self, lsp: LanguageServerProtocol) -> None:
        """
        Initialize progress manager.
        
        Parameters:
        - lsp: LanguageServerProtocol - Protocol instance for client communication
        """
    
    def create_task(
        self,
        token: ProgressToken,
        title: str,
        cancellable: bool = True,
        message: str = None,
        percentage: int = None
    ) -> Future:
        """
        Create progress task with client progress bar.
        
        Parameters:
        - token: ProgressToken - Unique token for progress tracking
        - title: str - Progress task title shown to user
        - cancellable: bool - Whether task can be cancelled (default: True)
        - message: str - Optional initial message
        - percentage: int - Optional initial percentage (0-100)
        
        Returns:
        Future that resolves when progress is created or fails
        """
    
    def update(
        self,
        token: ProgressToken,
        message: str = None,
        percentage: int = None
    ) -> None:
        """
        Update progress with new message or percentage.
        
        Parameters:
        - token: ProgressToken - Progress token to update
        - message: str - Optional progress message
        - percentage: int - Optional progress percentage (0-100)
        """
    
    def end(self, token: ProgressToken, message: str = None) -> None:
        """
        End progress reporting.
        
        Parameters:
        - token: ProgressToken - Progress token to end
        - message: str - Optional final message
        """
    
    @property
    def tokens(self) -> Dict[ProgressToken, Future]:
        """Access to active progress tokens and their futures."""
```

### Progress Message Types

Core progress message types for LSP work-done progress protocol implementation.

```python { .api }
# Progress message types from lsprotocol.types

class WorkDoneProgressBegin:
    """
    Progress begin notification structure.
    
    Attributes:
    - title: str - Progress title
    - cancellable: bool - Whether progress can be cancelled
    - message: str - Optional message
    - percentage: int - Optional percentage
    """

class WorkDoneProgressReport:
    """
    Progress update notification structure.
    
    Attributes:
    - message: str - Optional progress message
    - percentage: int - Optional percentage (0-100)
    - cancellable: bool - Whether progress can be cancelled
    """

class WorkDoneProgressEnd:
    """
    Progress end notification structure.
    
    Attributes:
    - message: str - Optional final message
    """

class WorkDoneProgressCreateParams:
    """
    Parameters for creating work-done progress.
    
    Attributes:
    - token: ProgressToken - Progress token
    """

class ProgressParams:
    """
    Progress notification parameters.
    
    Attributes:
    - token: ProgressToken - Progress token
    - value: Union[WorkDoneProgressBegin, WorkDoneProgressReport, WorkDoneProgressEnd]
    """
```

## Usage Examples

### Basic Progress Reporting

```python
import asyncio
import uuid
from pygls.server import LanguageServer
from pygls.progress import Progress

server = LanguageServer("progress-example", "1.0.0")

@server.command("myServer.longOperation")
async def long_operation(params):
    # Create unique progress token
    token = str(uuid.uuid4())
    
    # Start progress
    await server.lsp.progress.create_task(
        token=token,
        title="Processing Files",
        cancellable=True,
        message="Starting analysis..."
    )
    
    try:
        # Simulate long-running work with progress updates
        total_files = 100
        
        for i in range(total_files):
            # Check if operation was cancelled
            if token in server.lsp.progress.tokens:
                future = server.lsp.progress.tokens[token]
                if future.cancelled():
                    return {"cancelled": True}
            
            # Simulate file processing
            await asyncio.sleep(0.1)
            
            # Update progress
            percentage = int((i + 1) / total_files * 100)
            server.lsp.progress.update(
                token=token,
                message=f"Processing file {i + 1}/{total_files}",
                percentage=percentage
            )
        
        # End progress
        server.lsp.progress.end(
            token=token,
            message="Analysis completed successfully"
        )
        
        return {"result": "Operation completed", "files_processed": total_files}
        
    except Exception as e:
        # End progress with error
        server.lsp.progress.end(
            token=token,
            message=f"Operation failed: {str(e)}"
        )
        raise
```

### Progress with Cancellation Handling

```python
import time
from concurrent.futures import ThreadPoolExecutor

@server.command("myServer.heavyComputation")
@server.thread()
def heavy_computation(params):
    """Long-running computation with cancellation support."""
    token = str(uuid.uuid4())
    
    # Start progress on main thread
    asyncio.run_coroutine_threadsafe(
        server.lsp.progress.create_task(
            token=token,
            title="Heavy Computation",
            cancellable=True,
            message="Initializing computation..."
        ),
        server.loop
    )
    
    try:
        total_iterations = 1000
        
        for i in range(total_iterations):
            # Check cancellation status
            if token in server.lsp.progress.tokens:
                future = server.lsp.progress.tokens[token]
                if future.cancelled():
                    return {"cancelled": True, "iterations_completed": i}
            
            # Simulate heavy computation
            time.sleep(0.01)  # Blocking operation
            
            # Update progress every 10 iterations
            if i % 10 == 0:
                percentage = int(i / total_iterations * 100)
                
                # Update from thread
                def update_progress():
                    server.lsp.progress.update(
                        token=token,
                        message=f"Computing iteration {i}/{total_iterations}",
                        percentage=percentage
                    )
                
                asyncio.run_coroutine_threadsafe(
                    asyncio.create_task(update_progress()),
                    server.loop
                )
        
        # Complete progress
        def end_progress():
            server.lsp.progress.end(
                token=token,
                message="Computation completed"
            )
        
        asyncio.run_coroutine_threadsafe(
            asyncio.create_task(end_progress()),
            server.loop
        )
        
        return {"result": "Computation completed", "iterations": total_iterations}
        
    except Exception as e:
        # Handle error in progress
        def end_with_error():
            server.lsp.progress.end(
                token=token,
                message=f"Computation failed: {str(e)}"
            )
        
        asyncio.run_coroutine_threadsafe(
            asyncio.create_task(end_with_error()),
            server.loop
        )
        raise
```

### Multiple Progress Tasks

```python
@server.command("myServer.multiStepProcess")
async def multi_step_process(params):
    """Process with multiple progress bars for different steps."""
    
    # Step 1: File scanning
    scan_token = str(uuid.uuid4())
    await server.lsp.progress.create_task(
        token=scan_token,
        title="Scanning Files",
        cancellable=False,
        message="Scanning project directory..."
    )
    
    files = []
    for i in range(50):
        await asyncio.sleep(0.02)
        files.append(f"file_{i}.py")
        
        server.lsp.progress.update(
            token=scan_token,
            message=f"Found {len(files)} files",
            percentage=int(i / 50 * 100)
        )
    
    server.lsp.progress.end(scan_token, "File scanning completed")
    
    # Step 2: Analysis
    analysis_token = str(uuid.uuid4())
    await server.lsp.progress.create_task(
        token=analysis_token,
        title="Analyzing Files",
        cancellable=True,
        message="Starting analysis..."
    )
    
    analysis_results = {}
    for i, file in enumerate(files):
        # Check cancellation
        if analysis_token in server.lsp.progress.tokens:
            future = server.lsp.progress.tokens[analysis_token]
            if future.cancelled():
                return {"cancelled": True, "partial_results": analysis_results}
        
        await asyncio.sleep(0.05)
        analysis_results[file] = {"lines": i * 10, "functions": i * 2}
        
        percentage = int((i + 1) / len(files) * 100)
        server.lsp.progress.update(
            token=analysis_token,
            message=f"Analyzed {i + 1}/{len(files)} files",
            percentage=percentage
        )
    
    server.lsp.progress.end(analysis_token, "Analysis completed")
    
    return {
        "files_scanned": len(files),
        "analysis_results": analysis_results
    }
```

### Progress Context Manager

```python
from contextlib import asynccontextmanager

class ProgressManager:
    def __init__(self, server):
        self.server = server
    
    @asynccontextmanager
    async def progress_context(
        self, 
        title: str, 
        cancellable: bool = True,
        initial_message: str = None
    ):
        """Context manager for automatic progress lifecycle management."""
        token = str(uuid.uuid4())
        
        try:
            # Start progress
            await self.server.lsp.progress.create_task(
                token=token,
                title=title,
                cancellable=cancellable,
                message=initial_message or "Starting..."
            )
            
            # Yield progress controller
            controller = ProgressController(self.server, token)
            yield controller
            
        finally:
            # Always end progress
            self.server.lsp.progress.end(
                token=token,
                message="Operation completed"
            )

class ProgressController:
    def __init__(self, server, token):
        self.server = server
        self.token = token
    
    def update(self, message: str = None, percentage: int = None):
        """Update progress."""
        self.server.lsp.progress.update(
            token=self.token,
            message=message,
            percentage=percentage
        )
    
    def is_cancelled(self) -> bool:
        """Check if progress was cancelled."""
        if self.token in self.server.lsp.progress.tokens:
            future = self.server.lsp.progress.tokens[self.token]
            return future.cancelled()
        return False

# Usage with context manager
progress_manager = ProgressManager(server)

@server.command("myServer.contextManagedOperation")
async def context_managed_operation(params):
    async with progress_manager.progress_context(
        title="Context Managed Operation",
        cancellable=True,
        initial_message="Initializing..."
    ) as progress:
        
        for i in range(100):
            if progress.is_cancelled():
                return {"cancelled": True}
            
            await asyncio.sleep(0.01)
            
            progress.update(
                message=f"Step {i + 1}/100",
                percentage=i + 1
            )
        
        return {"result": "Operation completed successfully"}
```

### Custom Progress Notifications

```python
from lsprotocol.types import (
    PROGRESS,
    WINDOW_WORK_DONE_PROGRESS_CREATE,
    WorkDoneProgressBegin,
    WorkDoneProgressReport,
    WorkDoneProgressEnd
)

class CustomProgress:
    def __init__(self, server):
        self.server = server
        self.active_progress = {}
    
    async def start_custom_progress(
        self, 
        token: str, 
        title: str,
        custom_data: dict = None
    ):
        """Start progress with custom data."""
        
        # Create progress
        await self.server.lsp.send_request(
            WINDOW_WORK_DONE_PROGRESS_CREATE,
            {"token": token}
        )
        
        # Send begin with custom data
        begin_data = WorkDoneProgressBegin(
            title=title,
            cancellable=True,
            message="Custom progress started"
        )
        
        # Add custom fields
        if custom_data:
            begin_data.__dict__.update(custom_data)
        
        self.server.lsp.send_notification(
            PROGRESS,
            {"token": token, "value": begin_data}
        )
        
        self.active_progress[token] = {
            "title": title,
            "custom_data": custom_data
        }
    
    def update_custom_progress(
        self, 
        token: str, 
        percentage: int,
        custom_fields: dict = None
    ):
        """Update progress with custom fields."""
        
        report_data = WorkDoneProgressReport(
            percentage=percentage,
            message=f"Progress: {percentage}%"
        )
        
        # Add custom fields
        if custom_fields:
            report_data.__dict__.update(custom_fields)
        
        self.server.lsp.send_notification(
            PROGRESS,
            {"token": token, "value": report_data}
        )
    
    def end_custom_progress(self, token: str, final_data: dict = None):
        """End progress with custom final data."""
        
        end_data = WorkDoneProgressEnd(
            message="Custom progress completed"
        )
        
        if final_data:
            end_data.__dict__.update(final_data)
        
        self.server.lsp.send_notification(
            PROGRESS,
            {"token": token, "value": end_data}
        )
        
        self.active_progress.pop(token, None)

# Usage
custom_progress = CustomProgress(server)

@server.command("myServer.customProgressOperation")
async def custom_progress_operation(params):
    token = str(uuid.uuid4())
    
    await custom_progress.start_custom_progress(
        token=token,
        title="Custom Operation",
        custom_data={"operation_type": "analysis", "version": "2.0"}
    )
    
    for i in range(10):
        await asyncio.sleep(0.1)
        
        custom_progress.update_custom_progress(
            token=token,
            percentage=(i + 1) * 10,
            custom_fields={"current_step": f"step_{i}", "details": f"Processing item {i}"}
        )
    
    custom_progress.end_custom_progress(
        token=token,
        final_data={"total_processed": 10, "success": True}
    )
    
    return {"completed": True}
```