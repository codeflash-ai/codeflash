# Cron Monitoring

Scheduled job monitoring with automatic check-ins, failure detection, and alerting for cron jobs and scheduled tasks with support for various scheduling systems.

## Capabilities

### Monitor Decorator

Automatic monitoring for scheduled functions using a decorator that handles check-ins and failure detection.

```python { .api }
def monitor(
    monitor_slug: str = None,
    **monitor_config
) -> Callable:
    """
    Decorator for automatic cron job monitoring.
    
    Parameters:
    - monitor_slug: Unique identifier for the monitor (auto-generated if None)
    - **monitor_config: Monitor configuration options
    
    Returns:
    Callable: Decorator function that wraps the target function
    """
```

**Usage Examples:**

```python
import sentry_sdk

# Simple monitoring with auto-generated slug
@sentry_sdk.monitor
def daily_cleanup():
    """Daily cleanup job - monitored automatically."""
    cleanup_temp_files()
    remove_old_logs()
    vacuum_database()

# Custom monitor slug and configuration
@sentry_sdk.monitor(
    monitor_slug="hourly-data-sync",
    schedule_type="crontab",
    schedule="0 * * * *",  # Every hour
    checkin_margin=5,      # 5 minute margin
    max_runtime=1800,      # 30 minute timeout
    timezone="UTC"
)
def sync_data():
    """Hourly data synchronization job."""
    fetch_external_data()
    process_updates()
    update_cache()

# Interval-based monitoring
@sentry_sdk.monitor(
    monitor_slug="process-queue",
    schedule_type="interval",
    schedule={"value": 5, "unit": "minute"},
    checkin_margin=2,
    max_runtime=300
)
def process_message_queue():
    """Process message queue every 5 minutes."""
    messages = get_pending_messages()
    for message in messages:
        process_message(message)
```

### Manual Check-ins

Manual control over monitor check-ins for complex scheduling scenarios or custom job runners.

```python { .api }
def capture_checkin(
    monitor_slug: str = None,
    check_in_id: str = None,
    status: MonitorStatus = None,
    duration: float = None,
    **monitor_config
) -> str:
    """
    Send a check-in for a cron monitor.
    
    Parameters:
    - monitor_slug: Unique monitor identifier
    - check_in_id: Check-in ID for updating existing check-in
    - status: Check-in status (ok, error, in_progress, timeout)
    - duration: Job duration in seconds
    - **monitor_config: Monitor configuration for auto-creation
    
    Returns:
    str: Check-in ID for future updates
    """
```

**Usage Examples:**

```python
import sentry_sdk
import time

def run_batch_job():
    # Start check-in
    check_in_id = sentry_sdk.capture_checkin(
        monitor_slug="batch-processing",
        status=sentry_sdk.MonitorStatus.IN_PROGRESS,
        schedule_type="crontab",
        schedule="0 2 * * *",  # Daily at 2 AM
        max_runtime=3600
    )
    
    start_time = time.time()
    
    try:
        # Run the actual job
        process_daily_batch()
        
        # Success check-in
        duration = time.time() - start_time
        sentry_sdk.capture_checkin(
            monitor_slug="batch-processing",
            check_in_id=check_in_id,
            status=sentry_sdk.MonitorStatus.OK,
            duration=duration
        )
        
    except Exception as e:
        # Failure check-in
        duration = time.time() - start_time
        sentry_sdk.capture_checkin(
            monitor_slug="batch-processing", 
            check_in_id=check_in_id,
            status=sentry_sdk.MonitorStatus.ERROR,
            duration=duration
        )
        raise

# Simple success check-in
def simple_job():
    try:
        perform_task()
        sentry_sdk.capture_checkin(
            monitor_slug="simple-task",
            status=sentry_sdk.MonitorStatus.OK
        )
    except Exception:
        sentry_sdk.capture_checkin(
            monitor_slug="simple-task",
            status=sentry_sdk.MonitorStatus.ERROR
        )
        raise
```

## Monitor Configuration

### Schedule Types

#### Crontab Schedule

```python
@sentry_sdk.monitor(
    schedule_type="crontab",
    schedule="0 0 * * *",      # Daily at midnight
    timezone="America/New_York"
)
def daily_report():
    generate_daily_report()
```

#### Interval Schedule

```python
@sentry_sdk.monitor(
    schedule_type="interval",
    schedule={"value": 30, "unit": "minute"},  # Every 30 minutes
    checkin_margin=5
)
def check_system_health():
    monitor_system_metrics()
```

### Monitor Status Types

```python { .api }
class MonitorStatus:
    OK = "ok"                    # Job completed successfully
    ERROR = "error"              # Job failed with an error
    IN_PROGRESS = "in_progress"  # Job is currently running
    TIMEOUT = "timeout"          # Job exceeded max_runtime
    UNKNOWN = "unknown"          # Unknown status
```

### Configuration Options

- **schedule_type**: "crontab" or "interval"
- **schedule**: Cron expression or interval object
- **timezone**: Timezone for schedule interpretation
- **checkin_margin**: Grace period in minutes for late jobs
- **max_runtime**: Maximum expected runtime in seconds
- **failure_issue_threshold**: Consecutive failures before alert
- **recovery_threshold**: Consecutive successes to clear alert

## Integration Examples

### Celery Beat Integration

```python
import sentry_sdk
from celery import Celery
from celery.schedules import crontab

app = Celery('tasks')

@app.task
@sentry_sdk.monitor(
    monitor_slug="celery-cleanup",
    schedule_type="crontab", 
    schedule="0 3 * * *",
    timezone="UTC"
)
def cleanup_task():
    """Celery task with Sentry monitoring."""
    cleanup_old_data()
    return "Cleanup completed"

# Celery Beat configuration
app.conf.beat_schedule = {
    'cleanup-task': {
        'task': 'tasks.cleanup_task',
        'schedule': crontab(hour=3, minute=0),
    },
}
```

### APScheduler Integration

```python
import sentry_sdk
from apscheduler.schedulers.blocking import BlockingScheduler

scheduler = BlockingScheduler()

@sentry_sdk.monitor(
    monitor_slug="apscheduler-job",
    schedule_type="interval",
    schedule={"value": 10, "unit": "minute"}
)
def scheduled_job():
    """APScheduler job with monitoring."""
    process_pending_tasks()

scheduler.add_job(
    func=scheduled_job,
    trigger="interval",
    minutes=10,
    id='my_job'
)

scheduler.start()
```

### Crontab Integration

```bash
# System crontab entry
0 */6 * * * /usr/bin/python /path/to/monitored_script.py
```

```python
#!/usr/bin/env python
# monitored_script.py
import sentry_sdk

sentry_sdk.init(dsn="your-dsn-here")

@sentry_sdk.monitor(
    monitor_slug="system-cron-job",
    schedule_type="crontab",
    schedule="0 */6 * * *",  # Every 6 hours
    max_runtime=1800
)
def main():
    """Script run by system cron."""
    perform_maintenance_tasks()

if __name__ == "__main__":
    main()
```

### Custom Job Runner Integration

```python
import sentry_sdk
import time
from datetime import datetime

class JobRunner:
    def __init__(self):
        self.jobs = []
    
    def add_job(self, func, schedule, monitor_slug):
        self.jobs.append({
            'func': func,
            'schedule': schedule,
            'monitor_slug': monitor_slug,
            'last_run': None
        })
    
    def run_job(self, job):
        """Run a single job with monitoring."""
        monitor_slug = job['monitor_slug']
        
        # Start check-in
        check_in_id = sentry_sdk.capture_checkin(
            monitor_slug=monitor_slug,
            status=sentry_sdk.MonitorStatus.IN_PROGRESS
        )
        
        start_time = time.time()
        
        try:
            # Execute the job
            job['func']()
            job['last_run'] = datetime.now()
            
            # Success check-in
            duration = time.time() - start_time
            sentry_sdk.capture_checkin(
                monitor_slug=monitor_slug,
                check_in_id=check_in_id,
                status=sentry_sdk.MonitorStatus.OK,
                duration=duration
            )
            
        except Exception as e:
            # Error check-in
            duration = time.time() - start_time
            sentry_sdk.capture_checkin(
                monitor_slug=monitor_slug,
                check_in_id=check_in_id,
                status=sentry_sdk.MonitorStatus.ERROR,
                duration=duration
            )
            # Log error but continue with other jobs
            print(f"Job {monitor_slug} failed: {e}")

# Usage
runner = JobRunner()
runner.add_job(
    func=lambda: backup_database(),
    schedule="0 2 * * *",
    monitor_slug="nightly-backup"
)
```

## Error Handling and Recovery

### Automatic Error Detection

The monitor decorator automatically detects and reports:
- Unhandled exceptions as ERROR status
- Function execution time for performance tracking
- Missing check-ins for scheduled jobs

### Recovery Scenarios

```python
import sentry_sdk
import time

@sentry_sdk.monitor(
    monitor_slug="resilient-job",
    schedule_type="interval",
    schedule={"value": 15, "unit": "minute"},
    failure_issue_threshold=3,    # Alert after 3 consecutive failures
    recovery_threshold=2          # Clear alert after 2 consecutive successes
)
def resilient_job():
    """Job with retry logic and monitoring."""
    max_retries = 3
    
    for attempt in range(max_retries):
        try:
            perform_critical_operation()
            return  # Success
        except Exception as e:
            if attempt == max_retries - 1:
                # Final attempt failed, let monitor record ERROR
                raise
            else:
                # Retry after delay
                time.sleep(30 * (attempt + 1))
```

## Best Practices

### Monitor Naming

Use descriptive, kebab-case monitor slugs:
- `daily-data-backup`
- `hourly-cache-refresh`
- `weekly-report-generation`

### Schedule Configuration

- Set appropriate `checkin_margin` for network delays
- Configure `max_runtime` based on historical job duration
- Use UTC timezone for consistency across environments

### Error Handling

- Always use try/catch blocks for critical jobs
- Implement retry logic for transient failures
- Send contextual information with check-ins

### Performance Monitoring

Monitor job performance trends:
- Track duration changes over time
- Set alerts for abnormal runtime increases
- Monitor resource usage during job execution

Cron monitoring provides comprehensive visibility into scheduled job health, enabling proactive maintenance and reliable automation workflows.