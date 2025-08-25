import pytest
import asyncio
from shocker import tasked


@pytest.mark.asyncio
async def test_tasked_basic():
    result = await tasked()
    assert result == "Tasked"


@pytest.mark.asyncio
async def test_tasked_gather():
    results = await asyncio.gather(*(tasked() for _ in range(5)))
    assert results == ["Tasked"] * 5


def test_tasked_many_parallel_invocations():
    async def run_many():
        tasks = [tasked() for _ in range(1000)]
        results = await asyncio.gather(*tasks)
        return results

    results = asyncio.run(run_many())
    assert len(results) == 1000, "Should return 1000 results"
    assert all(r == "Tasked" for r in results), "All results should be 'Tasked'"
