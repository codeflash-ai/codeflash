import pytest
from workload import process_data_list


@pytest.mark.asyncio
async def test_process_data_list():
    data = [1, 2, 3]
    result = await process_data_list(data)
    expected = [12, 14, 16]  # (1*2+10), (2*2+10), (3*2+10)
    assert result == expected


@pytest.mark.asyncio
async def test_process_data_list_empty():
    result = await process_data_list([])
    assert result == []


@pytest.mark.asyncio
async def test_process_data_list_single():
    result = await process_data_list([5])
    assert result == [20]  # 5*2+10