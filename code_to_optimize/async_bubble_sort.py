import asyncio
from typing import List, Union


async def async_sorter(lst: List[Union[int, float]]) -> List[Union[int, float]]:
    """
    Async bubble sort implementation for testing.
    """
    print("codeflash stdout: Async sorting list")
    
    await asyncio.sleep(0.01)
    
    n = len(lst)
    for i in range(n):
        for j in range(0, n - i - 1):
            if lst[j] > lst[j + 1]:
                lst[j], lst[j + 1] = lst[j + 1], lst[j]
    
    result = lst.copy()
    print(f"result: {result}")
    return result


class AsyncBubbleSorter:
    """Class with async sorting method for testing."""
    
    async def sorter(self, lst: List[Union[int, float]]) -> List[Union[int, float]]:
        """
        Async bubble sort implementation within a class.
        """
        print("codeflash stdout: AsyncBubbleSorter.sorter() called")
        
        # Add some async delay
        await asyncio.sleep(0.005)
        
        n = len(lst)
        for i in range(n):
            for j in range(0, n - i - 1):
                if lst[j] > lst[j + 1]:
                    lst[j], lst[j + 1] = lst[j + 1], lst[j]
        
        result = lst.copy()
        return result
