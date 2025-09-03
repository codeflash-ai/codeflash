import asyncio

from codeflash.code_utils.codeflash_wrap_decorator import \
    codeflash_behavior_async


@codeflash_behavior_async
async def process_data_list(data_list):
    results = []
    
    for item in data_list:
        await asyncio.sleep(0.1)
        processed = item * 2 + 10
        results.append(processed)
    
    return results


async def main():
    data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    result = await process_data_list(data)
    print(f"Processed {len(result)} items: {result}")


if __name__ == "__main__":
    asyncio.run(main())
