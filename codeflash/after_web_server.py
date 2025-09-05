async def get_endpoint(session, url):
    async with session.get(url) as response:
        return await response.text()


async def some_api_call(urls):
    async with aiohttp.ClientSession() as session:
        tasks = [get_endpoint(session, url) for url in urls]
        # Run requests concurrently
        results = await asyncio.gather(*tasks)
        return results
