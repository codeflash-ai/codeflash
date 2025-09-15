async def get_endpoint(session, url):
    async with session.get(url) as response:
        return await response.text()


async def some_api_call(urls):
    async with aiohttp.ClientSession() as session:
        results = []
        for url in urls:
            result = await get_endpoint(session, url)
            results.append(result)
        return results
