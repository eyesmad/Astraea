import asyncio
async def test():
    async def task():
        raise ValueError("Hidden error")
    asyncio.create_task(task())
    await asyncio.sleep(1)
    print("Main loop still running!")
asyncio.run(test())
