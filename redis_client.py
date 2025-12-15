import redis.asyncio as aioredis

redis_client = aioredis.from_url("redis://localhost:6379", decode_responses=True)