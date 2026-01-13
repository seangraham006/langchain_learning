"""
A stream is an append-only log of events.  Each stream just stores data.

Each role creates one consumer group on the stream, and agents attach to that group.

While agents are connected, they act as consumers within the consumer group.

When data is appened to the stream, it becomes eligible to be read by each consumer group.

Consumer groups are not auto created.

Consumer groups do not just 'receive' the metadata pings, they keep polling the stream instead.
"""


TOWNHALL_STREAM = "townhall"
CONTEXT_WINDOW = 10
MAX_REPLIES_PER_AGENT = 10
REPLY_COOLDOWN_SECONDS = 10
EVENTS_BEFORE_SUMMARY = 10
SUMMARY_SOFT_LIMIT_WORDS = 100
RETRIES_FOR_SUMMARISATION = 3
SUMMARISATION_CHECK_COOLDOWN_SECONDS = 1
DB_PATH = "/home/sean/projects/langchain/data/townhall_memory.db"