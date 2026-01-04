from typing import Any
from schemas.core import StreamMessage
from redis_client import redis_client
from typeguard import typechecked

@typechecked
async def process_unread_messages(reader_role: str, stream_name: str, raw: Any) -> list[StreamMessage]:
    """
    This function takes redis messages in the format returned by XREADGROUP,
    acknowledges each message for the given reader role, and converts them into
    StreamMessage objects.
    """

    parsed_batch: list[StreamMessage] = []

    for stream_name, entries in raw:
        for msg_id, fields in entries:
            await redis_client.xack(
                stream_name,
                reader_role,
                msg_id
            )

            role = fields.get("role", None)
            text = fields.get("text", None)

            if role is None or text is None:
                continue

            message = StreamMessage(
                msg_id=msg_id,
                role=role,
                text=text
            )

            parsed_batch.append(message)

    return parsed_batch

@typechecked
def process_read_messages(raw: Any) -> list[StreamMessage]:
    """
    This function takes redis messages in the format returned by XREAD or XREVRANGE
    and converts them into StreamMessage objects.
    """

    parsed_batch: list[StreamMessage] = []

    for message in raw:
        msg_id, fields = message

        role = fields.get("role", None)
        text = fields.get("text", None)

        if role is None or text is None:
            continue

        message = StreamMessage(
            msg_id=msg_id,
            role=role,
            text=text
        )
        parsed_batch.append(message)

    return parsed_batch