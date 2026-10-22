"""LiveKit server-side helpers: short-lived room tokens and control messages to the worker.

Credentials are read from Settings and never returned to clients; only the signed room
token (scoped to one room, short TTL) leaves the server.
"""

from __future__ import annotations

import json
from datetime import timedelta

from turnsense.config import Settings
from turnsense.tracing import iso, utc_now

EVENTS_TOPIC = "turnsense.events"
CONTROL_TOPIC = "turnsense.control"


def mint_room_token(settings: Settings, room: str, identity: str) -> tuple[str, str]:
    """Least-privilege grant: join this one room, publish mic audio and data, subscribe."""
    from livekit import api

    if not settings.livekit_configured:
        raise RuntimeError("LiveKit is not configured")
    ttl = timedelta(seconds=settings.token_ttl_s)
    token = (
        api.AccessToken(
            settings.livekit_api_key.get_secret_value(),  # type: ignore[union-attr]
            settings.livekit_api_secret.get_secret_value(),  # type: ignore[union-attr]
        )
        .with_identity(identity)
        .with_name("Participant")
        .with_ttl(ttl)
        .with_grants(
            api.VideoGrants(
                room_join=True,
                room=room,
                can_publish=True,
                can_subscribe=True,
                can_publish_data=True,
                can_publish_sources=["microphone"],
            )
        )
        .to_jwt()
    )
    return token, iso(utc_now() + ttl)


def _http_url(url: str) -> str:
    return "http" + url[2:] if url.startswith("ws") else url


async def send_control(settings: Settings, room: str, action: str) -> None:
    """Deliver ``{"action": ...}`` to the agent worker in ``room`` on the control topic."""
    from livekit import api

    lk = api.LiveKitAPI(
        _http_url(settings.livekit_url or ""),
        settings.livekit_api_key.get_secret_value(),  # type: ignore[union-attr]
        settings.livekit_api_secret.get_secret_value(),  # type: ignore[union-attr]
    )
    try:
        await lk.room.send_data(
            api.SendDataRequest(
                room=room,
                data=json.dumps({"action": action}).encode(),
                kind=api.DataPacket.Kind.RELIABLE,
                topic=CONTROL_TOPIC,
            )
        )
    finally:
        await lk.aclose()
