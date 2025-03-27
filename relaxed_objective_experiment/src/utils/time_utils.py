from datetime import datetime, timezone


def rfc3339_to_unix_milliseconds(rfc3339_time):
    dt = datetime.fromisoformat(rfc3339_time.replace("Z", "+00:00"))
    dt_utc = dt.astimezone(timezone.utc)
    unix_milliseconds = int(dt_utc.timestamp() * 1000)
    return unix_milliseconds
