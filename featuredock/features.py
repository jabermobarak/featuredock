"""Online/historical lookup. Both event time and knowledge time constrain every row."""

from .contracts import feature_row

AGGREGATES = """
    SUM(CASE WHEN event_at>:boundary THEN 1 ELSE 0 END) AS sample_count,
    SUM(CASE WHEN event_at<=:boundary THEN 1 ELSE 0 END) AS previous_count,
    AVG(CASE WHEN event_at>:boundary THEN temperature END) AS temperature_mean,
    MAX(CASE WHEN event_at>:boundary THEN temperature END) AS temperature_max,
    AVG(CASE WHEN event_at>:boundary THEN vibration END) AS vibration_mean,
    MAX(CASE WHEN event_at>:boundary THEN vibration END) AS vibration_max,
    AVG(CASE WHEN event_at>:boundary THEN load END) AS load_mean,
    AVG(CASE WHEN event_at<=:boundary THEN temperature END) AS previous_temperature_mean,
    MAX(CASE WHEN event_at>:boundary THEN event_at END) AS latest_event_at
"""


def lookup(store, asset_id, at, include_late=False):
    params = {"asset": asset_id, "at": at, "start": at - 7200, "boundary": at - 3600}
    known_filter = "" if include_late else " AND received_at<=:at"
    with store.connection() as db:
        values = dict(
            db.execute(
                f"SELECT {AGGREGATES} FROM events WHERE asset_id=:asset "
                f"AND event_at>:start AND event_at<=:at{known_filter}",
                params,
            ).fetchone()
        )
    return feature_row(asset_id, at, values)


def evidence(store, asset_id, at):
    with store.connection() as db:
        return [
            dict(r)
            for r in db.execute(
                "SELECT event_id,event_at,received_at,temperature,vibration,load,"
                "CASE WHEN received_at<=? THEN 1 ELSE 0 END AS known_at_prediction "
                "FROM events WHERE asset_id=? AND event_at>? AND event_at<=? "
                "ORDER BY event_at DESC LIMIT 30",
                (at, asset_id, at - 3600, at),
            )
        ]
