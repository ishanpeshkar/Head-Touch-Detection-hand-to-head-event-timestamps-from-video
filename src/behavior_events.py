"""Group raw touch events into higher-level "behavior" events.

The touch detector (head_touch_detector.py) only knows about individual
hand-to-head contacts. Some behaviors of interest are a *pattern* of several
contacts close together in time rather than one touch -- head banging being
the obvious one: several short contacts in quick succession, not one long
touch. This module sits between the raw touch events and everything
downstream (event CSVs, clip export) and turns a burst of close-together
touches into a single "head_banging" behavior event, passing an isolated
touch through unchanged as "head_touch".

Clustering is by *gap*, not a fixed sliding window: touches are chained into
the same cluster as long as each one starts within `window_seconds` of the
previous one's start, however long the whole bout runs. A cluster becomes
"head_banging" once it has at least `min_count` touches; smaller clusters are
reported as ordinary individual "head_touch" events. Both thresholds are
intuition, not measured against real head-banging footage (none was
available yet) -- expect to retune once real examples exist, the same way
the touch detector's own gate settings were tuned in Pass 6 (see README).

Two entry points, sharing the same clustering, for the two pipelines:
  * `cluster_repetitions` -- batch, given the whole finished list of events.
  * `LiveRepetitionAggregator` -- streaming, for live_demo.py. A cluster can
    only be closed once the stream has gone quiet for `window_seconds` --
    there's no way to know a bout is over any sooner than that. A knock-on
    effect: *every* behavior event, even a single isolated touch, is only
    finalized (and so only clipped) after that same quiet gap, adding a few
    seconds of latency and trailing footage to every clip. That's traded for
    one shared, simple definition of "done" that covers both a single touch
    and a bang bout. Lower --bang-window if that trade is unwelcome; it also
    makes the bang clustering itself stricter.
"""

DEFAULT_WINDOW_SECONDS = 4.0
DEFAULT_MIN_COUNT = 3


def _finish_cluster(cluster, min_count):
    if not cluster:
        return []
    if len(cluster) < min_count:
        return [{**e, "behavior": "head_touch", "tap_count": 1} for e in cluster]
    return [{
        "behavior": "head_banging",
        "start_time": cluster[0]["start_time"],
        "contact_time": cluster[0]["contact_time"],
        "end_time": cluster[-1]["end_time"],
        "hand": "/".join(sorted({e["hand"] for e in cluster})),
        "tap_count": len(cluster),
        "min_normalized_distance": min(e["min_normalized_distance"] for e in cluster),
    }]


def cluster_repetitions(events, window_seconds=DEFAULT_WINDOW_SECONDS, min_count=DEFAULT_MIN_COUNT):
    """Batch: turn a list of touch events (any hand, any order) into behavior
    events. Returns a new list sorted by start_time; the input is unchanged."""
    events = sorted(events, key=lambda e: e["start_time"])
    behaviors = []
    cluster = []
    for e in events:
        if cluster and e["start_time"] - cluster[-1]["start_time"] > window_seconds:
            behaviors.extend(_finish_cluster(cluster, min_count))
            cluster = []
        cluster.append(e)
    behaviors.extend(_finish_cluster(cluster, min_count))
    behaviors.sort(key=lambda e: e["start_time"])
    return behaviors


class LiveRepetitionAggregator:
    """Streaming version of cluster_repetitions, for live_demo.py.

    Feed each finished touch event with add(). Call tick(now) once per frame
    (same pattern as TouchDetector.tick) so a quiet gap can be noticed and the
    pending cluster closed. flush() closes whatever is left when the stream
    ends. See the module docstring for the latency this implies.
    """

    def __init__(self, window_seconds=DEFAULT_WINDOW_SECONDS, min_count=DEFAULT_MIN_COUNT):
        self.window_seconds = window_seconds
        self.min_count = min_count
        self._pending = []

    @property
    def active(self) -> bool:
        """True while a cluster is open, i.e. a clip should be recording."""
        return bool(self._pending)

    def add(self, event) -> None:
        self._pending.append(event)

    def tick(self, now):
        if self._pending and now - self._pending[-1]["end_time"] > self.window_seconds:
            return self._finish()
        return []

    def flush(self):
        return self._finish()

    def _finish(self):
        finished = _finish_cluster(self._pending, self.min_count)
        self._pending = []
        return finished
