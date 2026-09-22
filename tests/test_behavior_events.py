"""Tests for behavior_events.py (no video needed): synthetic touch events in,
behavior events (head_touch / head_banging) out, batch and streaming."""

import os
import sys
import unittest

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, os.path.join(ROOT, "src"))

from behavior_events import LiveRepetitionAggregator, cluster_repetitions  # noqa: E402


def touch(start, end, hand="Right", contact=None):
    return {"hand": hand, "start_time": start, "contact_time": contact if contact is not None else start,
            "end_time": end, "min_normalized_distance": 0.1}


WINDOW, MIN_COUNT = 4.0, 3


class ClusterRepetitionsBatch(unittest.TestCase):
    def test_single_isolated_touch_stays_head_touch(self):
        events = [touch(1.0, 1.5)]
        behaviors = cluster_repetitions(events, WINDOW, MIN_COUNT)
        self.assertEqual(len(behaviors), 1)
        self.assertEqual(behaviors[0]["behavior"], "head_touch")
        self.assertEqual(behaviors[0]["tap_count"], 1)

    def test_two_touches_within_window_are_not_yet_banging(self):
        events = [touch(1.0, 1.5), touch(3.0, 3.5)]
        behaviors = cluster_repetitions(events, WINDOW, MIN_COUNT)
        self.assertEqual([b["behavior"] for b in behaviors], ["head_touch", "head_touch"])

    def test_three_touches_within_window_become_one_head_banging_event(self):
        events = [touch(1.0, 1.5), touch(3.0, 3.5), touch(5.0, 5.5)]
        behaviors = cluster_repetitions(events, WINDOW, MIN_COUNT)
        self.assertEqual(len(behaviors), 1)
        b = behaviors[0]
        self.assertEqual(b["behavior"], "head_banging")
        self.assertEqual(b["tap_count"], 3)
        self.assertEqual(b["start_time"], 1.0)
        self.assertEqual(b["end_time"], 5.5)

    def test_touches_far_apart_stay_separate_even_if_three(self):
        events = [touch(0.0, 0.5), touch(10.0, 10.5), touch(20.0, 20.5)]
        behaviors = cluster_repetitions(events, WINDOW, MIN_COUNT)
        self.assertEqual([b["behavior"] for b in behaviors], ["head_touch"] * 3)

    def test_a_bout_can_run_longer_than_the_window_via_chained_gaps(self):
        # Each gap is 3s (<= window 4s), so all five chain into one bout even
        # though the whole thing spans 12s, well past a single fixed window.
        events = [touch(t, t + 0.5) for t in (0.0, 3.0, 6.0, 9.0, 12.0)]
        behaviors = cluster_repetitions(events, WINDOW, MIN_COUNT)
        self.assertEqual(len(behaviors), 1)
        self.assertEqual(behaviors[0]["tap_count"], 5)

    def test_two_separate_bouts_are_reported_separately(self):
        bout1 = [touch(t, t + 0.5) for t in (0.0, 1.0, 2.0)]
        bout2 = [touch(t, t + 0.5) for t in (20.0, 21.0, 22.0)]
        behaviors = cluster_repetitions(bout1 + bout2, WINDOW, MIN_COUNT)
        self.assertEqual([b["behavior"] for b in behaviors], ["head_banging", "head_banging"])

    def test_mixed_hands_in_one_bout_are_reported_together(self):
        events = [touch(0.0, 0.5, hand="Right"), touch(1.0, 1.5, hand="Left"), touch(2.0, 2.5, hand="Right")]
        behaviors = cluster_repetitions(events, WINDOW, MIN_COUNT)
        self.assertEqual(behaviors[0]["behavior"], "head_banging")
        self.assertEqual(behaviors[0]["hand"], "Left/Right")

    def test_out_of_order_input_is_sorted_first(self):
        events = [touch(5.0, 5.5), touch(1.0, 1.5), touch(3.0, 3.5)]
        behaviors = cluster_repetitions(events, WINDOW, MIN_COUNT)
        self.assertEqual(len(behaviors), 1)
        self.assertEqual(behaviors[0]["start_time"], 1.0)


class LiveRepetitionAggregatorStreaming(unittest.TestCase):
    def test_matches_batch_clustering_once_flushed(self):
        events = [touch(1.0, 1.5), touch(3.0, 3.5), touch(5.0, 5.5)]
        agg = LiveRepetitionAggregator(WINDOW, MIN_COUNT)
        for e in events:
            self.assertEqual(agg.add(e), None)  # add() doesn't emit anything itself
        self.assertEqual(agg.tick(5.5), [])      # no quiet gap yet
        finished = agg.tick(5.5 + WINDOW + 0.1)  # now the gap has passed
        self.assertEqual(finished, cluster_repetitions(events, WINDOW, MIN_COUNT))

    def test_active_is_true_only_while_a_cluster_is_pending(self):
        agg = LiveRepetitionAggregator(WINDOW, MIN_COUNT)
        self.assertFalse(agg.active)
        agg.add(touch(1.0, 1.5))
        self.assertTrue(agg.active)
        agg.tick(1.5 + WINDOW + 0.1)
        self.assertFalse(agg.active)

    def test_flush_closes_a_cluster_still_open_when_the_stream_ends(self):
        agg = LiveRepetitionAggregator(WINDOW, MIN_COUNT)
        agg.add(touch(1.0, 1.5))
        agg.add(touch(2.0, 2.5))
        agg.add(touch(3.0, 3.5))
        finished = agg.flush()
        self.assertEqual(len(finished), 1)
        self.assertEqual(finished[0]["behavior"], "head_banging")
        self.assertFalse(agg.active)


if __name__ == "__main__":
    unittest.main()
