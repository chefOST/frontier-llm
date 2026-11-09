import unittest

from evaluate_wandb_run import best_match_count, canonicalize, intervals_and_events, mapped_state_set


class EvaluationTest(unittest.TestCase):
    def test_documented_timing_example_and_interval_reset(self):
        matches, error = best_match_count([10, 20, 34], [11, 22, 40], 2)
        self.assertEqual((matches, error), (2, 3))

        eligible, intervals = intervals_and_events(
            "egg",
            [["egg"], ["egg"], [], ["egg"], ["egg"]],
            [["whole"], ["cracked"], [], ["whole"], ["cracked"]],
        )
        self.assertEqual(eligible, [True, True, False, True, True])
        self.assertEqual(intervals, [(0, 2, [1]), (3, 5, [4])])

    def test_longest_object_alias_and_unmapped_state_penalty(self):
        self.assertEqual(
            canonicalize("brownie batter in pan", {"batter": "batter", "brownie batter": "brownie batter"}),
            "brownie batter",
        )
        mappings = {"eggs": {"raw cracked egg": {"mapped": ["cracked"], "unmapped": ["raw"]}}}
        self.assertEqual(mapped_state_set("raw cracked egg", "eggs", mappings), {"cracked", "unmapped:raw"})


if __name__ == "__main__":
    unittest.main()
