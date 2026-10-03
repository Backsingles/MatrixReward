import unittest
from unittest import mock

import numpy as np

from generation_prompts import generation_messages
from judge import JUDGE_MODEL_NAME, all_dimensions_prompt, make_judge, parse_verdicts
from reward import (
    Rubric, build_implicit_rubricwise_matrix, build_rubricwise_matrix, combine_weights, compute_reward,
    extract_matrix_weights, spatial_distance_rewards,
)


class MatrixRewardTest(unittest.TestCase):
    def test_single_order_matrix_and_reward(self):
        rubrics = [Rubric("quality", 3), Rubric("clarity", 1)]
        ranking = {"a": 2, "b": 1, "c": 0}

        def judge(_query, first, second, _rubric):
            return "A" if ranking[first] > ranking[second] else "B"

        matrix = build_rubricwise_matrix("q", ["a", "b", "c"], rubrics, judge)
        np.testing.assert_allclose(matrix, [[1, 1], [.5, .5], [0, 0]])
        reward, details = compute_reward("q", ["a", "b", "c"], rubrics, judge)
        np.testing.assert_allclose(reward, [1, .5, 0])
        np.testing.assert_allclose(details["combined_weights"], [.75, .25])
        self.assertEqual(details["combination_fallback"], 0.0)

    def test_least_squares_weight_combination(self):
        extracted = np.array([.6, .3, .1])
        human = np.array([.2, .2, .6])
        gram = np.array([[extracted @ extracted, extracted @ human],
                         [human @ extracted, human @ human]])
        rhs = np.array([extracted @ extracted, human @ human])
        coefficients = np.abs(np.linalg.solve(gram, rhs))
        coefficients /= coefficients.sum()
        np.testing.assert_allclose(
            combine_weights(extracted, human),
            coefficients[0] * extracted + coefficients[1] * human,
        )
        self.assertFalse(np.allclose(combine_weights(extracted, human), (extracted + human) / 2))

    def test_parallel_weight_vectors(self):
        diagnostics = {}
        np.testing.assert_allclose(
            combine_weights([2, 1], [4, 2], diagnostics=diagnostics),
            [2 / 3, 1 / 3],
        )
        self.assertEqual(diagnostics["combination_fallback"], 1.0)
        self.assertEqual(diagnostics["matrix_coefficient"], .5)
        self.assertEqual(diagnostics["human_coefficient"], .5)

    def test_disjoint_weight_fallback(self):
        diagnostics = {}
        np.testing.assert_allclose(
            combine_weights([1, 0], [0, 1], diagnostics=diagnostics),
            [.5, .5],
        )
        self.assertEqual(diagnostics["combination_fallback"], 0.0)

    def test_explicit_tie_splits_the_win(self):
        rubric = [Rubric("rubric", 1)]
        matrix = build_rubricwise_matrix("q", ["a", "b"], rubric,
                                         lambda *_: "TIE")
        np.testing.assert_allclose(matrix, [[.5], [.5]])
        np.testing.assert_allclose(spatial_distance_rewards(matrix, [1]), [.5, .5])

    def test_swapped_verdicts_are_aligned_to_original_answers(self):
        rubrics = [Rubric("quality", 3), Rubric("clarity", 1)]
        with mock.patch("reward.getrandbits", side_effect=[0, 1]):
            matrix = build_rubricwise_matrix("q", ["a", "b"], rubrics, lambda *_: "A")
        np.testing.assert_allclose(matrix, [[1, 0], [0, 1]])
        with mock.patch("reward.getrandbits", return_value=1):
            matrix = build_implicit_rubricwise_matrix(
                "q", ["a", "b"], rubrics, lambda *_: ["A", "B"],
            )
        np.testing.assert_allclose(matrix, [[0, 1], [1, 0]])

    def test_matrix_weights_ignore_constant_column(self):
        matrix = np.array([[1, .5, 0], [.5, .5, .5], [0, .5, 1]])
        weights = extract_matrix_weights(matrix)
        self.assertAlmostEqual(weights[1], 0)
        np.testing.assert_allclose(weights.sum(), 1)

    def test_invalid_weights(self):
        with self.assertRaises(ValueError):
            spatial_distance_rewards([[1, 0]], [1, -1])
        with self.assertRaises(ValueError):
            compute_reward("q", ["a"], [Rubric("x", 1)], lambda *_: "TIE",
                           manual_weights=[0])

    def test_prompt_adapter(self):
        calls = []
        def request(model_name, prompt):
            calls.append((model_name, prompt))
            return '{"winner":"TIE"}'
        judge = make_judge(request)
        self.assertEqual(judge("query", "one", "two", Rubric("quality", 1)), "TIE")
        self.assertEqual(JUDGE_MODEL_NAME, "qwen36_35b_a3b")
        self.assertEqual(calls[0][0], JUDGE_MODEL_NAME)
        self.assertIn("Dimension: quality", calls[0][1])
        self.assertIn("Response A:\none", calls[0][1])

    def test_implicit_prompt_and_verdicts(self):
        rubrics = [Rubric("accuracy", 3), Rubric("clarity", 2)]
        prompt = all_dimensions_prompt("question", "first", "second", rubrics)
        self.assertIn("EXACTLY 2 entries", prompt)
        self.assertIn("1. accuracy\n2. clarity", prompt)
        self.assertIn("Response A:\nfirst", prompt)
        self.assertEqual(parse_verdicts('{"winners":["A","TIE"]}', 2), ["A", "TIE"])
        with self.assertRaises(ValueError):
            parse_verdicts('{"winners":["A"]}', 2)

    def test_generation_answer_blocks(self):
        messages = generation_messages("q", [(2, "low"), (10, "top"), (4, "mid"), (4, "tie")])
        user = messages[1]["content"]
        self.assertLess(user.index("Rank 1; overall score 10/10:\ntop"),
                        user.index("Rank 2; overall score 4/10:\nmid"))
        self.assertIn("Rank 3; overall score 4/10:\ntie", user)
        self.assertIn("Rank 4; overall score 2/10:\nlow", user)


if __name__ == "__main__":
    unittest.main()
