import json
import unittest
from io import BytesIO
from unittest import mock

from generation_prompts import generation_messages, parse_generated_rubrics
from judge_client import JudgeClient
from prepare_verl_data import convert_record
from verl_reward import compute_score


class PromptAndJudgeTest(unittest.TestCase):
    def test_generated_rubrics_flow_to_training_and_reward(self):
        messages = generation_messages(
            "Which answer is correct?", [(10, "best"), (8, "good"), (3, "weak"), (1, "worst")],
        )
        self.assertIn('"rubrics"', messages[1]["content"])
        generated = json.dumps({"rubrics": [
            {"text": "Correctness", "weight": 3},
            {"text": "Completeness", "weight": 2},
            {"text": "Clarity", "weight": 1},
        ]})
        row = {
            "group_id": 0,
            "prompt": [{"role": "user", "content": "Which answer is correct?"}],
            "query": "Which answer is correct?",
            "rubrics_json": json.dumps(parse_generated_rubrics(generated)),
        }
        self.assertEqual(json.loads(row["rubrics_json"]), [
            {"rubric": "Correctness", "points": 3},
            {"rubric": "Completeness", "points": 2},
            {"rubric": "Clarity", "points": 1},
        ])
        record = convert_record(row)

        def request(_model_name, prompt):
            first = prompt.split("Response A:\n", 1)[1].split("\n\nResponse B:", 1)[0]
            return json.dumps({"winner": "A" if first == "best" else "B"})

        rewards = compute_score(
            [{"solution_str": answer, "ground_truth": record["reward_model"]["ground_truth"]}
             for answer in ["best", "worst"]],
            num_generations=2, judge_workers=1, request=request,
        )
        self.assertEqual([item["score"] for item in rewards], [2.0, 1.0])

    def test_generated_rubric_schema_is_checked(self):
        valid = {"rubrics": [{"text": "a", "weight": 3},
                             {"text": "b", "weight": 2},
                             {"text": "c", "weight": 1}]}
        self.assertEqual(len(parse_generated_rubrics(valid)), 3)
        for bad in (
            {"rubrics": valid["rubrics"][:2]},
            {"rubrics": [*valid["rubrics"][:2], {"text": "c", "weight": True}]},
            {"rubrics": [*valid["rubrics"][:2], {"rubric": "c", "points": 1}]},
        ):
            with self.assertRaises(ValueError):
                parse_generated_rubrics(bad)


    def test_judge_request_contract(self):
        client = JudgeClient("https://example.invalid/chat/completions")
        response = BytesIO(b'{"choices":[{"message":{"content":"{\\"winner\\":\\"A\\"}"}}]}')
        with mock.patch("urllib.request.urlopen", return_value=response) as send:
            self.assertEqual(client("qwen36_35b_a3b", "prompt"), '{"winner":"A"}')
        payload = json.loads(send.call_args.args[0].data)
        self.assertEqual(payload["model"], "qwen36_35b_a3b")
        self.assertEqual(payload["chat_template_kwargs"], {"enable_thinking": False})
        self.assertEqual(payload["response_format"]["type"], "json_schema")


if __name__ == "__main__":
    unittest.main()
