"""Regression for upstream AgentInferencer's missing native persistence API."""
import json
from pathlib import Path
import tempfile
import unittest

from opencompass.openicl.icl_inferencer.icl_agent_inferencer import AgentInferencerOutputHandler


class AgentHandlerTest(unittest.TestCase):
    def test_empty_restore(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(AgentInferencerOutputHandler().restore_from_jsonl(tmp,'absent.jsonl'),{})

    def test_multiround_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            first=AgentInferencerOutputHandler()
            first.save_multiround_results('q1','answer1',[{'type':'IPythonInterpreter'}],0,'reference1')
            first.save_multiround_results('q2','answer2',[],0,'reference2')
            first.write_to_jsonl(tmp,'rows.jsonl')
            restored=AgentInferencerOutputHandler()
            self.assertEqual(restored.restore_from_jsonl(tmp,'rows.jsonl'),first.results_dict)
            restored.write_to_json(tmp,'final.json')
            self.assertEqual(json.loads((Path(tmp)/'final.json').read_text()),first.results_dict)

    def test_no_duplicate_dump(self):
        with tempfile.TemporaryDirectory() as tmp:
            handler=AgentInferencerOutputHandler()
            handler.save_results('q','a',[],0)
            handler.write_to_jsonl(tmp,'rows.jsonl')
            handler.write_to_jsonl(tmp,'rows.jsonl')
            self.assertEqual(len((Path(tmp)/'rows.jsonl').read_text().splitlines()),1)


if __name__=='__main__':unittest.main()
