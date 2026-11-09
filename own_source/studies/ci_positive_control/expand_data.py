"""Expand the frozen CI corpus without changing the active run or its holdout."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

from inventory import DATASET, REVISION, digest
from select_data import assign_source_groups, check_trace, evaluation_overlap_checker, normalized


def sha256(path):
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def main(root, output, target):
    root, output = Path(root), Path(output)
    output.mkdir(parents=True, exist_ok=False)
    (output / 'data').mkdir()
    metadata = [json.loads(line) for p in sorted((root / 'data/inventory-shards').glob('*.jsonl'))
                for line in p.open()]
    assign_source_groups(metadata)
    index = {row['id']: row for row in metadata}
    overlap, screened = evaluation_overlap_checker(root)
    frozen = {split: [json.loads(line) for line in (root / f'data/{split}-raw.jsonl').open()]
              for split in ('train', 'validation')}
    for split, rows in frozen.items():
        for row in rows:
            meta = index[row['source']['id']]
            assert meta['task_group'] == row['provenance']['task_group']
            assert meta['split'] == split
    assert target > len(frozen['train'])
    validation_questions = {normalized(x['source']['question']) for x in frozen['validation']}
    question_seen = {normalized(x['source']['question']) for rows in frozen.values() for x in rows}
    trace_seen = {x['source_trace_sha256'] for rows in frozen.values() for x in rows}
    frozen_ids = {x['source']['id'] for rows in frozen.values() for x in rows}
    selected = list(frozen['train'])
    counts = Counter()
    disagreements = output / 'reference-disagreement-review.jsonl'
    with disagreements.open('x') as review:
        for shard in sorted((root / 'data/raw-shards').glob('*.jsonl')):
            for line in shard.open():
                source = json.loads(line)
                if source['id'] in frozen_ids:
                    counts['already_frozen'] += 1
                    continue
                meta = index[source['id']]
                if meta['split'] != 'train':
                    counts['reserved_validation_source_group'] += 1
                    continue
                if source['executor_type'] != 'e2b':
                    counts['simulated_executor_excluded'] += 1
                    continue
                if not normalized(source['answer']) or not normalized(source['question']):
                    counts['empty_question_or_reference'] += 1
                    continue
                reason = overlap(source['question'])
                if reason:
                    counts[reason] += 1
                    continue
                quality = check_trace(source)
                if isinstance(quality, str):
                    counts[quality] += 1
                    continue
                if not quality['reference_answer_verbatim_in_final']:
                    counts['reference_answer_not_verbatim_in_final'] += 1
                    review.write(json.dumps(dict(id=source['id'], task_group=meta['task_group'],
                        source_reference=source['answer'], final_answer=quality['final_answer'],
                        source_shard=str(shard), approved=False), ensure_ascii=False) + '\n')
                    continue
                question = normalized(source['question'])
                trace = digest(dict(messages=source['messages'], tools=source['tools']))
                if question in question_seen or trace in trace_seen:
                    counts['duplicate_question_or_trace'] += 1
                    continue
                if len(selected) >= target:
                    counts['quota_already_met'] += 1
                    continue
                question_seen.add(question)
                trace_seen.add(trace)
                selected.append(dict(source=source, provenance=meta, quality=quality,
                                     source_trace_sha256=trace))
    assert not {x['provenance']['task_group'] for x in selected} & {
        x['provenance']['task_group'] for x in frozen['validation']}
    assert not {normalized(x['source']['question']) for x in selected} & validation_questions
    with (output / 'data/train-raw.jsonl').open('x') as handle:
        for row in selected:
            handle.write(json.dumps(row, ensure_ascii=False) + '\n')
    # Byte-for-byte retain the fixed validation file, including its serialization.
    validation = output / 'data/validation-raw.jsonl'
    validation.write_bytes((root / 'data/validation-raw.jsonl').read_bytes())
    manifest = dict(dataset=DATASET, revision=REVISION, target_training_trajectories=target,
        actual_training_trajectories=len(selected), validation_trajectories=len(frozen['validation']),
        target_met=len(selected) >= target, original_2000_preserved_as_prefix=True,
        validation_identical=sha256(validation) == sha256(root / 'data/validation-raw.jsonl'),
        source_group_disjoint=True, eval_questions_screened=screened, exclusions=dict(counts),
        assistant_decisions=sum(x['quality']['assistant_decisions'] for x in selected),
        python_calls=sum(x['quality']['python_calls'] for x in selected),
        source_groups=len({x['provenance']['task_group'] for x in selected}),
        notebooks=len({x['provenance']['notebook_id'] for x in selected}),
        train_raw_sha256=sha256(output / 'data/train-raw.jsonl'),
        validation_raw_sha256=sha256(validation),
        training_started=False, encoded=False,
        eligibility='Existing E2B + complete multiturn + reference agreement criteria unchanged; '
                    'disagreements remain unapproved, not silently treated as real success.')
    (output / 'selection.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(manifest), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('root')
    parser.add_argument('output')
    parser.add_argument('--target', type=int, default=10000)
    args = parser.parse_args()
    main(args.root, args.output, args.target)
