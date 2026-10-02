#!/usr/bin/env python3
"""从归档原始日志和 episode 记录在 CPU 上重新计算已发布结果。"""
import argparse
import hashlib
import json
import math
from pathlib import Path

from aggregate_sokoban_seeds import (
    CSV_FIELDS, FLOAT_FIELDS, INT_FIELDS, METHOD_ORDER, aggregate, markdown, read_csv, write_csv,
)
from summarize_sokoban_ablation import summarize


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--artifact-root', required=True, type=Path,
                        help='Directory containing the restored outputs/ folder')
    parser.add_argument('--output-dir', required=True, type=Path)
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[1]
    published = project / 'experiments/results/public_sokoban_main_v1'
    metadata = json.loads((published / 'metadata.json').read_text())
    reference = {(r['seed'], r['method']): r for r in read_csv(published / 'per_seed.csv')}
    rows, evidence, errors = [], [], []
    expected_boards = None
    for relative in metadata['raw_output_roots']:
        root = args.artifact_root / relative
        summaries = {r['method']: r for r in summarize(root)}
        if set(summaries) != set(METHOD_ORDER):
            raise ValueError(f'expected exactly four completed methods in {root}')
        for method in METHOD_ORDER:
            run = root / method
            manifest = json.loads((run / 'run_manifest.json').read_text())
            seed = int(manifest['seed'])
            if manifest['model_revision'] != metadata['model_revision']:
                raise ValueError(f'model revision mismatch: {run}')
            if manifest['verl_revision'] != metadata['verl_revision']:
                raise ValueError(f'verl revision mismatch: {run}')
            if manifest['data']['source_revision'] != metadata['dataset_revision']:
                raise ValueError(f'dataset revision mismatch: {run}')
            if manifest['total_training_steps'] != metadata['training_steps']:
                raise ValueError(f'training-step mismatch: {run}')
            episodes = [json.loads(p.read_text()) for p in
                        (run / 'rollouts/episodes/step-50').glob('*.json')]
            evaluated = [e for e in episodes if e.get('dataset_split') == 'test']
            if len(evaluated) != metadata['test_examples']:
                raise ValueError(f'incomplete final evaluation: {run}')
            boards = {e['task_id'] for e in evaluated}
            if len(boards) != len(evaluated):
                raise ValueError(f'duplicate evaluation boards: {run}')
            if expected_boards is None:
                expected_boards = boards
            elif boards != expected_boards:
                raise ValueError(f'evaluation board-set mismatch: {run}')
            scores = [float(e['final_reward']) for e in evaluated]
            if any(x not in (0.0, 1.0) for x in scores):
                raise ValueError(f'non-binary success score: {run}')
            summary = summaries[method]
            if summary.get('training/global_step') != metadata['training_steps']:
                raise ValueError(f'final logged training step mismatch: {run}')
            row = {field: summary.get(field) for field in CSV_FIELDS}
            row.update(seed=seed, method=method, success_count=int(sum(scores)),
                       eval_count=len(scores), final_val_success=sum(scores)/len(scores))
            expected = reference[(seed, method)]
            for field in INT_FIELDS:
                if int(row[field]) != int(expected[field]):
                    errors.append(f'{seed}/{method}: {field} differs')
            for field in FLOAT_FIELDS:
                a, b = row[field], expected[field]
                # 与保留完整精度的审计导出文件逐项比较。
                tolerance = 0.0 if field == 'final_val_success' else 1e-12
                if (a is None) != (b is None) or (a is not None and not
                    math.isclose(float(a), float(b), rel_tol=0, abs_tol=tolerance)):
                    errors.append(f'{seed}/{method}: {field} differs: {a} vs {b}')
            rows.append(row)
            evidence.append({'seed':seed, 'method':method, 'eval_count':len(scores),
                             'success_count':int(sum(scores)),
                             'training_source_revision':manifest['probegrpo_revision']})
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / 'per_seed.csv', rows)
    result = aggregate(rows)
    (args.output_dir / 'aggregate.json').write_text(json.dumps(result, indent=2)+'\n')
    (args.output_dir / 'summary.md').write_text(markdown(result))
    report = {'status':'passed' if not errors else 'failed', 'run_count':len(rows),
              'evaluation_episodes':sum(x['eval_count'] for x in evidence),
              'comparison':'exact counts and success; full-precision costs/timing within 1e-12',
              'runs':evidence, 'errors':errors,
              'eval_board_set_sha256':hashlib.sha256('\n'.join(sorted(expected_boards)).encode()).hexdigest(),
              'scope':'Re-aggregation of saved evidence; no GPU training rerun.'}
    (args.output_dir / 'verification.json').write_text(json.dumps(report, indent=2)+'\n')
    print(markdown(result))
    print(json.dumps({k:v for k,v in report.items() if k!='runs'},indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
