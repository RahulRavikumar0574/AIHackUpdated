# Strategy: pick the approach with highest F1, then precision, recall, accuracy,
# then ROC_AUC and AP as additional tie-breakers.

import os
import json
from glob import glob

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
OUT_DIR = os.path.join(BASE_DIR, 'experiments', 'outputs')
ARTIFACTS = os.path.join(BASE_DIR, 'artifacts.json')


def read_results():
    results = []
    for path in glob(os.path.join(OUT_DIR, '*_result.json')):
        try:
            with open(path, 'r', encoding='utf-8') as f:
                obj = json.load(f)
                approach = obj.get('approach', os.path.basename(path))
                thr = float(obj.get('threshold', 0.5))
                metrics = obj.get('metrics', {})
                roc = float(metrics.get('ROC_AUC', 0.0))
                ap = float(metrics.get('AP', 0.0))
                precision = float(metrics.get('precision', 0.0))
                recall = float(metrics.get('recall', 0.0))
                f1 = float(metrics.get('f1', 0.0))
                accuracy = float(metrics.get('accuracy', 0.0))
                results.append({
                    'approach': approach,
                    'threshold': thr,
                    'ROC_AUC': roc,
                    'AP': ap,
                    'precision': precision,
                    'recall': recall,
                    'f1': f1,
                    'accuracy': accuracy,
                    'path': path
                })
        except Exception as e:
            print('Failed to read', path, e)
    return results


def choose_best(results):
    if not results:
        raise SystemExit('No results found in experiments/outputs/*.json. Run the experiment scripts first.')
    results = sorted(
        results,
        key=lambda r: (
            r.get('f1', 0.0),
            r.get('precision', 0.0),
            r.get('recall', 0.0),
            r.get('accuracy', 0.0),
            r.get('ROC_AUC', 0.0),
            r.get('AP', 0.0)
        ),
        reverse=True
    )
    best = results[0]
    print('Candidate thresholds:')
    for r in results:
        print(
            f" - {r['approach']}: thr={r['threshold']:.4f}, "
            f"F1={r.get('f1',0.0):.4f}, P={r.get('precision',0.0):.4f}, R={r.get('recall',0.0):.4f}, Acc={r.get('accuracy',0.0):.4f}, "
            f"ROC_AUC={r.get('ROC_AUC',0.0):.4f}, AP={r.get('AP',0.0):.4f}"
        )
    print('\nSelected best:', best['approach'])
    return best


def update_artifacts(best):
    # Preserve other fields if file exists
    data = {
        'best_model_name': 'ensemble',
        'decision_threshold': best['threshold'],
        'metrics': {
            'ROC_AUC': best.get('ROC_AUC', 0.0),
            'AP': best.get('AP', 0.0),
            'precision': best.get('precision', 0.0),
            'recall': best.get('recall', 0.0),
            'f1': best.get('f1', 0.0),
            'accuracy': best.get('accuracy', 0.0)
        }
    }
    try:
        if os.path.exists(ARTIFACTS):
            with open(ARTIFACTS, 'r', encoding='utf-8') as f:
                old = json.load(f)
            old['decision_threshold'] = best['threshold']
            old_metrics = old.get('metrics', {})
            old_metrics['ROC_AUC'] = best.get('ROC_AUC', 0.0)
            old_metrics['AP'] = best.get('AP', 0.0)
            old_metrics['precision'] = best.get('precision', 0.0)
            old_metrics['recall'] = best.get('recall', 0.0)
            old_metrics['f1'] = best.get('f1', 0.0)
            old_metrics['accuracy'] = best.get('accuracy', 0.0)
            old['metrics'] = old_metrics
            data = old
    except Exception:
        pass

    with open(ARTIFACTS, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)
    print('Updated artifacts.json with decision_threshold =', best['threshold'])


if __name__ == '__main__':
    res = read_results()
    best = choose_best(res)
    update_artifacts(best)
