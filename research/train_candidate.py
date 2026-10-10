"""Offline evaluation only. Never loads legacy pickle files or activates a model."""
import argparse
import csv
import hashlib
import json
import math
import io
from datetime import datetime, timedelta, timezone
from pathlib import Path

FEATURES = ['age_hours', 'idle_hours', 'priority_high', 'priority_urgent', 'transfers', 'reopens', 'response_delay_hours', 'sla_hours_remaining', 'missing_transfers', 'missing_reopens', 'missing_response_delay', 'missing_sla']
HORIZON = timedelta(hours=24)


def timestamp(value):
    result = datetime.fromisoformat(value.replace('Z', '+00:00'))
    if result.tzinfo is None:
        raise ValueError('Timestamps must include a timezone')
    return result.astimezone(timezone.utc)


def number(row, name, maximum):
    value = row.get(name, '').strip()
    if not value:
        return 0.0, 1.0
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise ValueError(f'{name} must be a finite nonnegative number')
    return min(result, maximum), 0.0


def example(row):
    observed = timestamp(row['observed_at'])
    created, updated = timestamp(row['created_at']), timestamp(row['updated_at'])
    ended = timestamp(row['observation_end'])
    if not created <= updated <= observed:
        raise ValueError('Features must be snapshots captured at or before observed_at')
    if ended > datetime.now(timezone.utc):
        raise ValueError('Outcome observation cannot extend into the future')
    if ended < observed + HORIZON:
        raise ValueError('Each outcome needs a full 24-hour observation window')
    if row['status'] not in ['new', 'open', 'pending', 'hold']:
        raise ValueError('Training rows must be unresolved at prediction time')
    if row['priority'] not in ['low', 'normal', 'high', 'urgent']:
        raise ValueError('Unsupported priority')
    event = timestamp(row['escalated_at']) if row.get('escalated_at', '').strip() else None
    if event and (event <= observed or event > ended):
        raise ValueError('Escalation must occur after prediction and within observation')
    transfers, mt = number(row, 'transfers', 10)
    reopens, mr = number(row, 'reopens', 10)
    delay, md = number(row, 'response_delay_hours', 168)
    sla = row.get('sla_due_at', '').strip()
    sla_hours = max(-72, min(72, (timestamp(sla) - observed).total_seconds() / 3600)) if sla else 72
    values = [min(720, (observed-created).total_seconds()/3600), min(168, (observed-updated).total_seconds()/3600), float(row['priority']=='high'), float(row['priority']=='urgent'), transfers, reopens, delay, sla_hours, mt, mr, md, float(not sla)]
    return observed, values, int(event is not None and event <= observed + HORIZON)


def load(path):
    rows = list(csv.DictReader(io.StringIO(Path(path).read_text(encoding='utf-8-sig'))))
    if len(rows) < 200:
        raise ValueError('At least 200 genuine labeled ticket snapshots required; this is a floor, not a sufficiency claim')
    ids = [r['ticket_id'] for r in rows]
    if any(not x.strip() for x in ids) or len(ids) != len(set(ids)):
        raise ValueError('Use one snapshot per unique ticket to prevent cross-split ticket leakage')
    examples = sorted([example(r) for r in rows], key=lambda r: r[0])
    cutoff = examples[int(len(examples)*0.7)][0]
    train = [r for r in examples if r[0]+HORIZON < cutoff]
    test = [r for r in examples if r[0] >= cutoff]
    if len(train) < 100 or len(test) < 30 or any(len({r[2] for r in split})<2 for split in [train, test]):
        raise ValueError('Temporal train/test split needs enough rows and both classes after the 24-hour embargo')
    return train, test, cutoff


def evaluate(path, destination):
    import numpy as np
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, precision_score, recall_score, confusion_matrix
    train, test, cutoff = load(path)
    x, y = np.array([r[1] for r in train]), np.array([r[2] for r in train])
    xt, yt = np.array([r[1] for r in test]), np.array([r[2] for r in test])
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, random_state=42, solver='liblinear'))
    model.fit(x, y)
    probabilities = model.predict_proba(xt)[:, 1]
    if not np.isfinite(probabilities).all():
        raise ValueError('Model produced nonfinite probabilities; candidate rejected')
    predicted = probabilities >= 0.5
    prevalence = float(y.mean())
    count = max(1, math.ceil(len(test)*0.1))
    report = {'status':'candidate_only_not_deployed', 'target':'genuine escalation in next 24 hours', 'cutoff':cutoff.isoformat(), 'embargo_hours':24, 'train_rows':len(train), 'test_rows':len(test), 'excluded_embargo_rows':len(list(csv.DictReader(io.StringIO(Path(path).read_text(encoding='utf-8-sig')))))-len(train)-len(test), 'dataset_sha256':hashlib.sha256(Path(path).read_bytes()).hexdigest(), 'metrics':{'roc_auc':float(roc_auc_score(yt, probabilities)), 'average_precision':float(average_precision_score(yt, probabilities)), 'test_prevalence':float(yt.mean()), 'precision_at_threshold_0_5':float(precision_score(yt,predicted,zero_division=0)), 'recall_at_threshold_0_5':float(recall_score(yt,predicted,zero_division=0)), 'precision_top_10_percent':float(yt[np.argsort(-probabilities)[:count]].mean()), 'brier_score':float(brier_score_loss(yt, probabilities)), 'baseline_brier_score':float(brier_score_loss(yt,np.full(len(yt),prevalence))), 'confusion_matrix':confusion_matrix(yt,predicted).tolist()}, 'calibration_bins':[], 'limitations':['Single temporal holdout; no external validation', 'Input provenance must establish authentic escalation events and snapshot features', 'Threshold 0.5 is fixed before evaluation, not optimized on holdout', 'No serving activation; operational UI continues to use risk index']}
    for low in np.arange(0,1,0.2):
        mask=(probabilities>=low)&(probabilities<(low+0.2) if low<0.8 else probabilities<=1)
        if mask.any(): report['calibration_bins'].append({'count':int(mask.sum()),'mean_probability':float(probabilities[mask].mean()),'observed_rate':float(yt[mask].mean())})
    directory=Path(destination);directory.mkdir(parents=True,exist_ok=True)
    (directory/'evaluation.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    scaler, classifier=model.steps[0][1],model.steps[1][1]
    candidate={'format':'signaldesk-logistic-candidate-v1','features':FEATURES,'mean':scaler.mean_.tolist(),'scale':scaler.scale_.tolist(),'coefficients':classifier.coef_[0].tolist(),'intercept':float(classifier.intercept_[0]),'approved_for_serving':False}
    (directory/'candidate.json').write_text(json.dumps(candidate,indent=2,allow_nan=False)+'\n')
    print(f'Evaluation written to {directory}; candidate not deployed.')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('csv');parser.add_argument('--output',default='research/output')
    args=parser.parse_args()
    try:evaluate(args.csv,args.output)
    except (ValueError,KeyError) as error:parser.exit(2,f'Invalid training data: {error}\n')
