import json
import os
import sys
p='results/test_run_small.jsonl'
if not os.path.exists(p):
    print('file missing', p)
    sys.exit(1)
with open(p,'r',encoding='utf-8') as f:
    for line in f:
        o=json.loads(line)
        pred=o.get('prediction') or o.get('fused_summary') or ''
        corr=o.get('corrected') or ''
        def short(s,n=300):
            if not s:
                return ''
            return s[:n].replace('\n',' ') + ('...' if len(s)>n else '')
        out={
            'id': o.get('id'),
            'prediction_short': short(pred),
            'corrected_short': short(corr),
            'support_rate': o.get('support_rate'),
            'corrected_support_rate': o.get('corrected_support_rate'),
            'rouge1_fmeasure': (o.get('rouge') or {}).get('rouge1_fmeasure'),
            'rouge1_fmeasure_corrected': (o.get('rouge_corrected') or {}).get('rouge1_fmeasure'),
            'timing': o.get('timing')
        }
        print(json.dumps(out, ensure_ascii=False, indent=2))
