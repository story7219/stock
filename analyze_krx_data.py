#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
KRX 과거데이터 자동 분석 스크립트
- data/krx_smart/ 폴더의 모든 JSON 파일을 순회하며
- 주식, 선물, 옵션, 지수, ETF, K200, KOSDAQ50 데이터의 행 개수, 타임스탬프, 실제 데이터 존재 여부를 표로 요약 출력
"""
import os
import json
import glob
import pandas as pd

FOLDER = 'data/krx_smart'
TYPES = ['stock', 'futures', 'options', 'index', 'etf', 'k200', 'kosdaq50']

summary = []
files = sorted(glob.glob(os.path.join(FOLDER, '*.json')))

for f in files:
    with open(f, encoding='utf-8') as fp:
        d = json.load(fp)
    row = {'file': os.path.basename(f)}
    for k in TYPES:
        v = d.get(k, {})
        if isinstance(v, dict):
            # ETF 특수 처리
            if k == 'etf':
                etf_data = v.get('etf_top6') or v.get('etf')
                if etf_data and 'OutBlock_1' in etf_data:
                    row['etf_rows'] = len(etf_data['OutBlock_1'])
                    row['etf_ts'] = v.get('timestamp')
                    row['etf_has_data'] = len(etf_data['OutBlock_1']) > 0
                else:
                    row['etf_rows'] = 0
                    row['etf_ts'] = v.get('timestamp')
                    row['etf_has_data'] = False
                continue
            # 일반 타입
            for subk in v:
                if isinstance(v[subk], dict) and 'OutBlock_1' in v[subk]:
                    row[f'{k}_{subk}_rows'] = len(v[subk]['OutBlock_1'])
                    row[f'{k}_{subk}_ts'] = v.get('timestamp')
                    row[f'{k}_{subk}_has_data'] = len(v[subk]['OutBlock_1']) > 0
                elif isinstance(v[subk], list):
                    row[f'{k}_{subk}_rows'] = len(v[subk])
                    row[f'{k}_{subk}_ts'] = v.get('timestamp')
                    row[f'{k}_{subk}_has_data'] = len(v[subk]) > 0
    summary.append(row)

pd.set_option('display.max_columns', None)
df = pd.DataFrame(summary)
print(df.fillna('').to_string(index=False)) 