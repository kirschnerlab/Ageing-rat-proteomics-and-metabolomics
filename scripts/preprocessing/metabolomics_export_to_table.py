"""Metabolomics instrument export (Compounds sheet, normalized peak areas) -> feature table and log2 matrix by
instrument sample. Read-only on the export. Writes to outputs/metabolomics_export/; data/metabolite_log2.csv there
equals data/metabolomics/metabolite_log2_by_sample.csv."""
from pathlib import Path
import hashlib
import json
import re
import sys
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import paths
OUT = paths.out('metabolomics_export')
SOURCE = paths.METABOLOMICS_EXPORT
for directory in ['data', 'tables', 'figures', 'reports', 'qa']:
    (OUT / directory).mkdir(parents=True, exist_ok=True)
d = pd.read_excel(SOURCE, sheet_name='Compounds')
assert d.shape == (269, 40)
sample_columns = [c for c in d if str(c).startswith('Area:') and str(c).endswith('_normalized')]
assert len(sample_columns) == 32
sample_ids = ['kidney' + re.search(r'kidney(\d+)\.raw', c).group(1) for c in sample_columns]
assert len(set(sample_ids)) == 32
feature_ids = [f'MET_ROW_{i:04d}' for i in range(2, len(d) + 2)]
x = d[sample_columns].apply(pd.to_numeric, errors='raise')
assert np.isfinite(x.values).all() and (x.values > 0).all()
features = d.iloc[:, :8].copy()
features.columns = ['collaborator_tag', 'reported_name', 'formula', 'mz', 'retention_time_min',
                    'reference_ion', 'max_area_source', 'peak_rating_kidney17_only']
features.insert(0, 'source_excel_row', np.arange(2, len(d) + 2))
features.insert(0, 'feature_id', feature_ids)
features['annotation_evidence'] = features.collaborator_tag.map({'A': 'MS2 level per collaborator', 'B': 'MS1 level per collaborator'})
features['identity_limit'] = 'No authentic-standard confirmation documented; preserve individual feature row'
features.to_csv(OUT / 'data/metabolite_features.csv', index=False)
matrix = pd.DataFrame(x.values.T, index=sample_ids, columns=feature_ids).sort_index()
matrix.index.name = 'metabolomics_sample_id'
matrix.to_csv(OUT / 'data/metabolomics_normalized_intensity.csv')
np.log2(matrix).to_csv(OUT / 'data/metabolite_log2.csv')
sample_key = pd.DataFrame({'metabolomics_sample_id': sample_ids, 'source_column': sample_columns,
                           'rat_id': '', 'age_months': '', 'kidney_side_or_tissue': '',
                           'mapping_evidence': ''}).sort_values('metabolomics_sample_id')
sample_key.to_csv(OUT / 'data/sample_columns.csv', index=False)
sources = [SOURCE]
manifest = [{'path': paths.rel(p), 'bytes': p.stat().st_size, 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()} for p in sources]
(OUT / 'qa/input_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
(OUT / 'qa/extraction.json').write_text(json.dumps({'python': sys.version.split()[0], 'pandas': pd.__version__,
 'numpy': np.__version__, 'n_features': len(features), 'n_samples': len(matrix),
 'missing': int(matrix.isna().sum().sum()), 'nonpositive': int((matrix <= 0).sum().sum()),
 'collaborator_tags': features.collaborator_tag.value_counts().to_dict(),
 'sample_mapping_status': 'Unresolved; no cross-modal correlations or elastic net allowed yet'}, indent=2), encoding='utf-8')
print('Extracted 269 feature rows and 32 anonymized sample columns; source inputs preserved.')
