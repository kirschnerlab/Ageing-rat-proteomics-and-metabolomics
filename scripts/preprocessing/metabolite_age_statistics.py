"""Metabolite age statistics (used by Figures 4 and 5 and Supplementary Figure S4): fixed age estimands, all
features and all supplied animals. No cross-modal feature selection or second normalization.

Reads the log2 metabolite table by instrument sample and the sample key; writes the 32-rat metabolite matrix, the
animal table and age statistics for four cohorts to outputs/metabolite_age_statistics/. The cohort without P64
(tables/age_statistics_without_P64_31.csv) is the one used in the figures (data/metabolomics/metabolite_age_statistics.csv).
"""
from pathlib import Path
import hashlib, itertools, json, platform, sys
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import paths
import numpy as np
import pandas as pd
import scipy
from scipy import stats
from statsmodels.stats.multitest import multipletests
from sklearn.decomposition import PCA
import sklearn, statsmodels
sys.stdout.reconfigure(encoding='utf-8')

OUT=paths.out('metabolite_age_statistics')
for folder in ['data','tables','figures','qa','reports','metadata']:
    (OUT/folder).mkdir(parents=True,exist_ok=True)
features=pd.read_csv(paths.METABOLITE_FEATURES)
ledger=pd.read_csv(paths.METABOLOMICS_SAMPLE_KEY)
original=pd.read_csv(paths.METABOLITES_BY_SAMPLE,index_col=0)
Y=original.loc[ledger.metabolomics_sample_id,features.feature_id].copy()
Y.index=ledger.sample_id; Y.index.name='sample_id'
A=ledger.set_index('sample_id').loc[Y.index]
assert Y.shape==(32,269) and Y.index.is_unique and np.isfinite(Y.values).all()
assert A.age_months.value_counts().to_dict()=={25:8,21:8,18:8,9:8}
matched=pd.read_csv(paths.METABOLITES_PROTEOMICS_RATS,index_col=0)
assert np.allclose(Y.loc[matched.index],matched,rtol=0,atol=1e-12)
Y.to_csv(OUT/'data/metabolite_log2_all_rats.csv')
A.to_csv(OUT/'data/animals.csv')
features.to_csv(OUT/'metadata/source_feature_annotations.csv',index=False)
AGES=[9,18,21,25]

def bh(p):
    return multipletests(np.asarray(p),method='fdr_bh')[1]

def fit(data,animals):
    """HC3 covariance of four age means; saturated categorical-age OLS."""
    age=animals.loc[data.index,'age_months'].values
    vals=data.values
    groups=[vals[age==a] for a in AGES]
    n=np.array([len(g) for g in groups]); df=len(vals)-4
    means=np.array([g.mean(0) for g in groups])
    var=np.array([g.var(0,ddof=1) for g in groups])
    # Each group-mean HC3 variance equals sample variance/(n_g-1).
    vc=var/(n[:,None]-1)
    differences=means[1:]-means[0]
    se=np.sqrt(vc[1:]+vc[0])
    tt=differences/se
    pp=2*stats.t.sf(np.abs(tt),df)
    cov=np.zeros((vals.shape[1],3,3))
    for j in range(3):
        for k in range(3): cov[:,j,k]=vc[0]+(vc[j+1] if j==k else 0)
    omnibus=np.einsum('ij,ijk,ik->i',differences.T,np.linalg.inv(cov),differences.T)/3
    omnibus_p=stats.f.sf(omnibus,3,df)
    endpoint=differences[2]; endpoint_se=se[2]
    s0=var[0]/n[0]; s1=var[3]/n[3]
    welch_se=np.sqrt(s0+s1)
    welch_df=(s0+s1)**2/(s0**2/(n[0]-1)+s1**2/(n[3]-1))
    welch_p=2*stats.t.sf(abs(endpoint/welch_se),welch_df)
    kw=np.array([stats.kruskal(*[g[:,j] for g in groups]).pvalue for j in range(vals.shape[1])])
    out=features.copy()
    out['n_animals']=len(data)
    for i,a in enumerate(AGES):
        out[f'n_{a}']=n[i];out[f'mean_log2_{a}']=means[i]
        out[f'sd_log2_{a}']=np.sqrt(var[i])
        out[f'median_log2_{a}']=np.median(groups[i],axis=0)
    for i,a in enumerate(AGES[1:]):
        out[f'log2FC_{a}_vs_9']=differences[i]
        out[f'FC_{a}_vs_9']=2**differences[i]
        out[f'p_HC3_{a}_vs_9']=pp[i]
        out[f'q_HC3_joint807_{a}_vs_9']=bh(pp.ravel()).reshape(pp.shape)[i]
    out['se_HC3_25_vs_9']=endpoint_se
    out['HC3_df']=df
    out['CI95_HC3_low_log2FC']=endpoint-stats.t.ppf(.975,df)*endpoint_se
    out['CI95_HC3_high_log2FC']=endpoint+stats.t.ppf(.975,df)*endpoint_se
    out['q_HC3_25_vs_9']=bh(pp[2])
    out['p_Welch_25_vs_9']=welch_p
    out['q_Welch_25_vs_9']=bh(welch_p)
    out['Welch_df']=welch_df
    out['CI95_Welch_low_log2FC']=endpoint-stats.t.ppf(.975,welch_df)*welch_se
    out['CI95_Welch_high_log2FC']=endpoint+stats.t.ppf(.975,welch_df)*welch_se
    out['F_HC3_omnibus']=omnibus
    out['p_HC3_omnibus']=omnibus_p
    out['q_HC3_omnibus']=bh(omnibus_p)
    out['p_Kruskal_Wallis']=kw
    out['q_Kruskal_Wallis']=bh(kw)
    out['median_difference_log2_25_vs_9']=np.median(groups[3],axis=0)-np.median(groups[0],axis=0)
    out['largest_age_mean_span_log2']=means.max(0)-means.min(0)
    out['peak_age_by_mean']=[AGES[i] for i in means.argmax(0)]
    out['trough_age_by_mean']=[AGES[i] for i in means.argmin(0)]
    out['monotonic_four_means']=((np.diff(means,axis=0)>=0).all(0)|(np.diff(means,axis=0)<=0).all(0))
    out['endpoint_q_both_HC3_Welch_max']=np.maximum(out.q_HC3_25_vs_9,out.q_Welch_25_vs_9)
    return out

cohorts={
    'all32':Y.index,
    'matched31':Y.index[Y.index!='N146'],
    'without_P64_31':Y.index[Y.index!='P64'],
    'matched_without_P64_30':Y.index[(Y.index!='N146')&(Y.index!='P64')],
}
outputs={}
for name,ids in cohorts.items():
    outputs[name]=fit(Y.loc[ids],A)
    outputs[name].to_csv(OUT/f'tables/age_statistics_{name}.csv',index=False)
    print('fitted',name,flush=True)

# Exhaustive endpoint permutation, all C(16,8)=12870 assignments, as sensitivity.
# This tests label exchangeability, not a mean-only heteroskedastic null.
end=Y.loc[A.age_months.isin([9,25])].values
end_age=A.loc[A.age_months.isin([9,25]),'age_months'].values
assignments=np.array(list(itertools.combinations(range(16),8)))
perm_exceed=np.zeros(269,dtype=int)
observed=np.abs(outputs['all32'].log2FC_25_vs_9.values)
for start in range(0,len(assignments),500):
    selected=end[assignments[start:start+500]].mean(axis=1)
    delta=2*selected-end.mean(axis=0)*2
    perm_exceed+=(abs(delta)>=observed-1e-12).sum(0)
perm_p=perm_exceed/len(assignments)
outputs['all32']['p_exact_endpoint_mean_permutation']=perm_p
outputs['all32']['q_exact_endpoint_mean_permutation']=bh(perm_p)
outputs['all32'].to_csv(OUT/'tables/age_statistics_all32.csv',index=False)

# Deterministic leave-one-animal-out effect ranges, without significance selection.
loo=[]
for sid in Y.index:
    d=Y.drop(index=sid);a=A.loc[d.index,'age_months']
    loo.append((d.loc[a==25].mean()-d.loc[a==9].mean()).values)
loo=np.asarray(loo)
stability=outputs['all32'][['feature_id','reported_name','log2FC_25_vs_9','q_HC3_25_vs_9','q_Welch_25_vs_9']].copy()
stability['LOO_log2FC_min']=loo.min(0);stability['LOO_log2FC_max']=loo.max(0)
stability['LOO_same_direction_every_rat']=(np.sign(loo)==np.sign(outputs['all32'].log2FC_25_vs_9.values)).all(0)
stability['LOO_at_least_1p5fold_every_rat']=(abs(loo)>=np.log2(1.5)).all(0)
for name,result in outputs.items():
    for field in ['log2FC_25_vs_9','q_HC3_25_vs_9','q_Welch_25_vs_9','q_HC3_omnibus']:
        stability[name+'__'+field]=result[field].values
stability.to_csv(OUT/'tables/cohort_and_animal_sensitivity.csv',index=False)
assert np.allclose(outputs['all32'].log2FC_25_vs_9,outputs['without_P64_31'].log2FC_25_vs_9)
assert np.allclose(outputs['all32'].q_Welch_25_vs_9,outputs['without_P64_31'].q_Welch_25_vs_9)

# Counts distinguish endpoint/omnibus and statistical support/observed magnitude.
count_rows=[]
for name,result in outputs.items():
    for method,field in [('HC3','q_HC3_25_vs_9'),('Welch','q_Welch_25_vs_9'),('both_HC3_Welch','endpoint_q_both_HC3_Welch_max')]:
        for fc in [1,1.25,1.5,2]:
            chosen=(result[field]<.05)&(abs(result.log2FC_25_vs_9)>=np.log2(fc))
            count_rows.append({'cohort':name,'method':method,'q_cutoff':.05,'observed_min_fold_magnitude':fc,
                'n_features':int(chosen.sum()),'up':int((chosen&(result.log2FC_25_vs_9>0)).sum()),
                'down':int((chosen&(result.log2FC_25_vs_9<0)).sum()),
                'MS2_A':int((chosen&(result.collaborator_tag=='A')).sum()),
                'MS1_B':int((chosen&(result.collaborator_tag=='B')).sum())})
counts=pd.DataFrame(count_rows);counts.to_csv(OUT/'tables/endpoint_counts.csv',index=False)
all_counts=[]
for name,result in outputs.items():
    sig=result.q_HC3_omnibus<.05
    all_counts.append({'cohort':name,'n_animals':int(result.n_animals.iloc[0]),'n_features':len(result),
        'omnibus_HC3_q05':int(sig.sum()),'omnibus_KW_q05':int((result.q_Kruskal_Wallis<.05).sum()),
        'omnibus_HC3_and_KW_q05':int((sig&(result.q_Kruskal_Wallis<.05)).sum()),
        'omnibus_HC3_q05_and_1p5fold_age_mean_span':int((sig&(result.largest_age_mean_span_log2>=np.log2(1.5))).sum()),
        'omnibus_HC3_q05_monotonic_four_means':int((sig&result.monotonic_four_means).sum()),
        'any_young_contrast_joint807_q05':int((result[[f'q_HC3_joint807_{a}_vs_9' for a in [18,21,25]]]<.05).any(axis=1).sum())})
pd.DataFrame(all_counts).to_csv(OUT/'tables/overall_counts.csv',index=False)

# Descriptive PCA only: centered/unit-SD scaled log2 features, all32 animals.
Z=(Y-Y.mean())/Y.std(ddof=1)
pca=PCA(n_components=5).fit(Z)
scores=pd.DataFrame(pca.transform(Z),index=Y.index,columns=[f'PC{i+1}' for i in range(5)])
scores=scores.join(A[['age_months','metabolomics_sample_id']])
scores.to_csv(OUT/'tables/pca_scores.csv')
pd.DataFrame(pca.components_.T,index=Y.columns,columns=[f'PC{i+1}' for i in range(5)]).to_csv(OUT/'tables/pca_loadings.csv')

config={'n_animals':32,'age_counts':A.age_months.value_counts().sort_index().to_dict(),
    'feature_count':269,'primary_cohort':'all32; N146 included because standalone metabolomics is available',
    'input_policy':'Exported median-normalized peak areas, log2 once; no additional weight/volume or sample normalization',
    'primary_endpoint':'25-vs-9 month mean log2 difference; categorical-age HC3 t(n-4); BH269',
    'coequal_endpoint_sensitivity':'Welch unequal-variance t on 8 old and 8 young animals; BH269',
    'omnibus':'HC3 Wald/F(3,n-4) four-age test, BH269; Kruskal-Wallis BH269 sensitivity',
    'secondary_contrasts':'18/21/25 versus9; BH across all807 contrast-feature tests',
    'magnitude_thresholds':[1.25,1.5,2],
    'meaningful_wording':'FDR-supported differences with an estimated >=1.5fold magnitude; not an FDR guarantee for a minimum-effect null',
    'permutation_sensitivity':{'statistic':'absolute endpoint difference of log2 means','n_assignments':12870,'label_exchangeability_required':True,'resolution':2/12870},
    'PCA_explained_variance_ratio':pca.explained_variance_ratio_.tolist(),
    'technical_limits':'No acquisition sequence, pooled-QC, blanks or normalization factors supplied. Source sample labels are age-blocked; actual run-order confounding unknown.',
    'identity_limits':'269 feature rows, not independent standard-confirmed compounds; split peaks retained; A/MS2 is not authentic-standard confirmation',
    'versions':{'python':sys.version,'numpy':np.__version__,'pandas':pd.__version__,'scipy':scipy.__version__,'statsmodels':statsmodels.__version__,'sklearn':sklearn.__version__},
    'input_sha256':{paths.rel(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [paths.METABOLITE_FEATURES,paths.METABOLOMICS_SAMPLE_KEY,paths.METABOLITES_BY_SAMPLE]}}
(OUT/'qa/analysis_contract.json').write_text(json.dumps(config,indent=2),encoding='utf-8')
print(pd.DataFrame(all_counts).to_string(index=False))
print(counts[counts.cohort=='all32'].to_string(index=False))
top=outputs['all32'].sort_values('q_HC3_25_vs_9').head(35)
print(top[['feature_id','reported_name','FC_25_vs_9','q_HC3_25_vs_9','q_Welch_25_vs_9']].to_string(index=False))
