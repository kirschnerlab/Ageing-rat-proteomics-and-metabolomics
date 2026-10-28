"""Plot helpers and tables shared by Supplementary Figures S8 (proximal-tubule identity) and S10 (RNA score robustness)."""
from pathlib import Path
import sys, json, hashlib, shutil, zipfile, re, textwrap
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
from matplotlib.gridspec import GridSpec
from PIL import Image
import paths


def read(p, **kw):
    return pd.read_csv(p, **kw)


plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.titlesize':10,'axes.titleweight':'bold','axes.labelsize':9,'xtick.labelsize':8,'ytick.labelsize':8,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'svg.fonttype':'none','legend.fontsize':8,'axes.unicode_minus':True})
GREEN='#1b9e77';ORANGE='#d95f02';PURPLE='#7570b3';PINK='#e7298a';GRAY='#89929a';BLUE='#326eaa'
AGE={9:GREEN,18:PURPLE,21:ORANGE,25:PINK}; ages=[9,18,21,25]
def forest(ax,df,labels,est,lo,hi,color=BLUE):
    y=np.arange(len(df));v=df[est].to_numpy(float)
    ax.errorbar(v,y,xerr=np.vstack([v-df[lo],df[hi]-v]),fmt='o',color=color,ms=4,capsize=2,lw=1)
    ax.set_yticks(y,labels);ax.invert_yaxis();ax.axvline(0,c='.65',lw=.6);ax.set_xlabel('log2 change, 25 vs 9 months')
def groupdots(ax,values,age,title=None,legend=False):
    values=np.asarray(values,float);age=np.asarray(age); u=sorted(set(age));rng=np.random.default_rng(71)
    for j,a in enumerate(u):
        vals=values[age==a];vals=vals[np.isfinite(vals)]
        color=AGE.get(a,plt.cm.viridis(j/max(1,len(u)-1)))
        ax.scatter(np.full(len(vals),j)+rng.uniform(-.15,.15,len(vals)),vals,s=14,c=[color],alpha=.85,zorder=3,edgecolor='white',linewidth=.3)
        ax.plot([j-.22,j+.22],[np.mean(vals)]*2,c='black',lw=1.3)
    ax.set_xticks(range(len(u)),u);ax.set_xlabel('Age (months)')
    if title:ax.set_title(title,loc='left')
def linegroups(ax,df,cols,labels,colors):
    for col,label,color in zip(cols,labels,colors):
        v=df.groupby('age')[col].mean().reindex(ages);ax.plot(ages,v,'o-',c=color,label=label,ms=3,lw=1.2)
    ax.axhline(0,c='.7',lw=.6);ax.set_xticks(ages);ax.set_xlabel('Age (months)');ax.set_ylabel('Mean log2 vs 9 months')

mat=read(paths.PROTEIN_MATRIX);meta=read(paths.SAMPLE_METADATA);meta=meta[meta.role.eq('biological_sample')].set_index('label');cols=[x for x in mat if x in meta.index];meta=meta.loc[cols];age=meta.age_months.to_numpy(int)
pt=read(paths.PT_ADJUSTED_EFFECTS);seg=read(paths.PT_SEGMENT_EFFECTS);ps=read(paths.PT_SCORES_BY_RAT);nori=read(paths.NORI_PT_MEASURES)
rs=read(paths.RNA_SCORE_BY_RAT)
