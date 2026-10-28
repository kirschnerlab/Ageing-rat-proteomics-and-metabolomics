"""Supplementary Figure S8: proximal-tubule identity, metabolism and transport decline together.

(A) program means by age; (B) PT segment markers; (C) PT identity vs OXPHOS within age; (D) NoRI measurements of the
other kidney; (E) program changes before/after PT adjustment; (F) matched declining-set control.
Writes outputs/supp_s8/supp_s8_proximal_tubule_identity.{pdf,png,svg}.
"""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts' / 'shared')]
import json, shutil
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr
import paths
import pt_rna_plotting as E
F = paths.out('supp_s8')
A=np.load(paths.STATE_ARRAYS)
AN=pd.read_csv(paths.STATE_RAT_SCORES).rename(columns={'age_months':'age'})
age=AN.age.to_numpy()
G='#1b9e77'; P='#7570b3'; OJ='#d95f02'; K='#272727'; GR='#a9b2b7'; B='#326eaa'; AC=E.AGE
plt.rcParams.update({'font.size':8,'axes.labelsize':8,'axes.titlesize':9,'axes.titleweight':'normal','xtick.labelsize':7,'ytick.labelsize':7,'legend.fontsize':7,'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.6,'pdf.fonttype':42,'svg.fonttype':'none'})
def save(fig,name):
    for ext in ['pdf','png','svg']:fig.savefig(F/f'{name}.{ext}',bbox_inches='tight',dpi=230,facecolor='white')
    plt.close(fig);print('Saved',name,flush=True)
def panel(ax,l,title):
    ax.set_title(title,loc='left',pad=10,fontsize=9)
    ax.text(-.15,1.075,l,transform=ax.transAxes,weight='bold',fontsize=11,va='bottom')
def scatter(ax,x,y):
    for a,c in AC.items():ax.scatter(np.asarray(x)[age==a],np.asarray(y)[age==a],s=18,c=c,edgecolor='white',lw=.3)
    xx=np.array([min(x),max(x)]);ax.plot(xx,np.polyval(np.polyfit(x,y,1),xx),c=K,lw=.8)
    ax.axhline(0,c='.8',lw=.5);ax.axvline(0,c='.8',lw=.5)

def supp_s8():
    f=plt.figure(figsize=(8.7,9));gs=f.add_gridspec(3,2,hspace=.73,wspace=.72)
    ax=f.add_subplot(gs[0,0]);E.linegroups(ax,E.ps,['PT_identity','OXPHOS','FAO','SLC_transport','Glycolysis','Cytosolic_ribosome','Proteasome'],['PT identity','OXPHOS','FAO','Transport','Glycolysis','Ribosome','Proteasome'],[G,B,P,OJ,'#888','#aaa','#ccc']);ax.legend(frameon=False,ncol=2,fontsize=6.4,loc='lower left');panel(ax,'a','Differentiated tubular programs')
    ax=f.add_subplot(gs[0,1]);d=E.seg[E.seg.analysis.eq('segment_age_effect')];E.forest(ax,d,[f'{s} ({int(n)})' for s,n in zip(d.segment,d.n_proteins)],'estimate','ci_low','ci_high',G);panel(ax,'b','Mature segment markers')
    ax=f.add_subplot(gs[1,0]);ps=E.ps.set_index('label').loc[E.cols]
    X=A['X_age_only'];ys=E.ps[['PT_identity','OXPHOS']].to_numpy();rr=ys-X@np.linalg.lstsq(X,ys,rcond=None)[0]
    # E.ps ordering is explicitly reconciled with the sample matrix below.
    ys=ps[['PT_identity','OXPHOS']].to_numpy();rr=ys-X@np.linalg.lstsq(X,ys,rcond=None)[0]
    scatter(ax,rr[:,0],rr[:,1]);ax.set(xlabel='PT identity residual (log2)',ylabel='OXPHOS residual (log2)');ax.text(.04,.96,f'r = {pearsonr(rr[:,0],rr[:,1]).statistic:.3f}',transform=ax.transAxes,va='top');panel(ax,'c','Coordination among age-matched kidneys')
    ax=f.add_subplot(gs[1,1]);keys=['prox__nuclei_per_area','prox__lumen_fraction','prox__tb_size','prox__protein_level','comp__proximal_fraction_of_tubules'];d=E.nori.set_index('feature').loc[keys].copy()
    for k in ['estimate','ci_low','ci_high']:d[k]=100*d[k]/d.mean_9mo
    E.forest(ax,d,['Nuclear density','Lumen fraction','Profile area','Epithelial protein','PT-profile fraction'],'estimate','ci_low','ci_high',OJ);ax.set_xlabel('Change as % of observed 9-month mean');panel(ax,'d','Contralateral NoRI geometry')
    ax=f.add_subplot(gs[2,0]);d=E.pt.set_index('program').loc[['OXPHOS','FAO','SLC_transport']]
    for off,c,lab,e,l,h in [(-.13,GR,'Original','raw_age25_effect','raw_age25_ci_low','raw_age25_ci_high'),(.13,G,'With PT score','pt_adjusted_age25_effect','pt_adjusted_ci_low','pt_adjusted_ci_high')]:ax.errorbar(d[e],np.arange(3)+off,xerr=[d[e]-d[l],d[h]-d[e]],fmt='o',ms=4,c=c,lw=1,label=lab)
    ax.set_yticks(range(3),['OXPHOS','FAO','Transport']);ax.invert_yaxis();ax.axvline(0,c='.6',lw=.6);ax.set_xlabel('Change, 25 vs 9 months (log2)');ax.legend(frameon=False,fontsize=7,loc='lower left');panel(ax,'e','Original and PT-conditioned differences')
    ax=f.add_subplot(gs[2,1]);d=pd.read_csv(paths.PT_MATCHED_SETS);v=100*(1-d.attenuation);ax.hist(v,bins=40,color='#cad8dc');ax.axvline(25.78320205374092,c=G,lw=1.8,label='PT: 25.8%');ax.axvline(0,c=GR,lw=.7,ls=':');ax.set(xlabel='Age difference remaining (% of original)',ylabel='Accepted matched sets');ax.legend(frameon=False);ax.text(.03,.97,f'{len(d):,} matched declining sets\nMedian {np.median(v):.1f}%',transform=ax.transAxes,va='top',fontsize=7);panel(ax,'f','Specificity control for OXPHOS conditioning')
    save(f,'supp_s8_proximal_tubule_identity')

if __name__=='__main__':
    supp_s8()
