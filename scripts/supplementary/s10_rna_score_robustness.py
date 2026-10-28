"""Supplementary Figure S10: the RNA remodeling score does not depend on how it is built.

(A-D) RNA remodeling score in 54 rats, four constructions; (E) within-age correlation of the continuous-weight score
with markers, each marker left out of the score; (F) agreement between the four scores.
Writes outputs/supp_s10/supp_s10_rna_score_robustness.{pdf,png,svg}.
"""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts' / 'shared')]
import paths
from pt_rna_plotting import *
F = paths.out('supp_s10')
plt.rcParams.update({'font.size':8,'axes.labelsize':8,'axes.titlesize':9,'axes.titleweight':'normal','xtick.labelsize':7,'ytick.labelsize':7,'legend.fontsize':7,'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.6,'pdf.fonttype':42,'svg.fonttype':'none'})
def save(fig,name):
    for ext in ['pdf','png','svg']:fig.savefig(F/f'{name}.{ext}',bbox_inches='tight',dpi=230,facecolor='white')
    plt.close(fig);print('Saved',name,flush=True)
def panel(ax,l,title):
    ax.set_title(title,loc='left',pad=10,fontsize=9)
    ax.text(-.15,1.075,l,transform=ax.transAxes,weight='bold',fontsize=11,va='bottom')

def supp_s10():
    fig,axs=plt.subplots(3,2,figsize=(8.4,8.1));fig.subplots_adjust(wspace=.85,hspace=.86)
    names={'current_unweighted_300':'Broad 300-gene selection','optimized_unweighted_10':'Compact 10-gene selection','continuous_loading_weighted':'Continuous loading weights','stability_selected_unweighted':'Stability-selected genes'}
    for ax,(m,title),letter in zip(axs.flat,names.items(),'abcd'):
        z=rs[rs.method.eq(m)];v=z.score/z.loc[z.age.eq(6),'score'].std();groupdots(ax,v,z.age);ax.set_ylabel('Score / 6-month SD');panel(ax,letter,title)
    ax=axs[2,0];z=read(paths.RNA_SCORE_MARKERS);z=z[z.method.eq('continuous_loading_weighted')];ax.barh(range(len(z)),z.target_excluded_within_age_r,color=np.where(z.target_excluded_within_age_r>0,ORANGE,GREEN));ax.set_yticks(range(len(z)),['KIM-1','LCN2','CD45','TIMP1','OAT1','Adaptive PT','Adaptive TAL']);ax.invert_yaxis();ax.set(xlabel='Target-excluded within-age r',xlim=(-1,1));panel(ax,'e','RNA state retains biological meaning')
    ax=axs[2,1];z=read(paths.RNA_SCORE_AGREEMENT,index_col=0);im=ax.imshow(z,cmap='Blues',vmin=.98,vmax=1)
    for i in range(4):
        for j in range(4):ax.text(j,i,f'{z.iloc[i,j]:.3f}',ha='center',va='center',fontsize=8,color='white' if z.iloc[i,j]>.992 else '#222')
    ax.set_xticks(range(4),['300','10','Weighted','Stable'],rotation=35,ha='right');ax.set_yticks(range(4),['300','10','Weighted','Stable']);panel(ax,'f','Transfer methods agree');save(fig,'supp_s10_rna_score_robustness')

if __name__=='__main__':
    supp_s10()
