"""Export a compact exploratory error/time figure from recorded measurements."""
from pathlib import Path
import os,csv
P=Path(__file__).resolve().parent;R=P/'results';os.environ['MPLCONFIGDIR']=str(R/'mpl-cache')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
COL={'CrossGB':'#2563eb','XGB':'#d97706','LGB_constant':'#64748b','LGB_linear':'#059669'}
MARK={'CrossGB':'o','XGB':'s','LGB_constant':'D','LGB_linear':'^'}
rows=list(csv.DictReader((R/'real_frontier_summary.csv').open()));syn=list(csv.DictReader((R/'synthetic_frozen_check_summary.csv').open()))
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':180})
fig,axes=plt.subplots(1,3,figsize=(14,4.2))
for ax,cell,title,binary in [(axes[0],'kin8nm.csv::numeric','kin8nm: development subset',False),(axes[1],'ilpd.csv::drop','ILPD: development subset',True)]:
 for m in COL:
  points=sorted([(float(r['fit_wall_sec'])*1000,float(r['primary'])) for r in rows if r['cell']==cell and r['method']==m])
  ax.scatter([x for x,y in points],[y for x,y in points],s=15,c=COL[m],alpha=.22,marker=MARK[m])
  frontier=[];best=-float('inf') if binary else float('inf')
  for x,y in points:
   if (y>best if binary else y<best):frontier.append((x,y));best=y
  ax.plot([x for x,y in frontier],[y for x,y in frontier],marker=MARK[m],markersize=4,lw=1.7,color=COL[m],label=m)
 ax.set_xscale('log');ax.set_xlabel('Fit wall time (ms, log scale)');ax.set_ylabel('AUC (higher is better)' if binary else 'RMSE (lower is better)');ax.set_title(title);ax.grid(alpha=.18)
ax=axes[2];case='sparse_switch__n600__d32__r4'
for m in COL:
 r=next(r for r in syn if r['case']==case and r['method']==m)
 ax.errorbar(float(r['median_fit_wall_sec'])*1000,float(r['mean_rmse']),yerr=float(r['sd_rmse']),fmt=MARK[m],markersize=7,capsize=3,color=COL[m],label=m)
ax.set_xscale('log');ax.set_xlabel('Fit wall time (ms, log scale)');ax.set_ylabel('RMSE (lower is better)');ax.set_title('Sparse role switch: 3 fresh data seeds');ax.grid(alpha=.18)
handles,labels=axes[0].get_legend_handles_labels();fig.legend(handles,labels,loc='lower center',ncol=4,frameon=False,bbox_to_anchor=(.5,-.015))
fig.suptitle('Local preexperiments: observed quality / training-time trade-offs',fontsize=13,y=1.01)
fig.tight_layout(rect=(0,.06,1,.99));fig.savefig(R/'quality_time.png',bbox_inches='tight');fig.savefig(R/'quality_time.pdf',bbox_inches='tight')
print(R/'quality_time.png')
