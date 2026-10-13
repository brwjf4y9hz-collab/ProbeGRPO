#!/usr/bin/env python3
"""Render existing result figures locally. No model calls, training, or grading."""
from pathlib import Path
import csv
import json
import os
import statistics
import argparse
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
os.environ.setdefault('MPLCONFIGDIR',str(Path(os.environ.get('TMPDIR','/tmp'))/'swe-results-mpl'))
import matplotlib
matplotlib.use('Agg')
from matplotlib import font_manager
import matplotlib.pyplot as plt
import numpy as np
COLORS=['#6B7280','#009E73','#E69F00','#0072B2']
TRUE,FALSE,UNKNOWN='#0072B2','#D55E00','#CBD5E1'
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output',type=Path,required=True,help='New output directory for PNG and SVG figures')
args=parser.parse_args()
FIG=args.output.resolve()
FIG.mkdir(parents=True,exist_ok=False)
def read(path):
    return json.loads(path.read_text())

def configure():
    installed={f.name for f in font_manager.fontManager.ttflist}
    selected=next((f for f in ['Arial Unicode MS','PingFang SC','Heiti SC','Songti SC','Noto Sans CJK SC'] if f in installed),None)
    if not selected:
        raise RuntimeError('A CJK font is required; do not publish missing-glyph figures')
    plt.rcParams.update({'font.family':selected,'axes.unicode_minus':False,'font.size':10,
        'axes.titlesize':12,'axes.titleweight':'bold','axes.labelsize':10,'xtick.labelsize':9,'ytick.labelsize':9,
        'svg.fonttype':'path','figure.facecolor':'white','axes.facecolor':'white','axes.spines.top':False,
        'axes.spines.right':False,'axes.axisbelow':True,'savefig.facecolor':'white'})
    return selected

def panel(ax,title,xlabel=None,ylabel=None):
    ax.set_title(title,loc='left',pad=12)
    if xlabel: ax.set_xlabel(xlabel)
    if ylabel: ax.set_ylabel(ylabel)
    ax.grid(axis='y',color='#E5E7EB',linewidth=.7)

def finish(fig,name,title,subtitle,notes):
    single_row=len(fig.axes)==2
    fig.suptitle(title,x=.07,y=.98,ha='left',fontsize=19,fontweight='bold')
    fig.text(.07,.90 if single_row else .928,subtitle,fontsize=10,color='#475569')
    fig.subplots_adjust(left=.085,right=.965,top=.78 if single_row else .85,
        bottom=.25 if single_row else .20,hspace=.60,wspace=.32)
    fig.text(.07,.035,notes,fontsize=9,color='#475569',va='bottom',linespacing=1.6)
    for ext in ['png','svg']:
        path=FIG/f'{name}.{ext}'
        fig.savefig(path,dpi=220)
        if ext=='svg':
            path.write_text('\n'.join(line.rstrip() for line in path.read_text().splitlines())+'\n')
    plt.close(fig)

def official_stack(ax,rows,labels,title,denominator):
    left=np.zeros(len(rows))
    for key,color,label,hatch in [('true',TRUE,'官方通过',''),('false',FALSE,'官方未通过',''),('unknown',UNKNOWN,'Unknown / 未评分','///')]:
        values=[r[key] for r in rows]
        ax.barh(np.arange(len(rows)),values,left=left,color=color,height=.55,label=label,hatch=hatch,edgecolor='white')
        for y,(v,start) in enumerate(zip(values,left)):
            if v: ax.text(start+v/2,y,str(v),ha='center',va='center',color='#1F2937' if key=='unknown' else 'white',fontweight='bold')
        left+=values
    ax.set_yticks(np.arange(len(rows)),labels)
    ax.invert_yaxis(); ax.set_xlim(0,denominator)
    ax.set_xticks(range(0,denominator+1,2 if denominator>=12 else 1))
    panel(ax,title,xlabel=f'条件数（每行固定分母 {denominator}）')
    ax.grid(axis='y',visible=False); ax.grid(axis='x',color='#E5E7EB',linewidth=.7)
    ax.legend(loc='upper left',bbox_to_anchor=(0,-.20),ncol=3,fontsize=8,frameon=False)

def plot_probe(probe):
    fig,axes=plt.subplots(2,2,figsize=(12.8,8.8))
    labels=['GRPO','Random-B2','Surprisal-B2','LinearUCB-B2']; x=np.arange(4)
    ax=axes[0,0]
    means=np.array([r['success_mean']*100 for r in probe]); sd=[r['success_sample_sd']*100 for r in probe]
    ax.bar(x,means,color=COLORS,width=.58,alpha=.83)
    ax.errorbar(x,means,yerr=sd,fmt='none',ecolor='#263238',capsize=5,linewidth=1.2)
    for i,r in enumerate(probe):
        ax.scatter(i+np.array([-.14,0,.14]),np.array(r['success_by_seed'])*100,s=29,color='#111827',zorder=4)
        ax.text(i,means[i]+sd[i]+1.3,f'{means[i]:.1f} ± {sd[i]:.1f}%',ha='center',fontsize=9)
    ax.set_xticks(x,labels,rotation=12); ax.set_ylim(0,41)
    panel(ax,'A  留出成功率：均值、标准差与三个原始点',ylabel='成功率（%）')
    ax=axes[0,1]; grpo=probe[0]; ucb=probe[3]
    for i,(a,b) in enumerate(zip(grpo['success_by_seed'],ucb['success_by_seed'])):
        ax.plot([i,i],[a*100,b*100],color='#94A3B8',linewidth=2)
        ax.text(i+.16,(a+b)*50,f'+{(b-a)*100:.1f} pp',fontsize=9,va='center')
    ax.scatter(range(3),np.array(grpo['success_by_seed'])*100,s=65,color=COLORS[0],label='GRPO',zorder=3)
    ax.scatter(range(3),np.array(ucb['success_by_seed'])*100,s=65,color=COLORS[3],label='LinearUCB-B2',zorder=3)
    ax.set_xticks(range(3),[f'seed {s}' for s in ucb['seeds']]); ax.set_xlim(-.45,2.7); ax.set_ylim(15,37)
    ax.legend(frameon=False,loc='lower right',fontsize=9)
    panel(ax,'B  同种子配对：三次历史运行均有提高',ylabel='成功率（%）；pp = 百分点')
    ax=axes[1,0]; ratios=[1+r['rollout_overhead_mean'] for r in probe]
    ax.barh(x,ratios,color=COLORS,height=.55); ax.invert_yaxis(); ax.set_yticks(x,labels)
    ax.axvline(1.5,color='#334155',linestyle='--',linewidth=1)
    for i,(r,v) in enumerate(zip(probe,ratios)):
        ax.text(v+.025,i,f'{v:.3f}×  (+{r["rollout_overhead_mean"]*100:.1f}%)',va='center',fontsize=9)
    ax.set_xlim(0,2.1)
    panel(ax,'C  探测成本（虚线：原 1.5× 目标）',xlabel='总 rollout token / 各自主 rollout token')
    ax.grid(axis='y',visible=False); ax.grid(axis='x',color='#E5E7EB',linewidth=.7)
    ax=axes[1,1]; values=[r['high_impact_per_1k_mean'] for r in probe[1:]]
    ax.bar(range(3),values,color=COLORS[1:],width=.55)
    for i,v in enumerate(values): ax.text(i,v+.3,f'{v:.2f}',ha='center')
    ax.set_xticks(range(3),labels[1:],rotation=12); ax.set_ylim(0,15)
    panel(ax,'D  Anchor 诊断：LinearUCB 未领先此项',ylabel='高影响 anchor / 每千额外 token')
    finish(fig,'01_probegrpo','ProbeGRPO｜结果与成本一起展示',
        'Qwen3.5-2B · 512 train / 128 共同留出关卡 · 四方法 × 三种子 × 50 次训练更新',
        '误差线为跨种子样本标准差，不是置信区间；三种子不证明显著性。seed17 GRPO 有已记录的 padding 配置差异。\n'
        '配对提高约 7.0 pp；LinearUCB 额外 token 为 51.6%，未满足原 1.5× 目标。非等总 token 对照，未证明最优调度器或普遍因果收益。')

probe=read(ROOT/'experiments/results/public_sokoban_main_v1/aggregate.json')
rows=list(csv.DictReader((ROOT/'experiments/results/public_sokoban_main_v1/per_seed.csv').open()))
for result in probe:
    values=sorted([r for r in rows if r['method']==result['method']],key=lambda r:int(r['seed']))
    assert len(values)==3 and all(int(r['eval_count'])==128 for r in values)
    points=[int(r['success_count'])/128 for r in values]
    assert points==result['success_by_seed']
    assert abs(statistics.mean(points)-result['success_mean'])<1e-12
    assert abs(statistics.stdev(points)-result['success_sample_sd'])<1e-12
configure()
plot_probe(probe)
print('Source checks passed; wrote one PNG and one SVG; no model requests.')
