"""Render reproducible figures and a Chinese report from measured CSV results."""
import json
import os
from pathlib import Path
import numpy as np
import pandas as pd
from data import ROOT
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'.mplconfig'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

R=ROOT/'reports';F=R/'figures'
NAMES={'fusion_clean':'Clean fusion','fusion_aug':'Fusion + block training','robust_norec':'Mask + dynamic fusion','robust_rec':'Mask + fusion + reconstruction','robust_nogate':'Mask + reconstruction','text_only':'Text only'}


def table(headers,rows):
    def cell(x):return str(x).replace('|','\\|').replace('\n',' ')
    return '\n'.join(['| '+' | '.join(map(cell,headers))+' |','| '+' | '.join(['---']*len(headers))+' |']+['| '+' | '.join(map(cell,row))+' |' for row in rows])


def save(fig,name):
    fig.savefig(F/name,dpi=180,bbox_inches='tight');plt.close(fig)


def main():
    F.mkdir(exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'axes.grid':True,'grid.alpha':.2})
    rep=pd.read_csv(R/'core_metrics.csv');s=pd.read_csv(R/'missing_scenarios.csv')
    keys=['variant','seed','split','condition']
    measures=['accuracy','macro_f1','weighted_f1','mae','pearson','raw_mae']
    rep.groupby(keys)[measures].std().fillna(0).to_csv(R/'mask_variability.csv',encoding='utf-8-sig')
    # Average masks within a training run; only then summarize three training seeds.
    c=rep.groupby(keys,as_index=False)[measures].mean()
    selected=json.loads((R/'selection.json').read_text());name=selected['selected_variant'];seed=selected['selected_seed']
    aggregate=c.groupby(['variant','split','condition'])[['accuracy','macro_f1','mae','pearson','raw_mae']].agg(['mean','std']).reset_index()
    aggregate.columns=['_'.join(x).strip('_') if isinstance(x,tuple) else x for x in aggregate.columns]
    aggregate.to_csv(R/'metrics_mean_std.csv',index=False,encoding='utf-8-sig')
    fig,axs=plt.subplots(2,2,figsize=(11,7),layout='constrained',sharey=True)
    for ax,mod in zip(axs.flat,['T','A','V','TAV']):
        for model in s['variant'].unique():
            z=s[(s.variant==model)&(s.group=='rate')&(s.scenario==mod)].groupby('rate').mae.agg(['mean','std'])
            clean=s[(s.variant==model)&(s.group=='clean')].mae.iloc[0]
            xx=np.r_[0,z.index.to_numpy()];yy=np.r_[clean,z['mean'].to_numpy()];err=np.r_[0,z['std'].fillna(0).to_numpy()]
            ax.plot(xx,yy,'o-',label=NAMES[model]);ax.fill_between(xx,yy-err,yy+err,alpha=.15)
        ax.set(title=f'Missing {mod}',xlabel='Target fraction of content positions',ylabel='MAE (lower is better)',xticks=[0,.1,.3,.5,.7])
    axs[0,0].legend(fontsize=8)
    fig.suptitle('Validation robustness: mean ± SD over 3 mask realizations')
    save(fig,'01_missing_rates.png')
    fig,axs=plt.subplots(2,2,figsize=(11,7),layout='constrained',sharey=True)
    for ax,mod in zip(axs.flat,['T','A','V','TAV']):
        for model in s['variant'].unique():
            z=s[(s.variant==model)&(s.group=='rate')&(s.scenario==mod)].groupby('rate').mae.agg(['mean','std'])
            clean=s[(s.variant==model)&(s.group=='clean')].mae.iloc[0]
            xx=np.r_[0,z.index.to_numpy()];yy=np.r_[0,z['mean'].to_numpy()-clean];err=np.r_[0,z['std'].fillna(0).to_numpy()]
            ax.plot(xx,yy,'o-',label=NAMES[model]);ax.fill_between(xx,yy-err,yy+err,alpha=.15)
        ax.axhline(0,color='gray',lw=.8)
        ax.set(title=f'Missing {mod}',xlabel='Target fraction of content positions',ylabel='MAE increase from clean',xticks=[0,.1,.3,.5,.7])
    axs[0,0].legend(fontsize=8)
    fig.suptitle('Validation degradation: fixed training seed, 3 paired mask realizations')
    save(fig,'06_paired_degradation.png')
    z=s[(s.variant==name)&(s.group=='rate')].pivot_table(index='scenario',columns='rate',values='macro_f1').reindex(['T','A','V','TA','TV','AV','TAV'])
    fig,ax=plt.subplots(figsize=(7,5),layout='constrained');im=ax.imshow(z.to_numpy(),cmap='YlGnBu',aspect='auto');ax.grid(False)
    ax.set(xticks=range(4),xticklabels=['10%','30%','50%','70%'],yticks=range(7),yticklabels=z.index,xlabel='Target missing fraction',title='Validation Macro-F1 by missing modality combination')
    for i in range(7):
        for j in range(4):ax.text(j,i,f'{z.iloc[i,j]:.3f}',ha='center',va='center',color='white' if z.iloc[i,j]>(z.min().min()+z.max().max())/2 else 'black')
    fig.colorbar(im,ax=ax,label='Macro-F1 (higher is better)');save(fig,'02_missing_types.png')
    fig,axs=plt.subplots(1,2,figsize=(11,4),layout='constrained')
    pos=s[(s.variant==name)&(s.group=='position')].pivot_table(index='scenario',columns='position',values='mae').reindex(['T','A','V'])[['front','middle','back']]
    pos.plot.bar(ax=axs[0],rot=0);axs[0].set(title='Position effect at 30% missing',ylabel='MAE',xlabel='Missing modality')
    pat=s[(s.variant==name)&(s.group=='pattern')].pivot_table(index='scenario',columns='pattern',values='mae').reindex(['T','A','V'])[['block','two_blocks','points']]
    pat.plot.bar(ax=axs[1],rot=0);axs[1].set(title='Gap structure at 30% missing',ylabel='MAE',xlabel='Missing modality')
    for ax in axs:
        ax.set_ylim(0,max(pos.max().max(),pat.max().max())*1.22);ax.legend(loc='upper center',ncol=3,fontsize=9,frameon=False)
    save(fig,'03_position_and_gap.png')
    v=np.load(R/'validation_predictions.npz');predcls=v['logits'].argmax(1)
    cm=confusion_matrix(v['cls'],predcls,labels=[0,1,2])
    fig,axs=plt.subplots(1,2,figsize=(10,4),layout='constrained')
    axs[0].imshow(cm,cmap='Blues');axs[0].grid(False)
    for i in range(3):
        for j in range(3):axs[0].text(j,i,str(cm[i,j]),ha='center',va='center',color='white' if cm[i,j]>cm.max()/2 else 'black')
    axs[0].set(xticks=[0,1,2],yticks=[0,1,2],xticklabels=['Neg','Neu','Pos'],yticklabels=['Neg','Neu','Pos'],xlabel='Predicted',ylabel='True',title='Validation confusion matrix')
    axs[1].scatter(v['y'],v['pred'],s=12,alpha=.3);axs[1].plot([-3,3],[-3,3],'--',color='black',lw=1)
    axs[1].set(xlim=(-3.1,3.1),ylim=(-3.1,3.1),xlabel='True intensity',ylabel='Predicted intensity',title='Validation intensity (consistent output)')
    save(fig,'04_validation_errors.png')
    fig,axs=plt.subplots(1,2,figsize=(12,4),layout='constrained')
    g=c[(c.split=='valid')&(c.condition=='TAV30')].groupby('variant')[['mae','macro_f1']].agg(['mean','std'])
    order=['text_only','fusion_clean','fusion_aug','robust_norec','robust_rec','robust_nogate'];order=[x for x in order if x in g.index]
    for ax,metric in zip(axs,['mae','macro_f1']):
        vals=g.loc[order,metric];ax.barh(range(len(order)),vals['mean'],xerr=vals['std'],capsize=3,color='#387ea0')
        ax.set(yticks=range(len(order)),yticklabels=[NAMES[x] for x in order],xlabel=metric,title=f'30% TAV missing: {metric}');ax.invert_yaxis()
    save(fig,'05_ablation.png')
    selected_core=c[(c.variant==name)&(c.seed==seed)]
    rows=[]
    for _,x in selected_core.iterrows():rows.append([x['split'],x.condition,f'{x.accuracy:.4f}',f'{x.macro_f1:.4f}',f'{x.mae:.4f}',f'{x.pearson:.4f}'])
    core_table=table(['数据划分','条件','Accuracy','Macro-F1','MAE','Pearson'],rows)
    rows=[]
    for variant in order:
        x=c[(c.variant==variant)&(c.split=='valid')&(c.condition=='TAV30')]
        rows.append([variant,len(x),*[f'{x[k].mean():.4f} ± {x[k].std():.4f}' for k in ['accuracy','macro_f1','mae','pearson']]])
    ablation=table(['模型','训练种子数','Accuracy','Macro-F1','MAE','Pearson'],rows)
    test_rows=[]
    for variant in order:
        x=c[(c.variant==variant)&(c.split=='test')&(c.condition=='TAV30')]
        test_rows.append([variant,*[f'{x[k].mean():.4f} ± {x[k].std():.4f}' for k in ['macro_f1','mae']]])
    test_comparison=table(['模型','测试Macro-F1','测试MAE'],test_rows)
    tt=c[(c.variant==name)&(c.split=='test')&(c.condition=='TAV30')]
    tb=c[(c.variant=='fusion_clean')&(c.split=='test')&(c.condition=='TAV30')]
    test_mae_delta=tt.mae.mean()-tb.mae.mean();test_f1_delta=tt.macro_f1.mean()-tb.macro_f1.mean()
    selected_s=s[(s.variant==name)&(s.group=='rate')]
    at30=selected_s[selected_s.rate==.3].groupby('scenario')[['mae','macro_f1']].mean()
    worse=at30.loc[['T','A','V'],'mae'].idxmax()
    rise=selected_s[selected_s.rate.isin([.1,.7])].groupby(['scenario','rate']).mae.mean().unstack()
    raw_changed=selected_core[(selected_core.split=='valid')&(selected_core.condition=='clean')].iloc[0]
    b=c[(c.variant=='fusion_clean')&(c.split=='valid')&(c.condition=='TAV30')].mae.mean()
    a=c[(c.variant==name)&(c.split=='valid')&(c.condition=='TAV30')].mae.mean()
    improvement=(b-a)/b*100
    posranges=(pos.max(axis=1)-pos.min(axis=1)).to_dict()
    special_table=pd.read_csv(R/'附件3_第二问预测结果.csv')
    labels=special_table.pred_polarity.value_counts().to_dict()
    prediction_table=table(['文件样本编号','预测极性','预测强度'],[[x.sample_id,x.pred_polarity,f'{x.pred_intensity:.6f}'] for _,x in special_table.iterrows()])
    error_groups=[]
    for label,mask in [('真实负向',v['cls']==0),('真实中性',v['cls']==1),('真实正向',v['cls']==2),
                       ('强情感 |y|≥2',np.abs(v['y'])>=2),('短片段 ≤10位置',v['content_length']<=10),
                       ('长片段 ≥30位置',v['content_length']>=30),('视觉原生不可用≥10%',v['native_vision_unavailable']>=.1)]:
        if mask.any():error_groups.append([label,int(mask.sum()),f'{np.mean(np.abs(v["y"][mask]-v["pred"][mask])):.4f}',f'{np.mean(v["cls"][mask]==predcls[mask]):.4f}'])
    group_table=table(['验证子组','样本数','MAE','Accuracy'],error_groups)
    strong=np.abs(v['y'])>=2
    strong_true=float(np.abs(v['y'][strong]).mean());strong_pred=float(np.abs(v['pred'][strong]).mean())
    ranked=sorted(selected['variant_mean_score'].items(),key=lambda x:x[1])
    score_gap=ranked[1][1]-ranked[0][1]
    best_scores=table(['模型','验证选择分数（越低越好）'],[[k,f'{x:.6f}'] for k,x in ranked])
    deg=pd.read_csv(R/'paired_degradation.csv')
    dc=['clean_mae','missing_mae','mae_increase','relative_mae_increase','clean_macro_f1','missing_macro_f1','macro_f1_drop']
    dg=deg.groupby(['variant','seed','split','condition'],as_index=False)[dc].mean()
    da=dg.groupby(['variant','split','condition'])[dc].agg(['mean','std'])
    da.to_csv(R/'degradation_mean_std.csv',encoding='utf-8-sig')
    degradation_tables=[]
    for split in ['valid','test']:
        rows=[]
        for variant in order:
            x=dg[(dg.variant==variant)&(dg.split==split)&(dg.condition=='TAV30')]
            rows.append([variant,f'{x.clean_mae.mean():.4f}',f'{x.missing_mae.mean():.4f}',
                f'{x.mae_increase.mean():+.4f} ± {x.mae_increase.std():.4f}',
                f'{100*x.relative_mae_increase.mean():+.2f}%',
                f'{x.macro_f1_drop.mean():+.4f} ± {x.macro_f1_drop.std():.4f}'])
        degradation_tables.append(table(['模型','clean MAE','缺失 MAE','MAE增量 ± 训练SD','相对MAE增量','Macro-F1下降 ± 训练SD'],rows))
    con=pd.read_csv(R/'paired_baseline_contrasts.csv')
    con=con.groupby(['variant','seed','split','condition'],as_index=False)[['missing_mae_difference','mae_increase_difference','macro_f1_drop_difference']].mean()
    con.groupby(['variant','split','condition']).agg({k:['mean','std'] for k in ['missing_mae_difference','mae_increase_difference','macro_f1_drop_difference']}).to_csv(R/'baseline_contrast_mean_std.csv',encoding='utf-8-sig')
    if name!='fusion_clean':
        cc=con[(con.variant==name)&(con.split=='test')&(con.condition=='TAV30')]
        delta_claim=f"交付模型相对fusion_clean，在重用测试集TAV30下的绝对MAE差为{cc.missing_mae_difference.mean():+.4f}，MAE退化量之差为{cc.mae_increase_difference.mean():+.4f}，Macro-F1下降量之差为{cc.macro_f1_drop_difference.mean():+.4f}（三项均是越小越好）。绝对误差较小与退化较小是两个不同结论，应分别据数值判断。"
    else:
        delta_claim='验证规则选择了fusion_clean作为交付模型。本轮结果不支持把新增的缺失处理模块笼统描述为优于基线。'
    pattern_n=int(s[s.group=='pattern'].n.iloc[0]);pattern_excluded=int(s[s.group=='pattern'].excluded_short.iloc[0])
    auc_rows=[]
    for model in s.variant.unique():
        clean=float(s[(s.variant==model)&(s.group=='clean')].mae.iloc[0])
        for mod,z in s[(s.variant==model)&(s.group=='rate')].groupby('scenario'):
            for ms,zz in z.groupby('mask_seed'):
                zz=zz.sort_values('rate');xx=np.r_[0,zz.rate];yy=np.r_[clean,zz.mae]
                auc=float(np.trapezoid(yy,xx)/.7)
                auc_rows.append(dict(variant=model,scenario=mod,mask_seed=ms,normalized_mae_auc=auc,normalized_degradation_auc=auc-clean))
    pd.DataFrame(auc_rows).to_csv(R/'rate_auc.csv',index=False,encoding='utf-8-sig')
    source_text=f'''# 第二问实验报告

本报告为审计修订版 q2-v2，由重新训练与评价的结果生成。第二问与第一问独立。旧版已使用过附件2测试集，本轮属于修订迭代；测试分数来自重用的留出集，不能称为新的盲测。修订协议在训练前保存于protocol_v2.json。模型在附件2训练集学习、验证集选择，结构与超参数未因测试成绩变化，附件3仅用于最终30条推理。

## 主要结果

最终模型类型为 **{name}**，代表模型采用预先约定的seed {seed}，不是从三个种子中挑最高分。选择依据是五种固定验证条件下“投影后 MAE+1−Macro-F1” 的训练种子均值；测试集与附件3没有参与模型选择。

{best_scores}

排名前两种模型的选择分数相差{score_gap:.6f}。按预先设定的最小值规则确定交付模型，模型排序本身不构成某个模块稳定有效或显著提升的证据。

在验证集“三模态各增加30%连续缺失”条件下，最终模型类型的三训练种子平均MAE为{a:.4f}，未做缺失增强的融合基线为{b:.4f}，相对基线变化为{-improvement:+.2f}%（正数表示误差上升）。该描述只对应这一条件，不能推广为所有缺失场景均有提升。

## 数据及建模

训练/验证/测试数量为3395/728/727，分割之间没有样本ID或video_id交叉。附件3实际只提供text_bert、audio和vision，因此四个阶段统一使用词元输入，不使用附件2的预计算text或raw_text补回缺失信息。BERT-Tiny通用初始化为128维，语音74维，视觉35维。

有效时间掩码、原生观测掩码与人为连续遮挡分别记录。CLS/SEP/填充不计入内容位置。语音/视觉标准化只拟合训练集可用内容。天然全零和人工缺失无法由数值完全区分，原生全零统一视作不可用，不声称知道其真实成因。

网络包含小型BERT、三模态投影、模态内注意力、跨模态注意力、可选动态融合和三分类/回归头。重建版复用TFR-Net的官方Generator模块，以人工遮挡位置的完整训练特征作为辅助监督。它是针对赛题改写的轻量实现，不是官方模型完整复现。

训练损失为 Huber(y,yhat)+0.6×CE(c,chat)，重建版另加0.05×重建损失。CE使用仅由训练集频率计算的平方根逆频率权重。AdamW、BERT学习率1e-4、其他参数7e-4、权重衰减0.01、batch64、最多16轮、5轮早停。全部配置和日志附在项目中。

最终输出通过固定一致性投影保证：中性强度为0，正向严格大于0，负向严格小于0。所有模型评价同用这一规则；原始回归指标也保留在CSV的raw_mae、raw_pearson列。最终代表模型在无新增缺失验证集上，原始MAE为{raw_changed.raw_mae:.4f}，投影后为{raw_changed.mae:.4f}。训练检查点选择、模型类型选择、报告和最终交付全部使用投影后指标，raw列仅作诊断。

## 代表模型基础性能

{core_table}

clean表示没有额外施加人工遮挡，不代表原生特征所有位置均可用。T/A/V/TAV30表示对应模态各增加目标30%的连续位置遮挡。上表是固定训练seed代表模型对3个遮挡种子取均值的结果（clean只计算一次）；所有训练种子均值与标准差见metrics_mean_std.csv。选择阶段始终使用固定遮挡种子20260924，另外两个遮挡种子仅用于稳健性评价。

## 对照与消融

下表为验证集TAV30条件，先对每个训练运行的3个遮挡种子取均值，再给出3个训练种子的均值±标准差。标准差描述这三个训练运行的波动，不是统计显著性的证明。

{ablation}

![消融比较](figures/05_ablation.png)

fusion_clean与fusion_aug比较遮挡训练；robust_norec与robust_rec比较重建；robust_rec与robust_nogate比较动态融合。fusion_aug与robust_norec同时改变缺失表示和融合，不能将差异单独归因为其中某一个模块。更复杂模型未被选中时应按真实结果报告，不将新增模块默认写成有效创新。

重用测试集TAV30结果如下，展示全部三训练种子的均值与标准差，而非只展示代表模型的较好结果：

{test_comparison}

最终模型类型相对fusion_clean的测试MAE变化为{test_mae_delta:+.4f}（负数较好），Macro-F1变化为{test_f1_delta:+.4f}（正数较好）。两个指标应分别判断；不能把其中一个改善写成整体全面胜出。此处测试结果未反过来更换模型或调整超参数。

## 配对退化与结论边界

绝对缺失误差反映最终预测质量；鲁棒性还需要报告同一个模型从clean到缺失条件的变化。定义 ΔMAE=MAE_missing−MAE_clean，F1下降=F1_clean−F1_missing，均越小越好。相对MAE增量为ΔMAE/MAE_clean。每个对比严格配对训练种子、样本与遮挡种子，先平均遮挡随机性，再统计训练运行的标准差。

验证集TAV30：

{degradation_tables[0]}

重用测试集TAV30：

{degradation_tables[1]}

{delta_claim}

paired_degradation.csv保留逐次退化值，paired_baseline_contrasts.csv保留同种子配对基线差异，汇总表分别见degradation_mean_std.csv与baseline_contrast_mean_std.csv。3个训练种子只支持描述性波动分析，不足以宣称统计显著优越；9个“训练种子×遮挡种子”组合不是9次独立训练。mask_variability.csv单独记录每个训练模型的遮挡标准差。

## 缺失规律

![缺失率曲线](figures/01_missing_rates.png)

![配对退化曲线](figures/06_paired_degradation.png)

两图分别显示绝对误差与相对自身clean输入的MAE增量，不能互相替代。

![缺失类型](figures/02_missing_types.png)

在本次固定代表模型中，单模态目标30%缺失时，平均MAE最高的模态为{worse}。各模态从目标10%增加到70%缺失时，MAE的变化分别为：{'; '.join(f'{m}: {row[.1]:.4f} → {row[.7]:.4f}' for m,row in rise.iterrows())}。本次结果对文本缺失更敏感，单独语音或视觉缺失的MAE变化较小；这是本模型和实验条件下的观察，不能推广为模态固有重要性的结论。

随机缺失条件重复三个遮挡种子，图中阴影表示遮挡随机性而非训练随机性。遮挡比例按有效内容位置数定义，原有不可用位置可能使实际新增删除比例略小，逐次结果记录realized_T/A/V。遮挡由“种子、样本ID、模态”独立派生，因此T与TAV条件使用完全相同的文本遮挡，不受样本顺序或分批影响。长度不足时保留至少一个内容位置。rate_auc.csv还给出0至70%范围内的梯形积分平均MAE及相对clean的退化积分；它仅汇总所选固定代表模型的验证曲线，不能替代多训练种子或外部测试。

![位置与区间结构](figures/03_position_and_gap.png)

前/中/后位置的MAE最大差值为：{'; '.join(f'{k}={v:.4f}' for k,v in posranges.items())}。位置实验为确定性遮挡；单区间、双区间和离散点使用相同目标删除预算。本版双区间之间强制至少间隔一个未被本次人工删除的内容位置。模式比较的三种条件统一使用{pattern_n}条样本，均排除{pattern_excluded}条删除预算不足2的短序列；逐行记录样本集合哈希，以便核对各条件的样本集合一致。原生不可用位置仍可能影响实际新增删除数量，所谓严格双区间指人工遮挡的内容时间线。

特征不带逐词时间戳，因此这里的缺失时长仅为序列位置数/占比，不能换算为真实秒数。

## 错误分析

![验证集错误](figures/04_validation_errors.png)

混淆矩阵显示真实类别与预测类别的误判结构，散点图展示情感强度的偏差。验证集绝对误差最大的20条样本以及原文保存在validation_error_cases.csv，供论文逐条核查。该表只列事实，不自动把误差归因于讽刺、否定或模态冲突。

{group_table}

真实中性共有{int(cm[1].sum())}条，其中被判为负向{int(cm[1,0])}条、正向{int(cm[1,2])}条，中性召回率为{cm[1,1]/cm[1].sum():.4f}，中性识别是当前分类的主要短板。强情感子组的真实平均绝对强度为{strong_true:.4f}，预测为{strong_pred:.4f}，{'存在幅度收缩，模型倾向低估强烈情感' if strong_pred<strong_true else '没有观察到平均绝对幅度收缩'}。长短片段和原生视觉不可用子组的差异是条件关联，样本内容与标签组成也不同，不能单独归因为时长或视觉缺失；人为控制的缺失实验才用于支撑缺失影响结论。

## 附件3结果

已生成全部30条预测：附件3_第二问预测结果.csv。类别数量为{json.dumps(labels,ensure_ascii=False)}。文件中的sample_id使用原附件文件名，因为输入未提供原始样本ID。没有真实标签，不能计算附件3准确率或用类别分布推断模型质量。

{prediction_table}

## 复现与局限

- 输入接口、连续区间、严格双区间、跨条件/顺序/子集配对、投影选择口径、残缺完成记录、遮挡泄漏和空输入检查已纳入8项回归测试。
- 官方开源代码：https://github.com/thuiar/TFR-Net ，固定提交00f06a68a44a6b6fc0043f01954336f30ec3b090；复用模块保留MIT许可证。
- 通用编码器：https://huggingface.co/google/bert_uncased_L-2_H-128_A-2 ，固定版本30b0a37ccaaa32f332884b96992754e246e48c5f；未引入外部情感监督数据。
- 未完成官方完整TFR-Net、MMSA、CMAD的原版复现，不应在论文中声称已与这些原版模型比较。当前模型之间均为本项目统一架构下的对照与消融。
- 缺失实验主要在验证集展开，重用测试集只用于本轮方案冻结后的五种预定条件、三个遮挡种子的评价；验证集参与了模型选择，因此其分数存在选择乐观偏差，应同时报告测试结果。
- 本轮未做蒸馏、超参数大规模搜索、未对齐版本和多模态同步/错位遮挡的专门对照，也未进行显著性检验。不能将三个随机种子的差异直接称为显著提升。
- 提交包中历史配置与日志位于reports/training_runs，不会阻止重新训练；只有artifacts/runs中权重、完成记录、配置、源码和数据指纹一致时才允许跳过。无需重训即可通过src/infer.py使用最终权重离线推理。训练环境、运行命令与隔离烟雾训练见README.md。
'''
    (R/'实验报告.md').write_text(source_text,encoding='utf-8')
    print('Generated report and',len(list(F.glob('*.png'))),'figures')

if __name__=='__main__':main()
