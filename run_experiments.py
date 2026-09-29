"""Run HRC and baseline experiments using pre-generated LLM responses."""
from __future__ import annotations

from dataclasses import asdict, replace
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import t, ttest_1samp
from data_utils import ProjectConfig, validate_human, validate_outcomes
from evaluation import (
    MixedTypeEmbedding, compare_association, density_ratio_weights, metric_schema,
    raking_ipf_weights, sqrt_energy_distance, stratified_nested_order,
    uniform_tvae_sample, weighted_resample_indices,
)
from hrc import (
    CandidatePool, SeedPlan, derive_seed, fit_hrc_bundle,
    hybrid_selection_probabilities, seed_everything,
)


# Public experiments

ROOT=Path(__file__).resolve().parent

FAMILIES=['DeepSeek','Qwen','GPT4','GPT5']

SEEDS=[11,29,42,71,101]

def load_inputs(data_dir, config):
    directory=Path(data_dir); c=config.columns
    pool=pd.read_csv(directory/'calibration_pool.csv',dtype={c.id_col:str})
    benchmark=pd.read_csv(directory/'benchmark.csv',dtype={c.id_col:str})
    for name,frame in [('calibration pool',pool),('benchmark',benchmark)]:
        validate_human(frame,config,name,require_stratum=name=='calibration pool')
    if set(pool[c.id_col]) & set(benchmark[c.id_col]):raise ValueError('Calibration/benchmark ID overlap')
    if c.stratify_col not in pool or pool[c.stratify_col].isna().any():raise ValueError('Missing strata')
    llms={}
    for family in FAMILIES:
        outcomes=pd.read_csv(directory/f'llm/{family}.csv',dtype={c.id_col:str})
        if c.id_col not in outcomes:raise ValueError(f'{family}: missing ID column')
        if set(outcomes[c.id_col])!=set(benchmark[c.id_col]) or outcomes[c.id_col].duplicated().any():
            raise ValueError(f'{family}: IDs must match benchmark exactly')
        validate_outcomes(outcomes,config,family)
        llms[family]=benchmark[[c.id_col,*c.condition_cols]].merge(
            outcomes[[c.id_col,*c.outcome_cols]],on=c.id_col,validate='one_to_one',sort=False)
    return pool,benchmark,llms

def metrics(reference, synthetic, config, embedding):
    association=compare_association(reference,synthetic,config.columns.metric_ordered,config.columns.metric_nominal)
    arr=association.difference.to_numpy()
    upper=arr[np.triu_indices_from(arr,k=1)]
    # Explicitly retain a count instead of silently claiming every pair is defined.
    return {'sqrt_energy_distance':sqrt_energy_distance(embedding.transform(reference),embedding.transform(synthetic)),
            'association_rmse':float(np.sqrt(np.mean(upper**2))) if np.isfinite(upper).all() else np.nan,
            'undefined_association_pairs':int((~np.isfinite(upper)).sum())}

def paired_select(frame,bundle,pools,seed,family,variant):
    c=bundle.calibrator
    selected=[]
    for _,row in frame.iterrows():
        key=row[bundle.id_col]
        pool=pools[key]
        distance=np.linalg.norm(pool.encoded-c.outcome_encoder.transform(pd.DataFrame([row[list(c.outcome_cols)]]))[0],axis=1)
        probability,_,_=hybrid_selection_probabilities(distance,pool.human_score,gamma=variant.gamma,
            scaling=variant.score_scaling,temperature=variant.temperature)
        u=np.random.default_rng(derive_seed(seed,'public_paired_selection',family,key)).random()
        index=min(int(np.searchsorted(np.cumsum(probability),u,side='right')),len(probability)-1)
        selected.append(pool.encoded[index])
    outcomes=c.outcome_encoder.inverse_transform(np.vstack(selected))
    result=frame.copy().reset_index(drop=True)
    for column in c.outcome_cols:result[column]=outcomes[column].to_numpy()
    return result

def variants(config, robust, demo):
    yield 'default','default',config
    grid=([0.,.5,1.],[.5,1.]) if demo else (robust['gamma_selection_smoothness']['gamma_values'],robust['gamma_selection_smoothness']['selection_smoothness_values'])
    for gamma in grid[0]:
        for smooth in grid[1]:
            yield f'gamma={gamma};smoothness={smooth}','selection',replace(config,calibration=replace(config.calibration,gamma=gamma,temperature=smooth))
    for k in ([8,16,32] if demo else robust['candidate_pool']['values']):
        yield f'K={k}','candidate_pool',replace(config,calibration=replace(config.calibration,candidate_pool_size=k))
    grids={'hidden_dim':[16,32], 'latent_dim':[4,8], 'epochs':[1,2,3]} if demo else {k:robust['tvae_ofat'][k+'_values'] for k in ['hidden_dim','latent_dim','epochs']}
    for key,values in grids.items():
        for value in values:
            yield f'{key}={value}','generator',replace(config,model=replace(config.model,**{key:value}))

def interval(values):
    values=np.asarray(values,dtype=float)
    if not np.isfinite(values).all():return {'mean':np.nan,'sd':np.nan,'ci_low':np.nan,'ci_high':np.nan,'n':len(values)}
    mean=float(values.mean());sd=float(values.std(ddof=1)) if len(values)>1 else np.nan
    half=float(t.ppf(.975,len(values)-1)*sd/np.sqrt(len(values))) if len(values)>1 else np.nan
    return {'mean':mean,'sd':sd,'ci_low':mean-half,'ci_high':mean+half,'n':len(values)}

def summarize(frame, output):
    keys=['dataset','setting','block','train_size','method']
    rows=[]
    metrics_cols=['sqrt_energy_distance','association_rmse']
    seed_macro=frame.groupby([*keys,'seed'],as_index=False)[metrics_cols].agg(lambda s: s.mean() if s.notna().all() else np.nan)
    seed_macro.to_csv(output/'seed_macro.csv',index=False)
    for key,group in frame.groupby(keys):
        for metric in metrics_cols:
            if key[-1]=='Raw LLM':
                values=group.drop_duplicates('llm')[metric]
                stats=interval(values);stats.update(ci_low=np.nan,ci_high=np.nan)
                unit='fixed_llm_conditions'
            else:
                values=group.groupby('seed')[metric].agg(lambda s: s.mean() if s.notna().all() else np.nan)
                stats=interval(values);unit='seed'
            rows.append({**dict(zip(keys,key)),'metric':metric,'unit':unit,**stats})
    pd.DataFrame(rows).to_csv(output/'summary.csv',index=False)
    raw=frame[frame.method.eq('Raw LLM')][['dataset','seed','train_size','llm','sqrt_energy_distance']]
    hrc=frame[frame.method.eq('HRC')].merge(raw.rename(columns={'sqrt_energy_distance':'raw_distance'}),on=['dataset','seed','train_size','llm'])
    if len(hrc):
        hrc['benefit_percent']=100*(1-hrc.sqrt_energy_distance/hrc.raw_distance.replace(0,np.nan))
        benefit=hrc.groupby(['dataset','setting','train_size','seed'],as_index=False).benefit_percent.mean()
        benefit.to_csv(output/'benefit_per_seed.csv',index=False)
    default=seed_macro[seed_macro.setting.eq('default') & seed_macro.method.eq('HRC')][['dataset','train_size','seed','sqrt_energy_distance']]
    if len(default):
        paired=seed_macro[seed_macro.method.eq('HRC')].merge(default.rename(columns={'sqrt_energy_distance':'default_distance'}),on=['dataset','train_size','seed'])
        paired['relative_change_percent']=100*(paired.sqrt_energy_distance/paired.default_distance-1)
        paired.to_csv(output/'paired_relative_change.csv',index=False)

def run(country,data_dir,output,mode,*,demo=False,seeds=None,sizes=None,device='cpu'):
    import torch
    torch.set_num_threads(1)
    output=Path(output)
    if output.exists():raise FileExistsError('Use a new output directory; existing results are not overwritten')
    output.mkdir(parents=True)
    config=ProjectConfig.load(ROOT/f'configs/{country}.json')
    if demo:
        config=replace(config,model=replace(config.model,epochs=2,hidden_dim=32,latent_dim=8),calibration=replace(config.calibration,candidate_pool_size=16))
    pool,benchmark,llms=load_inputs(data_dir,config)
    seeds=seeds or ([11,29] if demo else SEEDS)
    if not sizes:
        if mode=='efficiency':
            sizes=[40,80] if demo else json.loads((ROOT/'configs/experiment_grid.json').read_text())[country]
        else:sizes=[80 if demo else 600]
    if min(sizes)<2 or max(sizes)>len(pool):raise ValueError('Training size outside available pool')
    robust=json.loads((ROOT/'configs/robustness.json').read_text())
    embedding=MixedTypeEmbedding(metric_schema(config)).fit(benchmark)
    raw_metrics={family:metrics(benchmark,frame,config,embedding) for family,frame in llms.items()}
    table=[];diagnostics=[];structural=[];probabilities=[]
    (output/'run_config.json').write_text(json.dumps({'dataset':country,'mode':mode,'fictional_demo':demo,'seeds':seeds,'sizes':sizes,'device':device,'config':config.to_dict()},indent=2),encoding='utf-8')
    for size in sizes:
        for seed in seeds:
            print(f'{country} {mode}: n={size}, seed={seed}',flush=True)
            order=stratified_nested_order(pool,stratify_col=config.columns.stratify_col,seed=derive_seed(seed,'vb_nested_sample_order','China' if country=='china' else 'UK'))
            training=pool.iloc[order[:size]].reset_index(drop=True)
            plan=SeedPlan(seed,f'public|{country}|n={size}')
            seed_everything(plan.for_stage('process'))
            choices=list(variants(config,robust,demo)) if mode=='robustness' else [('default','default',config)]
            largest=max(v.calibration.candidate_pool_size for _,_,v in choices)
            bundles={}
            for setting,block,variant in choices:
                model_key=json.dumps(asdict(variant.model),sort_keys=True)
                if model_key not in bundles:
                    fit_config=replace(variant,calibration=replace(variant.calibration,candidate_pool_size=largest))
                    bundle=fit_hrc_bundle(training,config=fit_config,seed_plan=plan,device=device)
                    bundle.id_col=config.columns.id_col
                    pools={}
                    for _,record in benchmark.iterrows():
                        key=record[config.columns.id_col]
                        pools[key]=bundle.calibrator.generate_candidate_pool(record,record_key=key)
                    bundles[model_key]=(bundle,pools)
                bundle,big_pools=bundles[model_key]
                k=variant.calibration.candidate_pool_size
                pools={key:CandidatePool(p.encoded[:k],p.human_score[:k],p.candidate_seed) for key,p in big_pools.items()}
                common={'dataset':country,'setting':setting,'block':block,'train_size':size,'seed':seed}
                for family,frame in llms.items():
                    calibrated=paired_select(frame,bundle,pools,seed,family,variant.calibration)
                    table.append({**common,'method':'HRC','llm':family,**metrics(benchmark,calibrated,config,embedding)})
                    if setting=='default':table.append({**common,'method':'Raw LLM','llm':family,**raw_metrics[family]})
                    if mode=='structure':
                        for label,data in [('Human',benchmark),('Raw LLM',frame),('HRC',calibrated)]:
                            comparison=compare_association(benchmark,data,config.columns.metric_ordered,config.columns.metric_nominal)
                            for a in comparison.synthetic.index:
                                for b in comparison.synthetic.columns:
                                    structural.append({**common,'llm':family,'method':label,'variable_a':a,'variable_b':b,'association':comparison.synthetic.loc[a,b],'difference':comparison.difference.loc[a,b]})
                            for variable in config.columns.outcome_cols:
                                shares=data[variable].value_counts(normalize=True)
                                for category in config.columns.category_levels[variable]:
                                    probabilities.append({**common,'llm':family,'method':label,'variable':variable,'category':category,'probability':shares.get(category,0.)})
                    if mode=='comparison':
                        c=config.columns
                        weights,diag=raking_ipf_weights(training,frame,categorical_cols=c.outcome_categorical,numerical_cols=c.outcome_numeric)
                        density,ddiag=density_ratio_weights(training,frame,categorical_cols=(*c.condition_categorical,*c.outcome_categorical),numerical_cols=(*c.condition_numeric,*c.outcome_numeric),seed=derive_seed(seed,'density',country,family))
                        for label,w,d in [('Raking/IPF',weights,diag),('Density-ratio weighting',density,ddiag)]:
                            idx=weighted_resample_indices(w,n=len(benchmark),seed=derive_seed(seed,'resample',country,family,label))
                            table.append({**common,'method':label,'llm':family,**metrics(benchmark,frame.iloc[idx],config,embedding)})
                            diagnostics.append({**common,'llm':family,**asdict(d)})
                if setting=='default' and mode in {'comparison','efficiency'}:
                    pure=uniform_tvae_sample(benchmark,config=config,seeds=plan,pools=pools,outcome_encoder=bundle.outcome_encoder)
                    table.append({**common,'method':'Pure conditional TVAE','llm':'independent',**metrics(benchmark,pure,config,embedding)})
    frame=pd.DataFrame(table)
    frame.to_csv(output/'per_run_metrics.csv',index=False)
    summarize(frame,output)
    for name,rows in [('weighting_diagnostics',diagnostics),('associations',structural),('category_probabilities',probabilities)]:
        if rows:pd.DataFrame(rows).to_csv(output/f'{name}.csv',index=False)
    if mode=='structure':
        try:
            summarize_structure(frame,country,output,demo=demo,seeds=seeds,sizes=sizes)
        except (OSError,ValueError) as exc:
            print(f'Structure results saved; significance not written: {exc}',flush=True)
    return frame


def summarize_structure(frame, country, output, *, demo=False, seeds=None, sizes=None):
    """Write RMSE tests with Holm correction over the four LLMs in one country."""
    output = Path(output)
    sizes = list(sizes) if sizes is not None else [80 if demo else 600]
    seeds = list(seeds) if seeds is not None else ([11, 29] if demo else SEEDS)
    if len(sizes) != 1 or sizes[0] < 2:
        raise ValueError('Significance testing requires exactly one --sizes value of at least 2')
    if len(seeds) < 2 or len(set(seeds)) != len(seeds):
        raise ValueError('Significance testing requires at least two distinct seeds')
    train_size = sizes[0]
    if country not in ('china','england'):
        raise ValueError('Country must be china or england')

    required = {'dataset', 'setting', 'train_size', 'seed',
                'method', 'llm', 'association_rmse'}
    if not required.issubset(frame.columns):
        raise ValueError('Missing required metric columns')
    if not frame['dataset'].eq(country).all():
        raise ValueError(f'All metric rows must belong to {country}')
    frame = frame.loc[
        frame['setting'].eq('default') & frame['train_size'].eq(train_size)
        & frame['seed'].isin(seeds)
        & frame['method'].isin(['Raw LLM', 'HRC'])
    ].copy()
    keys = ['dataset', 'llm', 'seed', 'method']
    expected = pd.MultiIndex.from_product(
        [[country], FAMILIES, seeds, ['Raw LLM', 'HRC']],
        names=keys,
    )
    frame = frame.set_index(keys, verify_integrity=True)
    if len(frame) != len(expected) or len(expected.difference(frame.index)):
        raise ValueError(f'Need four LLMs and seeds {seeds} for {country} at n={train_size}')
    frame = frame.reindex(expected)
    values = frame['association_rmse'].to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError('RMSE values must be finite and nonnegative')

    rows = []
    for llm in FAMILIES:
        raw = np.array([frame.loc[(country, llm, s, 'Raw LLM'),
                                  'association_rmse'] for s in seeds])
        hrc = np.array([frame.loc[(country, llm, s, 'HRC'),
                                  'association_rmse'] for s in seeds])
        if not np.allclose(raw, raw[0], rtol=0, atol=1e-12):
            raise ValueError(f'{country}/{llm}: Raw RMSE is not fixed')
        differences = raw[0] - hrc
        sd = differences.std(ddof=1)
        if not np.isfinite(sd) or sd <= 0:
            raise ValueError(f'{country}/{llm}: invalid difference variance')
        test = ttest_1samp(differences, popmean=0, alternative='two-sided')
        if not np.isfinite([test.statistic, test.pvalue]).all():
            raise ValueError(f'{country}/{llm}: invalid test result')
        rows.append({'country': country, 'llm': llm, 'p_raw': test.pvalue})

    # Each country is a separate family of four comparisons.
    result = pd.DataFrame(rows)
    p_values = result['p_raw'].to_numpy()
    order = np.argsort(p_values)
    adjusted = np.empty_like(p_values)
    adjusted[order] = np.minimum(
        1.0, np.maximum.accumulate(p_values[order] * np.arange(len(p_values), 0, -1))
    )
    result['p_holm'] = adjusted
    result['significance'] = np.select(
        [adjusted < 0.001, adjusted < 0.01, adjusted < 0.05],
        ['***', '**', '*'], default='n.s.',
    )
    result = result[['country', 'llm', 'p_holm', 'significance']]
    output_file = output/'significance.csv'
    result.to_csv(output_file,index=False,mode='x')
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description='Run HRC and baseline experiments on local data.')
    p.add_argument('--country',choices=['china','england'],required=True)
    p.add_argument('--mode',choices=['comparison','efficiency','structure','robustness'],required=True)
    p.add_argument('--demo',action='store_true')
    p.add_argument('--data-dir',type=Path)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--seeds',type=int,nargs='+')
    p.add_argument('--sizes',type=int,nargs='+')
    p.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    a=p.parse_args()
    if not a.demo and a.data_dir is None:
        p.error('Supply --data-dir for restricted research data, or use --demo')
    path=a.data_dir or ROOT/f'data/demo/{a.country}'
    run(a.country,path,a.output,a.mode,demo=a.demo,seeds=a.seeds,sizes=a.sizes,device=a.device)
