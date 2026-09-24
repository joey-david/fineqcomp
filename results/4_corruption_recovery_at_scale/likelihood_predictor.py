"""Score the likelihood predictor against measured accuracy (spectral_breadth_v1). Run from the repo root."""
import json,glob
def rank(v):
    o=sorted(range(len(v)),key=lambda i:v[i]); r=[0]*len(v)
    i=0
    while i<len(o):
        j=i
        while j+1<len(o) and v[o[j+1]]==v[o[i]]: j+=1
        for t in range(i,j+1): r[o[t]]=(i+j)/2
        i=j+1
    return r
def spear(a,b):
    ra,rb=rank(a),rank(b); n=len(a); ma,mb=sum(ra)/n,sum(rb)/n
    c=sum((x-ma)*(y-mb) for x,y in zip(ra,rb)); return c/(sum((x-ma)**2 for x in ra)*sum((y-mb)**2 for y in rb))**.5
B='.cache/reports/spectral_breadth_v1'
lik={}
for f in glob.glob(f'{B}/likelihood/*/*.json'):
    d=json.load(open(f)); lik.setdefault(d['condition']['key'],{})[d['split']]=d['bits_per_token']
acc,norm={},{}
for f in glob.glob(f'{B}/search/*/search/*.json'):
    d=json.load(open(f)); k=f.split('/')[-3]; acc[k]=d['exact_match']; norm[k]=d.get('update_norm',0.0)
keys=[k for k in acc if k in lik]
x=[lik[k]['selection'] for k in keys]; y=[acc[k] for k in keys]
print(f"near: {len(keys)} conditions, spearman(-bits on selection, search acc) = {spear([-v for v in x],y):.3f}")
print(f"  baseline spearman(-update_norm, acc) = {spear([-norm[k] for k in keys],y):.3f}")
best=max(y); pick=keys[x.index(min(x))]
print(f"  oracle best {best:.3f} ({keys[y.index(best)]}); likelihood pick {pick} -> {acc[pick]:.3f}")
top=sorted(keys,key=lambda k:lik[k]['selection'])[:6]; print("  top-6 by likelihood:",[(k,round(acc[k],3),round(lik[k]['selection'],4)) for k in top])
for k in ['base','clean_raw','permuted_raw','band00_01_binary','band00_16_binary','band04_16_fp16','band01_02_fp16','scale_r1_0.5']:
    if k in lik: print(f"  {k:18s} bits={lik[k]['selection']:.4f} search_acc={acc.get(k,float('nan')):.3f}")

import csv
panel={r['probe']:r for r in csv.DictReader(open('results/4_corruption_recovery_at_scale/breadth_panel.csv'))}
# Across probes: does the clean-minus-base likelihood change predict the clean-minus-base accuracy change?
dl,da,dlc,dac=[],[],[],[]
for p,r in panel.items():
    if p not in lik['base']: continue
    base=float(r['base_second_half'])
    dl.append(lik['base'][p]-lik['clean_raw'][p]); da.append(float(r['clean_second_half'])-base)
    dlc.append(lik['base'][p]-lik['band00_01_binary'][p]); dac.append(float(r['compressed_second_half'])-base)
print(f"far, across {len(dl)} probes: spearman(bits saved by clean, clean acc gain) = {spear(dl,da):.3f}")
print(f"                        spearman(bits saved by compressed, compressed acc gain) = {spear(dlc,dac):.3f}")
hurt=[p for p,r in panel.items() if r['clean_hurts_first_half']=='True' and r['distance']!='0']
pred=[p for p in panel if p in lik['base'] and lik['clean_raw'][p]>lik['base'][p]]
print(f"  clean hurts (first half): {len(hurt)}; likelihood says clean worse than base on {len(pred)}; overlap {len(set(hurt)&set(pred))}")
# which adapter is better per probe: clean vs compressed
agree=tot=0
for p,r in panel.items():
    if p not in lik['base']: continue
    a=float(r['compressed_second_half'])-float(r['clean_second_half'])
    if a==0: continue
    tot+=1; agree+= (a>0)==(lik['band00_01_binary'][p]<lik['clean_raw'][p])
print(f"  picks the better of clean vs compressed on {agree}/{tot} probes")
