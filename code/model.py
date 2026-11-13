"""Учебная сеть дронопортов. Денежная единица модели: миллион рублей.
Ограниченный класс решений: маятниковые маршруты с 1–2 участками,
закреплёнными дронами и резервированием места в каждом посещаемом порту.
Оптимальность MILP относится только к этому классу и набору шаблонов.
"""
from pathlib import Path
import json, time, math, argparse, heapq
from collections import deque
import numpy as np
import pandas as pd
from scipy.optimize import milp, linprog, Bounds, LinearConstraint
from scipy.sparse import coo_matrix, csr_matrix, block_diag, hstack

ROOT = Path(__file__).resolve().parents[1]
PARAMS = dict(speed_kmh=180, service_minutes=35, work_minutes=600,
              days=300, seats=10, range_km=200, port_million=60,
              drone_million=100, port_annual_million=5,
              drone_annual_million=11, flight_million_per_km=.000015,
              reserved_places=3, annual_budget_million=1000,
              demand_K=.003, demand_decay_km=150, demand_growth=.08,
              max_two_leg_templates=90)

def distances(df):
    lat=np.deg2rad(df.latitude_dd.to_numpy(float)); lon=np.deg2rad(df.longitude_dd.to_numpy(float))
    a=np.sin((lat[:,None]-lat)/2)**2+np.cos(lat[:,None])*np.cos(lat)*np.sin((lon[:,None]-lon)/2)**2
    return 6371*2*np.arcsin(np.sqrt(np.clip(a,0,1)))

def demand(df, p=PARAMS):
    pop=df.population.to_numpy(float); d=distances(df)
    q=p['demand_K']*np.sqrt(pop[:,None]*pop)*np.exp(-d/p['demand_decay_km'])
    np.fill_diagonal(q,0)
    # Целое количество поездок за год. Суточная величина - среднее.
    return np.array([np.floor(q*(1+p['demand_growth'])**t*p['days'])/p['days'] for t in range(3)])

class Instance:
    def __init__(self, df, p=None):
        self.df=df.reset_index(drop=True); self.n=len(df); self.p={**PARAMS,**(p or {})}
        self.d=distances(self.df); self.Q=demand(self.df,self.p); self.routes=[]
        for i in range(self.n):
            for j in range(i+1,self.n):
                if self.d[i,j]<=self.p['range_km']: self.routes.append((i,j))
        longer=[]
        for i in range(self.n):
            for j in range(i+1,self.n):
                if self.d[i,j]<=self.p['range_km']: continue
                ks=[k for k in range(self.n) if k not in (i,j) and max(self.d[i,k],self.d[k,j])<=self.p['range_km']]
                if ks:
                    k=min(ks,key=lambda k:self.d[i,k]+self.d[k,j])
                    longer.append((self.Q[0,i,j],(i,k,j)))
        longer.sort(key=lambda a:(-a[0],a[1]))
        self.routes += [r for _,r in longer[:self.p['max_two_leg_templates']]]
        self.R=len(self.routes); self.cycles=[]; self.flightcost=[]; self.tripvars=[]; seg=[]
        self.inc=np.zeros((self.n,self.R))
        segrow=0
        for r,path in enumerate(self.routes):
            self.inc[list(path),r]=1
            fl=[5*math.ceil((self.d[a,b]/self.p['speed_kmh']*60)/5) for a,b in zip(path,path[1:])]
            dur=2*sum(f+self.p['service_minutes'] for f in fl)
            cyc=self.p['work_minutes']//dur; self.cycles.append(cyc)
            km=2*sum(self.d[a,b] for a,b in zip(path,path[1:]))*cyc*self.p['days']
            self.flightcost.append(km*self.p['flight_million_per_km'])
            for direction in (1,-1):
                pp=path if direction==1 else path[::-1]
                new=[]
                for a in range(len(pp)):
                    for b in range(a+1,len(pp)):
                        idx=len(self.tripvars); self.tripvars.append((r,pp[a],pp[b],a,b,direction)); new.append(idx)
                for k in range(len(pp)-1):
                    seg.append((r,[v for v in new if self.tripvars[v][3]<=k<self.tripvars[v][4]])); segrow+=1
        self.cycles=np.array(self.cycles); self.flightcost=np.array(self.flightcost)
        self.V=len(self.tripvars); self.segments=seg
        rows=[];cols=[];vals=[]
        for v,(_,o,d,*_) in enumerate(self.tripvars): rows.append(o*self.n+d);cols.append(v);vals.append(1)
        self.od=coo_matrix((vals,(rows,cols)),shape=(self.n*self.n,self.V)).tocsr()
        rows=[];cols=[]
        for k,(_,vs) in enumerate(seg): rows+= [k]*len(vs);cols+=vs
        self.seat=coo_matrix((np.ones(len(rows)),(rows,cols)),shape=(len(seg),self.V)).tocsr()
        from scipy.sparse import vstack
        self.lpA=vstack([self.od,self.seat],format='csr')

    def allocate(self,z,t=0,factor=1,range_km=None,failed=None,ret=False):
        z=np.array(z,float).copy()
        for r,path in enumerate(self.routes):
            if (failed is not None and failed in path) or (range_km is not None and any(self.d[a,b]>range_km for a,b in zip(path,path[1:]))): z[r]=0
        caps=[self.p['seats']*self.cycles[r]*z[r] for r,_ in self.segments]
        b=np.r_[self.Q[t].ravel()*factor,caps]
        sol=linprog(-np.ones(self.V),A_ub=self.lpA,b_ub=b,bounds=(0,None),method='highs')
        if not sol.success: raise RuntimeError(sol.message)
        return (-sol.fun,sol.x) if ret else -sol.fun

    def accounting(self,Z):
        Z=np.asarray(Z,int); x=np.maximum.accumulate((Z@self.inc.T)>0,axis=0).astype(int)
        N=np.maximum.accumulate(Z.sum(axis=1)); dx=np.diff(np.vstack([np.zeros(self.n),x]),axis=0)
        buy=np.diff(np.r_[0,N]); cap=dx.sum(axis=1)*self.p['port_million']+buy*self.p['drone_million']
        op=x.sum(axis=1)*self.p['port_annual_million']+N*self.p['drone_annual_million']+Z@self.flightcost
        return dict(x=x,N=N,buy=buy,cap=cap,op=op,cost=cap+op)

    def feasible(self,Z,budget=1000):
        if np.min(Z)<0: return False
        if np.any(np.array(Z)@self.inc.T>self.p['reserved_places']): return False
        a=self.accounting(Z)
        return bool(np.all(np.cumsum(a['cost'])<=budget*np.arange(1,4)+1e-6))

    def score(self,Z): return sum(self.allocate(z,t) for t,z in enumerate(Z))*self.p['days']

    def solve(self,budget=1000,limit=20,years=3,fixed_previous=None):
        """MILP: порты, накопленный флот, назначения дронов и потоки по шаблонам."""
        start=time.perf_counter(); n,R,V=self.n,self.R,self.V
        B=n+1+R+V; T=years; total=B*T; c=np.zeros(total); integ=np.zeros(total); ub=np.full(total,np.inf)
        # c измеряет среднесуточные поездки, итог умножается на days.
        for t in range(T):
            off=t*B; integ[off:off+n+1+R]=1; ub[off:off+n]=1; ub[off+n]=30;ub[off+n+1:off+n+1+R]=3
            c[off+n+1+R:off+B]=-1
        rr=[];cc=[];vv=[]; lows=[]; highs=[]
        def row(items,lo=-np.inf,hi=np.inf):
            ix=len(lows)
            for j,v in items:
                if v:rr.append(ix);cc.append(j);vv.append(v)
            lows.append(lo);highs.append(hi)
        for t in range(T):
            off=t*B;zi=off+n+1;fi=zi+R
            for i in range(n):
                row([(zi+r,self.inc[i,r]) for r in range(R)]+[(off+i,-3)],hi=0)
                if t: row([(off-B+i,1),(off+i,-1)],hi=0)
            row([(zi+r,1) for r in range(R)]+[(off+n,-1)],hi=0)
            if t: row([(off-B+n,1),(off+n,-1)],hi=0)
            for od in range(n*n):
                s,e=self.od.indptr[od:od+2]
                if e>s: row([(fi+int(v),1) for v in self.od.indices[s:e]],hi=self.Q[t].ravel()[od])
            for r,vs in self.segments: row([(fi+v,1) for v in vs]+[(zi+r,-10*self.cycles[r])],hi=0)
            terms=[]
            for s in range(t+1):
                os=s*B
                terms += [(os+i,self.p['port_annual_million']) for i in range(n)]
                terms += [(os+n,self.p['drone_annual_million'])]
                terms += [(os+n+1+r,self.flightcost[r]) for r in range(R)]
            terms += [(off+i,self.p['port_million']) for i in range(n)]+[(off+n,self.p['drone_million'])]
            row(terms,hi=(t+1)*budget)
        if fixed_previous is not None:
            for t,z in enumerate(fixed_previous):
                for r,a in enumerate(z): row([(t*B+n+1+r,1)],lo=int(a),hi=int(a))
                acc=self.accounting(np.vstack([fixed_previous,np.zeros((3-len(fixed_previous),R),int)]))
                for i,a in enumerate(acc['x'][t]): row([(t*B+i,1)],lo=int(a),hi=int(a))
                row([(t*B+n,1)],lo=int(acc['N'][t]),hi=int(acc['N'][t]))
        A=coo_matrix((vv,(rr,cc)),shape=(len(lows),total)).tocsc()
        res=milp(c,integrality=integ,bounds=Bounds(np.zeros(total),ub),constraints=LinearConstraint(A,lows,highs),options={'time_limit':limit,'mip_rel_gap':.001})
        if res.x is None: raise RuntimeError('MILP не нашла допустимое решение: '+res.message)
        Z=np.array([np.rint(res.x[t*B+n+1:t*B+n+1+R]).astype(int) for t in range(T)])
        lower=-float(res.fun)*self.p['days']; upper=-float(res.mip_dual_bound)*self.p['days']
        return dict(Z=Z,objective=lower,upper_bound=upper,gap=float(res.mip_gap),status=int(res.status),message=res.message,seconds=time.perf_counter()-start)

    def greedy(self,budget=1000,seed=None):
        start=time.perf_counter(); Z=np.zeros((3,self.R),int); rng=np.random.default_rng(seed)
        standalone=np.array([self.allocate(np.eye(1,self.R,r).ravel()) for r in range(self.R)])
        for t in range(3):
            if t: Z[t]=Z[t-1]
            while True:
                base=self.allocate(Z[t],t); a=self.accounting(Z); props=[]
                for r in range(self.R):
                    W=Z.copy();W[t:,r]+=1
                    if not self.feasible(W,budget):continue
                    delta=self.accounting(W)['cost'].sum()-a['cost'].sum()
                    # Список кандидатов до точной LP-оценки приращения.
                    rank=standalone[r]/max(delta,1)
                    if seed is not None:rank*=rng.uniform(.7,1.3)
                    props.append((rank,r,delta))
                best=None
                for _,r,cost in sorted(props,reverse=True)[:12]:
                    w=Z[t].copy();w[r]+=1; gain=self.allocate(w,t)-base
                    merit=gain/max(cost,1)
                    if best is None or merit>best[0]:best=(merit,r,gain)
                if best is None or best[2]<1e-5:break
                Z[t,best[1]]+=1
        assert self.feasible(Z,budget)
        return dict(Z=Z,objective=self.score(Z),seconds=time.perf_counter()-start)

    def anneal(self,initial,seed=0,budget=1000,steps=100):
        start=time.perf_counter();rng=np.random.default_rng(seed);Z=initial.copy(); val=self.score(Z);best=Z.copy();bv=val
        for k in range(steps):
            W=Z.copy();t=int(rng.integers(3));r=int(rng.integers(self.R));mode=int(rng.integers(3))
            if mode==0:W[t:,r]+=1
            else:
                used=np.flatnonzero(W[t]>0)
                if not len(used):continue
                old=int(rng.choice(used));W[t:,old]-=1
                if mode==1:W[t:,r]+=1
            if not self.feasible(W,budget):continue
            v=self.score(W);temp=max(10, .02*max(val,1)*(1-k/steps))
            if v>val or rng.random()<math.exp(min(0,(v-val)/temp)):Z=W;val=v
            if v>bv:best=W.copy();bv=v
        return dict(Z=best,objective=bv,seconds=time.perf_counter()-start,seed=seed)

def clean_and_select():
    raw=pd.read_csv(ROOT/'data/raw/region_letter_extract.csv')
    raw['settlement']=raw.settlement.str.strip(); raw['population_year']=2020
    raw['population_status']='наблюдение из исторического набора, период 2020'
    raw['source']='https://data.rcsi.science/data-catalog/datasets/160/'
    keys=['region','municipality','settlement','type']
    good=raw.drop_duplicates(keys).copy()
    valid=good.latitude_dd.between(49,56)&good.longitude_dd.between(41,51)&(good.population>=0)
    invalid=good[~valid];good=good[valid]
    # Правило до оптимизации: десять наиболее населённых в каждом регионе.
    selected=good.sort_values(['region','population','id'],ascending=[True,False,True]).groupby('region',sort=True).head(10).reset_index(drop=True)
    good.to_csv(ROOT/'data/processed/all_eligible.csv',index=False)
    selected.to_csv(ROOT/'data/processed/selected_30.csv',index=False)
    invalid.to_csv(ROOT/'data/processed/invalid_coordinates.csv',index=False)
    log=dict(raw_count=len(raw),clean_count=len(good),invalid_count=len(invalid),duplicates=len(raw)-len(good)-len(invalid),selected=len(selected),rule='top 10 population per region; tie=id; fixed before optimization',source_period=2020)
    (ROOT/'data/processed/transformation_log.json').write_text(json.dumps(log,ensure_ascii=False,indent=2))
    return selected,log

def timetable(ins,Z):
    a=ins.accounting(Z); rows=[]; routes=[]; ident=0; place=np.zeros(ins.n,int)
    for r,z in enumerate(Z[-1]):
        if not z:continue
        path=ins.routes[r];cyclepath=list(path)+list(path[-2::-1]); cyc=int(ins.cycles[r]);dist=sum(ins.d[u,v] for u,v in zip(path,path[1:]))
        routes.append(dict(route=r,path=' — '.join(ins.df.iloc[i].settlement for i in path),drones=int(z),round_trips=cyc,one_way_km=dist))
        for _ in range(z):
            ident+=1; lanes={i:int(place[i]+1) for i in path}
            for i in path:place[i]+=1
            now=480
            for c in range(cyc):
                for u,v in zip(cyclepath,cyclepath[1:]):
                    fly=5*math.ceil(ins.d[u,v]/ins.p['speed_kmh']*60/5);arr=now+fly;end=arr+ins.p['service_minutes']
                    rows.append(dict(drone=ident,route=r,cycle=c+1,origin=ins.df.iloc[u].settlement,destination=ins.df.iloc[v].settlement,origin_index=u,destination_index=v,depart_min=now,arrive_min=arr,ready_min=end,arrival_place=lanes[v],distance_km=ins.d[u,v]))
                    now=end
            assert now<=1080
            assert cyclepath[0]==cyclepath[-1]
    assert max(place)<=ins.p['reserved_places']
    df=pd.DataFrame(rows)
    for col in ['depart','arrive','ready']:
        df[col]=df[col+'_min'].map(lambda x:f'{x//60:02d}:{x%60:02d}')
    df.to_csv(ROOT/'results/timetable_year3.csv',index=False)
    pd.DataFrame(routes).to_csv(ROOT/'results/routes_year3.csv',index=False)
    return dict(drones=ident,flights=len(df),max_distance=float(df.distance_km.max()),last_ready=df.ready.iloc[-1],reserved_places_max=int(max(place)),all_return_home=True)

def routing_benchmark(ins):
    adj={i:[j for j in range(ins.n) if i!=j and ins.d[i,j]<=ins.p['range_km']] for i in range(ins.n)}
    rows=[]
    for s in range(ins.n):
        for goal in range(ins.n):
            if s==goal:continue
            t=time.perf_counter(); queue=deque([(s,[s])]);seen={s};exp=0;bp=[]
            while queue:
                u,path=queue.popleft();exp+=1
                if u==goal:bp=path;break
                for v in adj[u]:
                    if v not in seen:seen.add(v);queue.append((v,path+[v]))
            bt=time.perf_counter()-t
            t=time.perf_counter();heap=[(ins.d[s,goal]/3,0,s,[s])];best={s:0};ap=[];ae=0
            while heap:
                _,cost,u,path=heapq.heappop(heap)
                if cost>best.get(u,np.inf):continue
                ae+=1
                if u==goal:ap=path;break
                for v in adj[u]:
                    g=cost+5*math.ceil(ins.d[u,v]/3/5)+35
                    if g<best.get(v,np.inf):best[v]=g;heapq.heappush(heap,(g+ins.d[v,goal]/3,g,v,path+[v]))
            at=time.perf_counter()-t
            def minutes(path):return sum(5*math.ceil(ins.d[u,v]/3/5)+35 for u,v in zip(path,path[1:]))
            rows.append(dict(origin=s,destination=goal,bfs_legs=max(0,len(bp)-1),astar_legs=max(0,len(ap)-1),bfs_minutes=minutes(bp),astar_minutes=minutes(ap),bfs_nodes=exp,astar_nodes=ae,bfs_seconds=bt,astar_seconds=at))
    result=pd.DataFrame(rows);result.to_csv(ROOT/'results/bfs_astar.csv',index=False)
    return dict(bfs_mean_nodes=float(result.bfs_nodes.mean()),astar_mean_nodes=float(result.astar_nodes.mean()),time_improved_pairs=int((result.bfs_minutes>result.astar_minutes).sum()),pairs=len(result))

def serial(obj):
    if isinstance(obj,np.ndarray):return obj.tolist()
    if isinstance(obj,(np.integer,np.floating)):return obj.item()
    raise TypeError(type(obj).__name__)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--budget',type=float,default=1000);parser.add_argument('--range',type=float,default=200);parser.add_argument('--quick',action='store_true');args=parser.parse_args()
    df,log=clean_and_select();ins=Instance(df,{'range_km':args.range});budget=args.budget
    out=ROOT/'results';out.mkdir(exist_ok=True)
    print('DATA',log,'routes',ins.R,flush=True)
    mainmilp=ins.solve(budget=budget,limit=30 if not args.quick else 5);print('MILP',mainmilp['objective'],mainmilp['gap'],flush=True)
    greedy=ins.greedy(budget);print('GREEDY',greedy['objective'],flush=True)
    trials=[]
    for seed in range(10):
        initial=greedy['Z'] if seed==0 else ins.greedy(budget,seed=seed)['Z']
        result=ins.anneal(initial,seed,budget,steps=100 if not args.quick else 20);trials.append(result);print('SA',seed,result['objective'],flush=True)
    best=max([mainmilp,greedy]+trials,key=lambda a:a['objective']);Z=best['Z'];assert ins.feasible(Z,budget)
    acc=ins.accounting(Z);stats=[]
    for t in range(3):
        score,flow=ins.allocate(Z[t],t,ret=True);daily=ins.od@flow
        flows=[]
        for v,f in enumerate(flow):
            if f>1e-7:
                r,o,d,*_=ins.tripvars[v];flows.append(dict(route=r,origin=ins.df.iloc[o].settlement,destination=ins.df.iloc[d].settlement,daily_trips=f,annual_trips=f*300))
        pd.DataFrame(flows).to_csv(out/f'passengers_year{t+1}.csv',index=False)
        stats.append(dict(year=t+1,ports=int(acc['x'][t].sum()),fleet=int(acc['N'][t]),purchases=int(acc['buy'][t]),trips=score*300,demand=ins.Q[t].sum()*300,share=score/ins.Q[t].sum(),capex=acc['cap'][t],opex=acc['op'][t],cost=acc['cost'][t],cum_cost=acc['cost'][:t+1].sum()))
    pd.DataFrame(stats).to_csv(out/'annual_plan.csv',index=False)
    pd.DataFrame([dict(method='MILP',objective=mainmilp['objective'],seconds=mainmilp['seconds'],upper_bound=mainmilp['upper_bound'],gap=mainmilp['gap'])]+[dict(method='Greedy',objective=greedy['objective'],seconds=greedy['seconds'])]+[dict(method='SA',seed=r['seed'],objective=r['objective'],seconds=r['seconds']) for r in trials]).to_csv(out/'algorithm_comparison.csv',index=False)
    print('SCHEDULE',timetable(ins,Z),flush=True)
    small_df=df.sort_values(['population','id'],ascending=[False,True]).head(8).reset_index(drop=True);small=Instance(small_df)
    sm=small.solve(limit=60 if not args.quick else 10,budget=budget);sg=small.greedy(budget);ss=small.anneal(sg['Z'],0,budget,100)
    print('SMALL',sm['objective'],sm['gap'],flush=True)
    staged=[];stage_details=[]
    for t in range(3):
        res=small.solve(budget=budget,limit=15,years=t+1,fixed_previous=np.array(staged) if staged else None);staged.append(res['Z'][-1]);stage_details.append({k:v for k,v in res.items() if k!='Z'})
    staged_score=small.score(staged)
    stress=[]
    for title,factor,rng in [('Базовый',1,200),('Спрос -30%',.7,200),('Спрос +30%',1.3,200),('Дальность 160 км',1,160)]:
        val=sum(ins.allocate(z,t,factor,rng) for t,z in enumerate(Z))*300
        stress.append(dict(scenario=title,trips=val,cost=acc['cost'].sum()))
    failures=[]
    for node in np.flatnonzero(acc['x'][-1]):
        val=ins.allocate(Z[-1],2,failed=int(node))*300
        failures.append(dict(node=int(node),name=ins.df.iloc[node].settlement,trips=val))
    worst=min(failures,key=lambda a:a['trips']);stress.append(dict(scenario='Отказ '+worst['name']+' на весь год 3',trips=stats[0]['trips']+stats[1]['trips']+worst['trips'],cost=acc['cost'].sum()))
    pd.DataFrame(stress).to_csv(out/'stress_tests.csv',index=False);pd.DataFrame(failures).to_csv(out/'node_failures_year3.csv',index=False)
    print('STRESS',stress,flush=True)
    budget_rows=[]
    for b in [400,600,800,1000,1200]:
        r=ins.greedy(b);budget_rows.append(dict(annual_budget_million=b,trips=r['objective'],cost=ins.accounting(r['Z'])['cost'].sum()))
    pd.DataFrame(budget_rows).to_csv(out/'budget_experiment.csv',index=False)
    scale=[]
    for size in [8,15,30]:
        sub=Instance(df.sort_values(['population','id'],ascending=[False,True]).head(size));r=sub.solve(limit=5,budget=budget)
        scale.append(dict(nodes=size,routes=sub.R,seconds=r['seconds'],trips=r['objective'],gap=r['gap'],upper_bound=r['upper_bound']))
    pd.DataFrame(scale).to_csv(out/'scale_experiment.csv',index=False)
    # Фиксированные искусственные пропуски на неизменном наборе узлов.
    corrupted=df.copy();mask=np.random.default_rng(42).choice(len(df),size=6,replace=False);corrupted.loc[mask,'population']=np.nan
    quality=[]
    for method in ['regional_median','nearest3_median']:
        im=corrupted.copy()
        for i in mask:
            known=corrupted.population.notna().to_numpy()
            if method=='regional_median':pop=corrupted.loc[known & (corrupted.region==df.iloc[i].region),'population'].median()
            else:
                ids=np.flatnonzero(known);near=ids[np.argsort(ins.d[i,ids])[:3]];pop=corrupted.iloc[near].population.median()
            im.loc[i,'population']=pop
        ii=Instance(im);r=ii.greedy(budget); aa=ii.accounting(r['Z']);built=set(np.flatnonzero(aa['x'][-1]));base=set(np.flatnonzero(ins.accounting(greedy['Z'])['x'][-1]))
        forecast=r['objective'];ii.Q=ins.Q.copy()
        quality.append(dict(method=method,trips=r['objective'],common_demand_trips=ii.score(r['Z']),ports=len(built),different_ports=len(built.symmetric_difference(base)),population_mae=float(abs(im.loc[mask,'population']-df.loc[mask,'population']).mean())))
        im.to_csv(out/f'imputed_{method}.csv',index=False)
    corrupted.to_csv(out/'artificial_missingness.csv',index=False);pd.DataFrame(quality).to_csv(out/'data_quality_experiment.csv',index=False)
    # Робастный выбор среди конечного набора действительно рассчитанных планов.
    candidates=[greedy]+trials+[mainmilp];robust=[]
    for k,r in enumerate(candidates):
        zz=r['Z'];base=ins.score(zz);vals=[base,ins.score(zz)*0+sum(ins.allocate(z,t,.7) for t,z in enumerate(zz))*300,sum(ins.allocate(z,t,1,160) for t,z in enumerate(zz))*300]
        act=np.flatnonzero(ins.accounting(zz)['x'][-1]);vals += [sum(ins.allocate(zz[t],t) for t in (0,1))*300+ins.allocate(zz[-1],2,failed=int(i))*300 for i in act]
        robust.append(dict(candidate=k,base=base,worst=min(vals),cost=ins.accounting(zz)['cost'].sum()))
    rb=max(robust,key=lambda r:r['worst']);pd.DataFrame(robust).to_csv(out/'robust_candidates.csv',index=False)
    routing=routing_benchmark(ins)
    # Конечная игра на малой задаче. Последний уровень - до 4 вариантов
    # переназначения одного освободившегося дрона без роста лётных расходов.
    small_candidates=[sg,ss,sm];payoffs=[]
    for ci,r in enumerate(small_candidates):
        zz=r['Z'][-1];aa=small.accounting(r['Z']);actions=[]
        for node in np.flatnonzero(aa['x'][-1]):
            z0=zz.copy();blocked=[j for j,p in enumerate(small.routes) if node in p];z0[blocked]=0
            policies=[z0];spare=int(zz.sum()-z0.sum())
            if spare:
                for j,p in enumerate(small.routes):
                    if node in p or not all(aa['x'][-1,i] for i in p):continue
                    w=z0.copy();w[j]+=1
                    if max(w@small.inc.T)<=3 and w@small.flightcost<=zz@small.flightcost+1e-8:policies.append(w)
                    if len(policies)==4:break
            actions.append([small.allocate(w,2)*300 for w in policies])
        payoffs.append(actions)
    def minimax(tree,prune):
        leaves=0;nodes=0
        def run(v,depth,alpha=-np.inf,beta=np.inf):
            nonlocal leaves,nodes
            nodes+=1
            if not isinstance(v,list):leaves+=1;return v
            maximizing=depth%2==0;value=-np.inf if maximizing else np.inf
            for child in v:
                y=run(child,depth+1,alpha,beta)
                if maximizing:value=max(value,y);alpha=max(alpha,value)
                else:value=min(value,y);beta=min(beta,value)
                if prune and beta<=alpha:break
            return value
        start=time.perf_counter();value=run(tree,0)
        return dict(value=value,visited_nodes=nodes,leaves=leaves,seconds=time.perf_counter()-start)
    full=minimax(payoffs,False);ab=minimax(payoffs,True);assert abs(full['value']-ab['value'])<1e-5
    result=dict(parameters=ins.p,data=log,routes=ins.R,annual=stats,main_milp={k:v for k,v in mainmilp.items() if k!='Z'},greedy={k:v for k,v in greedy.items() if k!='Z'},best_total=ins.score(Z),small={k:v for k,v in sm.items() if k!='Z'},small_greedy=sg['objective'],small_sa=ss['objective'],sequential_small=staged_score,sequential_details=stage_details,stress=stress,worst_node=worst,robust_best=rb,robust_base_candidate=next(i for i,r in enumerate(candidates) if r is best),routing=routing,game=dict(full=full,alpha_beta=ab),schedule=timetable(ins,Z))
    (out/'summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,default=serial))
    (out/'solution.json').write_text(json.dumps(dict(Z=Z,routes=ins.routes),ensure_ascii=False,indent=2,default=serial))
    pd.DataFrame([dict(route=r,path=' — '.join(df.iloc[i].settlement for i in p),indices=json.dumps(p),cycles=int(ins.cycles[r]),flight_opex_million=ins.flightcost[r]) for r,p in enumerate(ins.routes)]).to_csv(out/'route_templates.csv',index=False)
    for t in range(3):pd.DataFrame(ins.Q[t],index=df.id,columns=df.id).to_csv(ROOT/f'data/processed/demand_year{t+1}.csv')
    (ROOT/'parameters.json').write_text(json.dumps(ins.p,ensure_ascii=False,indent=2))
    print('DONE',result['best_total'],flush=True)

if __name__=='__main__':main()
