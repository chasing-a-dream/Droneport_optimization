"""Независимые проверки сохранённого плана и быстрые вопросы на защите."""
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
from model import Instance, ROOT

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--demand-scale',type=float,default=1)
    ap.add_argument('--range',type=float,default=200);ap.add_argument('--disable-node',type=int)
    ap.add_argument('--failure-year',type=int,choices=[1,2,3],default=3,help='Первый год отказа узла, по умолчанию 3')
    args=ap.parse_args()
    params=json.loads((ROOT/'parameters.json').read_text())
    ins=Instance(pd.read_csv(ROOT/'data/processed/selected_30.csv'),params)
    saved=json.loads((ROOT/'results/solution.json').read_text());Z=np.array(saved['Z'],int)
    assert ins.feasible(Z), 'Нарушен бюджет или резервирование мест'
    years=[]
    for t,z in enumerate(Z):
        score,f=ins.allocate(z,t,ret=True)
        assert np.all(ins.od@f<=ins.Q[t].ravel()+1e-6)
        for r,vs in ins.segments:assert f[vs].sum()<=10*ins.cycles[r]*z[r]+1e-6
        failed=args.disable_node if t>=args.failure_year-1 else None
        actual=ins.allocate(z,t,args.demand_scale,args.range,failed)
        years.append(actual*300)
    tt=pd.read_csv(ROOT/'results/timetable_year3.csv')
    for drone,flights in tt.groupby('drone'):
        flights=flights.reset_index(drop=True)
        assert flights.origin_index.iloc[0]==flights.destination_index.iloc[-1]
        assert flights.ready_min.max()<=1080
        assert (flights.distance_km<=params['range_km']+1e-7).all()
        for k in range(1,len(flights)):
            assert flights.depart_min.iloc[k]>=flights.ready_min.iloc[k-1]
            assert flights.origin_index.iloc[k]==flights.destination_index.iloc[k-1]
    # Места закреплены за дронами, поэтому одновременное обслуживание
    # проверяется по идентификатору места, независимо от суммарных часов.
    for (port,lane),group in tt.groupby(['destination_index','arrival_place']):
        intervals=sorted(zip(group.arrive_min,group.ready_min))
        assert all(b<=c for (a,b),(c,d) in zip(intervals,intervals[1:]))
    annual=pd.read_csv(ROOT/'results/annual_plan.csv')
    assert np.allclose(annual.cost.cumsum(),annual.cum_cost)
    assert np.all(annual.cum_cost<=np.arange(1,4)*1000+1e-6)
    print('Проверки спроса, мест, расположения флота, расписания и бюджета пройдены.')
    print('Сценарий: поездки по годам', [round(a,1) for a in years])
    print('Сумма за три года:',round(sum(years),1))

if __name__=='__main__':main()
