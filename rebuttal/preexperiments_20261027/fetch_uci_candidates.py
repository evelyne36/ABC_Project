"""Fetch official UCI archives and normalize declared targets without outcome selection."""
from pathlib import Path
import requests,zipfile,io,json,hashlib,pandas as pd
ROOT=Path(__file__).resolve().parents[2]/'local_sync/new_tabular_data'
ROOT.mkdir(parents=True,exist_ok=True)
SPECS=[(243,'yacht+hydrodynamics'),(294,'combined+cycle+power+plant'),(477,'real+estate+valuation+data+set'),(182,'concrete+slump+test'),(87,'servo'),(9,'auto+mpg'),(29,'computer+hardware')]
rows=[]
for ident,slug in SPECS:
 url=f'https://archive.ics.uci.edu/static/public/{ident}/{slug.replace("+","%2B")}.zip'
 archive=ROOT/f'uci_{ident}.zip'
 if not archive.exists():
  r=requests.get(url,timeout=60);r.raise_for_status();zipfile.ZipFile(io.BytesIO(r.content));archive.write_bytes(r.content)
 z=zipfile.ZipFile(archive)
 def file(suffix):return io.BytesIO(z.read(next(n for n in z.namelist() if n.endswith(suffix))))
 tasks=[]
 if ident==243:
  df=pd.read_csv(file('yacht_hydrodynamics.data'),sep=r'\s+',header=None);df.columns=['buoyancy','prismatic','length_displacement','beam_draught','length_beam','froude','resistance'];tasks=[('yacht',df,'measured','six geometry/velocity inputs')]
 elif ident==294:tasks=[('ccpp',pd.read_excel(file('Folds5x2_pp.xlsx'),sheet_name=0),'measured','first sheet only; other sheets are shuffled copies')]
 elif ident==477:tasks=[('real_estate',pd.read_excel(file('.xlsx')).drop(columns='No'),'observational','exclude No ID')]
 elif ident==182:
  df=pd.read_csv(file('slump_test.data'))
  for i,name in enumerate(['slump','flow','strength']):tasks.append(('concrete_'+name,pd.concat([df.iloc[:,1:8],df.iloc[:,8+i]],axis=1),'measured','seven ingredients only; exclude ID and both other outputs'))
 elif ident==87:tasks=[('servo',pd.read_csv(file('servo.data'),header=None,names=['motor','screw','pgain','vgain','rise_time']),'simulation','motor/screw onehot; gains numeric')]
 elif ident==9:
  df=pd.read_csv(file('auto-mpg.data'),sep=r'\s+',header=None,na_values='?');df.columns=['mpg','cylinders','displacement','horsepower','weight','acceleration','model_year','origin','car_name']
  tasks=[('auto_mpg',df[['cylinders','displacement','horsepower','weight','acceleration','model_year','origin','mpg']],'observational','drop car_name ID; seven original numeric inputs; missing horsepower train-imputed')]
 else:
  df=pd.read_csv(file('machine.data'),header=None,names=['vendor','model','MYCT','MMIN','MMAX','CACH','CHMIN','CHMAX','PRP','ERP'])
  tasks=[('computer_hardware',df[['MYCT','MMIN','MMAX','CACH','CHMIN','CHMAX','PRP']],'measured','exclude target-derived ERP, model ID and vendor; six numeric hardware predictors')]
 for name,df,kind,note in tasks:
  path=ROOT/(name+'.csv');df.to_csv(path,index=False)
  rows.append(dict(dataset=name,path=str(path),uci_id=ident,source=f'https://archive.ics.uci.edu/dataset/{ident}/{slug}',archive_url=url,archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),rows=len(df),features=len(df.columns)-1,provenance=kind,preparation=note,duplicate_X=int(df.iloc[:,:-1].duplicated().sum()),duplicate_rows=int(df.duplicated().sum())))
  print(name,df.shape,flush=True)
(ROOT/'manifest.json').write_text(json.dumps(rows,indent=2)+'\n')
