"""Explicit operator invocation creates a short, bounded local execution envelope."""
from pathlib import Path
import argparse,datetime,json,re
def main():
    p=argparse.ArgumentParser();p.add_argument('--name',required=True);p.add_argument('--minutes',type=int,default=30);p.add_argument('--sessions',type=int,default=12);a=p.parse_args()
    assert re.fullmatch(r'[a-zA-Z0-9_-]+',a.name) and 3<=a.minutes<=180 and 1<=a.sessions<=24
    root=Path(__file__).resolve().parents[1];study=root/'studies'/a.name;study.mkdir(parents=True,exist_ok=False)
    for d in ['evidence','raw']:(study/d).mkdir()
    now=datetime.datetime.now(datetime.timezone.utc);end=now+datetime.timedelta(minutes=a.minutes)
    auth=dict(source='Explicit local operator invocation of new_study.py',maximum_physical_sessions=a.sessions,maximum_forwards=150000,maximum_individual_seconds=180,scientific_cutoff_utc=(end-datetime.timedelta(minutes=2)).isoformat(),hard_stop_utc=end.isoformat())
    (study/'AUTHORITY.json').write_text(json.dumps(auth,indent=2)+'\n')
    print(f'export AKD_LAB_STUDY=studies/{a.name}')
if __name__=='__main__':main()
