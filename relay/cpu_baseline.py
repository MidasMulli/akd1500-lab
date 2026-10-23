"""Small task-level CPU comparator; separate from the exact native-rule emulator."""
import json
from pathlib import Path
import numpy as np
from relay.design import schedule,metrics


def evaluate():
    x,rows=schedule('recognition_repair');memory=[[] for _ in range(4)];routing=[0]*4;outputs=[]
    for values,row in zip(x,rows):
        bits=sum(1<<int(k) for k in np.flatnonzero(values))
        candidates=[((bits^p).bit_count(),j) for j,bank in enumerate(memory) for p in bank]
        identity=min(candidates)[1] if candidates else 0
        result=[0]*4;result[routing[identity]]=1;outputs.append(result)
        if row['learning_mask']==1:
            bank=memory[row['label']]
            # Two exemplars/class, acquired from the same teaching stream.
            # No domain flag, signature generator or future outcomes are visible.
            if not bank or min((bits^p).bit_count() for p in bank)>64:
                if len(bank)<2:bank.append(bits)
        elif row['learning_mask']==2:routing[identity]=row['label']
    result=dict(status='CPU_TASK_BASELINE',algorithm='Two binary exemplars per identity plus learned destination table',
                metrics=metrics(rows,outputs),exemplars_per_class=[len(b) for b in memory],
                design_status='Post hoc diagnostic comparator, authored after inspecting the first native case; not a preregistered qualification',
                scope='Functional software comparator; no CPU or electrical timing measurement')
    return result


def main():
    result=evaluate()
    p=Path(__file__).resolve().parents[1]/'results/2026-10-23-signal-relay-cpu.json'
    p.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))


if __name__=='__main__':main()
