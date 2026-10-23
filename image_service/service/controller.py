"""CPU comparator: same supplied search rule, experience-only memory updates."""
class CPUController:
    def __init__(self,twohop=True):self.mapping=[None]*8;self.history=[None]*4;self.twohop=twohop;self.learn_transition=True
    def choose(self,state,probe=False):
        direct=[self.mapping[2*state+a]==0 for a in range(2)]
        predecessors=[s for s in range(4) if any(self.mapping[2*s+a]==0 for a in range(2))]
        hop=[self.mapping[2*state+a] in predecessors for a in range(2)]
        if probe:action=0 if self.history[state] is not None and self.history[state][0]==1 else 1;depth=0
        elif any(direct):action=direct.index(True);depth=1
        elif self.twohop and any(hop):action=hop.index(True);depth=2
        else:action=0 if self.history[state] is not None and self.history[state][0]==1 else 1;depth=0
        return dict(state=state,action=action,depth=depth,predecessor_set=predecessors,predicted_next_state=self.mapping[2*state+action])
    def observe(self,state,action,outcome):
        if self.learn_transition:self.mapping[2*state+action]=outcome
        self.history[state]=[action,outcome]
