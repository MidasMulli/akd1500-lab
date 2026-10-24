# Conservative repair preserved development quality but missed its cost gate

CPU-only; zero physical sessions. This repair retained the prior failed study
and prospectively compared exactly two conservative policies on seed98401. It
kept the corpus, delayed-feedback contract, resources and quality gates fixed.

| Policy | Adaptation teaching jobs | Final old | Final new | Disposition |
|---|---:|---:|---:|---|
|Stock|256|108/128|126/128|Comparator|
|Match31 / margin4|197|114/128|127/128|Quality passes;23.05% fewer teaching jobs misses25% target|
|Match32 / margin4|215|112/128|125/128|Quality passes;16.02% fewer teaching jobs misses25% target|

Neither was selected under this protocol. Its fresh-seed confirmation remained
unrun. The narrower31 policy was then frozen as an exploratory candidate in a
separately named out-of-sample validation with a different operational objective;
that does not turn this failed gate into a pass.
