# Fresh-seed validation rejects a consistent native-policy improvement

CPU forecasts only; no native learning run was admitted. The candidate was frozen
at match31 / margin4 before forecasting fresh seeds98411–98414. This validation
explicitly targeted>=5% fewer total adaptation jobs (queries + teaching), rather
than claiming the original>=25% teaching-job target had passed. Accuracy gates
and the original delayed-feedback task were retained.

| Seed | Candidate teaching jobs | Candidate old / new correct | Stock old / new correct | Gate |
|---|---:|---:|---:|---|
|98411|176|113 /124|115 /123|Pass|
|98412|188|121 /119|118 /126|Fail: new accuracy loses7, allowed2|
|98413|186|110 /116|113 /115|Fail: old accuracy loses3, allowed2|
|98414|174|124 /123|125 /121|Pass|

The cost proxy improved in all four seeds; quality was not consistently
noninferior. Two successes do not override two failed gates. Native expansion
stays closed, with no replacement seeds or further policy search. This finding
identifies a real retention/adaptation tradeoff in the proposed controller, not
a physical failure of the chip or a native-algorithm improvement.
