Dataset from https://github.com/applicaai/kleister-nda

Python script generating results is in [BERT.py](./BERT.py)



dev-0 set

```
$ ./geval -t dev-0
	F1	P	R
(UC)	0.568±0.049	0.575±0.050	0.565±0.047
date	0.830±0.098	0.856±0.091	0.793±0.099
party	0.492±0.081	0.485±0.085	0.497±0.080
juris	0.57±0.11	0.57±0.11	0.57±0.11
term	0.51±0.18	0.54±0.18	0.50±0.17

F1	0.544±0.046
Accuracy	0.108±0.060
Mean-F1	0.561±0.054
```

train set

```
$ ./geval -t train
	F1	P	R
(UC)	0.576±0.027	0.576±0.025	0.577±0.030
date	0.873±0.044	0.915±0.042	0.838±0.048
party	0.462±0.046	0.454±0.043	0.470±0.055
juris	0.649±0.048	0.654±0.047	0.646±0.047
term	0.397±0.099	0.40±0.10	0.388±0.099

F1	0.552±0.028
Accuracy	0.069±0.030
Mean-F1	0.578±0.027
```