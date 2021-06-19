# PolEval punctuation restoral challenge

Solution based on BERT using hugginface model Polbert

https://github.com/kldarek/polbert

Original transformer was a fill-in language model. Fine tuned to become token classifier.
This model tries to guess one of the following 9 punctuation symbols `None`, `.`, `?`, `!`, `:`, `;`, `,`, `-`, `...`. It achieves F-score of


|  predicted \\ actual     | `None` |  `.`  |  `?`  |   `!`  |  `:`  | `;`   | `,`    |   `-` |  `...`   |              
| ------ | ------ | ----- | ----- | ----- | ---- | ---- | ----- | ---- | ----- |
| `None` | 26936  |   218 |    45 |    15 |   61 |    7 |   663 |  162 |    12 |
|   `.`  |   94   |  1936 |    20 |    12 |   29 |    0 |    18 |    7 |    10 |
|  `?`   |    20  |    41 |  113  |     3 |    2 |    0 |     5 |    0 |    4  |
|  `!`   |    1   |     2 |     0 |     1 |    0 |    0 |     1 |    0 |    1  |
|  `:`   |   19   |    21 |    3  |     1 |   94 |    1 |     6 |    2 |    0  |  
|  `;`   |    0   |     0 |     0 |     0 |    0 |    0 |     0 |    0 |    0  |
|  `,`   |  252   |   21  |   3   |     0 |   11 |    5 |  1340 |   68 |    3  |
|  `-`   |   47   |    22 |    3  |     1 |    5 |    1 |    51 |  204 |    2  |
|  `...` |    1   |     2 |    0  |     0 |    0 |    0 |     0 |    0 |    2  |


F-score:

- `None`  0.9708590507507324
- `.`  0.8822054266929626
- `?` 0.6026666760444641
- `!` 0.05128205567598343
- `:` 0.5386819839477539
- `;` nan (not guessed at all)
- `,` 0.7076841592788696
- `-` 0.5237483978271484
- `...` 0.10256410390138626


Geval evaluation

```
$ ./geval -t train
0.91
```
