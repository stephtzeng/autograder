# autograder
Automatic grade level complexity classifier

### Kneser-Ney Smoothing
To use `article_process.py`, you will need to install from: https://github.com/kpu/kenlm.

To instantiate the object, you will need the paths to:
- the folder with the text data
- the path to where you installed kenlm via git
- the subpath to the language models:
```python
path_to_data = 'path/to/data'
path_to_kenlm = 'path/to/github/kenlm'
path_to_arpa = path_to_kenlm + '/lm'
```

Usage:

```python
from article_lm import ArticleLM
articleLM = ArticleLM(path_to_data,
                      path_to_kenlm,
                      path_to_arpa, 5, 'grade_level', False)
```
where 5 denotes 5-gram, and you are creating grade level models, and original articles are not used.