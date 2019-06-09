# autograder
Automatic grade level complexity classifier


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

Take a look at `create_arpas_n5-all-data.ipynb` for examples of usage.