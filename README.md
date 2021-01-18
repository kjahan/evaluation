# evaluation
Recommendations, ranking, and search metrics


## Setup environment

```
$ conda create -n recom python=3.6
```

```
$ conda install -n recom pandas
```

```
$ conda install -n recom numpy
```

```
$ conda install -n recom pytest
```

```
pip install implicit==0.4.0
```

```
conda install -c conda-forge notebook -n recom
```

```
conda install nb_conda -n recom
```

```
conda install h5py -n recom
```

```
conda install -c conda-forge ipywidgets -n recom
```

```
jupyter nbextension enable --py widgetsnbextension
```

## Activate conda env:

```
$ conda activate recom
```
(On MacOS: `unset PYTHONPATH`)


## Deactivae conda env:

```
$ conda deactivate
```

## Run tests:

`pytest`

## ALS with MovieLens Demo:
To run ALS with Movie Lense data and get offline metrics (Precison@k & Recall@k), run:

`python src/implicit_demo.py`
