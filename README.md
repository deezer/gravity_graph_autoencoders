## Gravity-Inspired Graph Autoencoders for Directed Link Prediction

This repository provides Python code to reproduce experiments from the article [Gravity-Inspired Graph Autoencoders for Directed Link Prediction](https://arxiv.org/pdf/1905.09570.pdf) published in the proceedings of the 28th ACM International Conference on Information and Knowledge Management (CIKM 2019).

We release Tensorflow implementations of the following **four directed graph embedding models** from the paper:
 - *Gravity-Inspired Graph Autoencoders*
 - *Gravity-Inspired Graph Variational Autoencoders*
 - *Source-Target Graph Autoencoders*
 - *Source-Target Graph Variational Autoencoders*

together with standard *Graph Autoencoders (AE)* and *Graph Variational Autoencoders (VAE)* models from [Kipf and Welling (2016)](https://arxiv.org/pdf/1611.07308.pdf). 

We evaluate all six models on the **three directed link prediction tasks** introduced in section 4.1 of our paper:
- *General Directed Link Prediction*
- *Biased Negative Samples Directed Link Prediction*
- *Bidirectionality Prediction*

Our code builds upon Thomas Kipf's [original Tensorflow implementation](https://github.com/tkipf/gae) of standard Graph AE/VAE.

<br>
<p align="center">
  <img height="550" src="graph_visu_cora.png">
</p>

## Installation

```bash
python setup.py install
```

Requirements: tensorflow (1.x), networkx, numpy, scikit-learn, scipy


## Run Experiments

```bash
cd gravity_gae
python train.py --model=gcn_vae --dataset=cora --task=task_1
python train.py --model=gravity_gcn_vae --dataset=cora --task=task_1
```

The above commands will train a *Graph VAE (line 2)* and a *Gravity-Inspired Graph VAE (line 3)* on *Cora dataset* and will evaluate node embdeddings on *Task 1: General Directed Link Prediction*, with all parameters set to default values.

#### Complete list of parameters


| Parameter        | Type           | Description  | Default Value |
| :-------------: |:-------------:| :-------------------------------|:-------------: |
| `model`     | string | Name of the model, among:<br> - `gcn_ae`: Graph AE from Kipf and Welling (2016), with 2-layer<br> GCN encoder and inner product decoder<br> - `gcn_vae`: Graph VAE from Kipf and Welling (2016), with Gaussian <br> distributions, 2-layer GCN encoders and inner product decoder<br> - `source_target_gcn_ae`: Source-Target Graph AE, as introduced <br> in section 2.6 of paper, with 2-layer GCN encoder and asymmetric inner product decoder <br> - `source_target_gcn_vae`: Source-Target Graph VAE, as introduced <br> in section 2.6, with Gaussian distributions, 2-layer GCN encoders and asymmetric inner product<br> - `gravity_gcn_ae`: Gravity-Inspired Graph AE, as introduced in <br> section 3.3 of paper, with 2-layer GCN encoder and  gravity-inspired asymmetric decoder <br> - `gravity_gcn_vae`: Gravity-Inspired Graph VAE, as introduced in <br> section 3.4 of paper, with Gaussian distributions, 2-layer GCN encoders and gravity-inspired decoder| `gcn_ae` |
| `dataset`    | string      | Name of the dataset, among:<br> - `cora`: scientific publications citation network, from [LINQS](https://linqs.soe.ucsc.edu/data) <br> - `citeseer`: scientific publications citation network, from [LINQS](https://linqs.soe.ucsc.edu/data) <br> - `google`: hyperlink network from web pages, from [KONECT](http://konect.uni-koblenz.de/networks/) <br> <br> Note: you can specify any additional graph dataset, in *edgelist* format,<br> by editing `input_data.py`| `cora`|
| `task` | string |Name of the link prediction evaluation task, among: <br> - `task_1`: General Directed Link Prediction <br> - `task_2`: Biased Negative Samples Directed Link Prediction <br> - `task_3`: Bidirectionality Prediction| `task_1`|
| `dropout`| float | Dropout rate | `0.` |
| `epoch`| int | Number of epochs in model training | `200` |
| `features`| boolean | Include node features or not in GCN encoder | `False` |
| `lamb`| float | "Lambda" parameter from Gravity AE/VAE models as introduced in <br> section 3.5 of paper, to balance mass and proximity terms' | `1.` |
| `learning_rate`| float | Initial learning rate (with Adam optimizer) | `0.1` |
| `hidden`| int | Number of units in GCN encoder hidden layer | `64` |
| `dimension`| int | Dimension of GCN output. It is: <br> - equal to embedding dimension for standard AE/VAE <br> and Source-Target AE/VAE models <br> - equal to (embedding dimension - 1) for gravity-inspired AE/VAE <br> models, as the last dimension captures the "mass" parameter <br> <br> Dimension must be *even* for Source-Target AE/VAE model | `32` |
| `normalize`| boolean | For Gravity models: whether to normalize embedding vectors | `False`|
| `epsilon`| float | For Gravity models: add epsilon to L2 distances computations, for numerical stability | `0.01`|
| `nb_run`| integer | Number of model runs + tests | `1` |
| `prop_val`| float | Proportion of edges in validation set (for Task 1) | `5.` |
| `prop_test`| float | Proportion of edges in test set (for Tasks 1, 2) | `10.` |
| `validation`| boolean | Whether to report validation results  at each epoch (for Task 1) | `False` |
| `verbose`| boolean | Whether to print full comments details | `True` |

#### Models from the paper

**Cora** - Task 1

```Bash
python train.py --dataset=cora --model=gcn_vae --task=task_1 --epochs=200 --learning_rate=0.1 --hidden=64 --dimension=32 --nb_run=5
python train.py --dataset=cora --model=gcn_ae --task=task_1 --epochs=200 --learning_rate=0.1 --hidden=64 --dimension=32 --nb_run=5
python train.py --dataset=cora --model=source_target_gcn_vae --task=task_1 --epochs=200 --learning_rate=0.1 --hidden=64 --dimension=32 --nb_run=5
python train.py --dataset=cora --model=source_target_gcn_ae --task=task_1 --epochs=200 --learning_rate=0.1 --hidden=64 --dimension=32 --nb_run=5
python train.py --dataset=cora --model=gravity_gcn_vae --task=task_1 --epochs=200 --learning_rate=0.1 --hidden=64 --dimension=33 --lamb=1.0 --nb_run=5
python train.py --dataset=cora --model=gravity_gcn_ae --task=task_1 --epochs=200 --learning_rate=0.1 --hidden=64 --dimension=33 --lamb=1.0 --nb_run=5
```

**Cora** - Task 2

```Bash
python train.py --dataset=cora --model=gcn_vae --task=task_2 --epochs=200 --learning_rate=0.1 --hidden=64 --dimension=32 --nb_run=5
python train.py --dataset=cora --model=gcn_ae --task=task_2 --epochs=200 --learning_rate=0.1 --hidden=64 --dimension=32 --nb_run=5
python train.py --dataset=cora --model=source_target_gcn_vae --task=task_2 --epochs=200 --learning_rate=0.1 --hidden=64 --dimension=32 --nb_run=5
python train.py --dataset=cora --model=source_target_gcn_ae --task=task_2 --epochs=200 --learning_rate=0.1 --hidden=64 --dimension=64 --nb_run=5
python train.py --dataset=cora --model=gravity_gcn_vae --task=task_2 --epochs=200 --learning_rate=0.1 --hidden=64 --dimension=33 --lamb=0.05 --nb_run=5
python train.py --dataset=cora --model=gravity_gcn_ae --task=task_2 --epochs=200 --learning_rate=0.1 --hidden=64 --dimension=33 --lamb=0.05 --normalize=True --nb_run=5
```
**Cora** - Task 3

```Bash
python train.py --dataset=cora --model=gcn_vae --task=task_3 --epochs=200 --learning_rate=0.1 --hidden=64 --dimension=32 --nb_run=5
python train.py --dataset=cora --model=gcn_ae --task=task_3 --epochs=200 --learning_rate=0.1 --hidden=64 --dimension=32 --nb_run=5
python train.py --dataset=cora --model=source_target_gcn_vae --task=task_3 --epochs=200 --learning_rate=0.1 --hidden=64 --dimension=32 --nb_run=5
python train.py --dataset=cora --model=source_target_gcn_ae --task=task_3 --epochs=200 --learning_rate=0.1 --hidden=64 --dimension=32 --nb_run=5
python train.py --dataset=cora --model=gravity_gcn_vae --task=task_3 --epochs=200 --learning_rate=0.1 --hidden=64 --dimension=33 --lamb=1.0 --nb_run=5
python train.py --dataset=cora --model=gravity_gcn_ae --task=task_3 --epochs=200 --learning_rate=0.1 --hidden=64 --dimension=33 --lamb=1.0 --nb_run=5
```

**Citeseer** - Task 1

```Bash
python train.py --dataset=citeseer --model=gcn_vae --task=task_1 --epochs=200 --learning_rate=0.1 --hidden=64 --dimension=32 --nb_run=5
python train.py --dataset=citeseer --model=gcn_ae --task=task_1 --epochs=200 --learning_rate=0.1 --hidden=64 --dimension=32 --nb_run=5
python train.py --dataset=citeseer --model=source_target_gcn_vae --task=task_1 --epochs=200 --learning_rate=0.1 --hidden=64 --dimension=32 --nb_run=5
python train.py --dataset=citeseer --model=source_target_gcn_ae --task=task_1 --epochs=200 --learning_rate=0.1 --hidden=64 --dimension=32 --nb_run=5
python train.py --dataset=citeseer --model=gravity_gcn_vae --task=task_1 --epochs=200 --learning_rate=0.1 --hidden=64 --dimension=33 --lamb=1.0 --nb_run=5
python train.py --dataset=citeseer --model=gravity_gcn_ae --task=task_1 --epochs=200 --learning_rate=0.1 --hidden=64 --dimension=33 --lamb=1.0 --nb_run=5
```

**Citeseer** - Task 2

```Bash
python train.py --dataset=citeseer --model=gcn_vae --task=task_2 --epochs=200 --learning_rate=0.1 --hidden=64 --dimension=32 --nb_run=5
python train.py --dataset=citeseer --model=gcn_ae --task=task_2 --epochs=200 --learning_rate=0.1 --hidden=64 --dimension=32 --nb_run=5
python train.py --dataset=citeseer --model=source_target_gcn_vae --task=task_2 --epochs=200 --learning_rate=0.1 --hidden=64 --dimension=32 --nb_run=5
python train.py --dataset=citeseer --model=source_target_gcn_ae --task=task_2 --epochs=200 --learning_rate=0.1 --hidden=64 --dimension=32 --nb_run=5
python train.py --dataset=citeseer --model=gravity_gcn_vae --task=task_2 --epochs=200 --learning_rate=0.1 --hidden=64 --dimension=33 --lamb=0.05 --nb_run=5
python train.py --dataset=citeseer --model=gravity_gcn_ae --task=task_2 --epochs=200 --learning_rate=0.1 --hidden=64 --dimension=33 --lamb=0.05 --normalize=True --nb_run=5
```
**Citeseer** - Task 3

```Bash
python train.py --dataset=citeseer --model=gcn_vae --task=task_3 --epochs=200 --learning_rate=0.1 --hidden=64 --dimension=32 --nb_run=5
python train.py --dataset=citeseer --model=gcn_ae --task=task_3 --epochs=200 --learning_rate=0.1 --hidden=64 --dimension=32 --nb_run=5
python train.py --dataset=citeseer --model=source_target_gcn_vae --task=task_3 --epochs=200 --learning_rate=0.1 --hidden=64 --dimension=32 --nb_run=5
python train.py --dataset=citeseer --model=source_target_gcn_ae --task=task_3 --epochs=200 --learning_rate=0.1 --hidden=64 --dimension=32 --nb_run=5
python train.py --dataset=citeseer --model=gravity_gcn_vae --task=task_3 --epochs=200 --learning_rate=0.1 --hidden=64 --dimension=33 --lamb=1.0 --nb_run=5
python train.py --dataset=citeseer --model=gravity_gcn_ae --task=task_3 --epochs=200 --learning_rate=0.1 --hidden=64 --dimension=33 --lamb=1.0 --nb_run=5
```

**Google** - Task 1

```Bash
python train.py --dataset=google --model=gcn_vae --task=task_1 --epochs=200 --learning_rate=0.2 --hidden=64 --dimension=32 --nb_run=5
python train.py --dataset=google --model=gcn_ae --task=task_1 --epochs=200 --learning_rate=0.2 --hidden=64 --dimension=32 --nb_run=5
python train.py --dataset=google --model=source_target_gcn_vae --task=task_1 --epochs=200 --learning_rate=0.2 --hidden=64 --dimension=32 --nb_run=5
python train.py --dataset=google --model=source_target_gcn_ae --task=task_1 --epochs=200 --learning_rate=0.2 --hidden=64 --dimension=32 --nb_run=5
python train.py --dataset=google --model=gravity_gcn_vae --task=task_1 --epochs=200 --learning_rate=0.2 --hidden=64 --dimension=33 --lamb=10.0 --nb_run=5
python train.py --dataset=google --model=gravity_gcn_ae --task=task_1 --epochs=200 --learning_rate=0.2 --hidden=64 --dimension=33 --lamb=10.0 --nb_run=5
```

**Google** - Task 2

```Bash
python train.py --dataset=google --model=gcn_vae --task=task_2 --epochs=200 --learning_rate=0.2 --hidden=64 --dimension=32 --nb_run=5
python train.py --dataset=google --model=gcn_ae --task=task_2 --epochs=200 --learning_rate=0.2 --hidden=64 --dimension=32 --nb_run=5
python train.py --dataset=google --model=source_target_gcn_vae --task=task_2 --epochs=200 --learning_rate=0.2 --hidden=64 --dimension=32 --nb_run=5
python train.py --dataset=google --model=source_target_gcn_ae --task=task_2 --epochs=200 --learning_rate=0.2 --hidden=64 --dimension=32 --nb_run=5
python train.py --dataset=google --model=gravity_gcn_vae --task=task_2 --epochs=200 --learning_rate=0.2 --hidden=64 --dimension=33 --lamb=0.05 --nb_run=5
python train.py --dataset=google --model=gravity_gcn_ae --task=task_2 --epochs=200 --learning_rate=0.2 --hidden=64 --dimension=33 --lamb=0.05 --normalize=True --epsilon=1.0 --nb_run=5
```

**Google** - Task 3

```Bash
python train.py --dataset=google --model=gcn_vae --task=task_3 --epochs=200 --learning_rate=0.2 --hidden=64 --dimension=32 --nb_run=5
python train.py --dataset=google --model=gcn_ae --task=task_3 --epochs=200 --learning_rate=0.2 --hidden=64 --dimension=32 --nb_run=5
python train.py --dataset=google --model=source_target_gcn_vae --task=task_3 --epochs=200 --learning_rate=0.2 --hidden=64 --dimension=32 --nb_run=5
python train.py --dataset=google --model=source_target_gcn_ae --task=task_3 --epochs=200 --learning_rate=0.2 --hidden=64 --dimension=32 --nb_run=5
python train.py --dataset=google --model=gravity_gcn_vae --task=task_3 --epochs=200 --learning_rate=0.2 --hidden=64 --dimension=33 --lamb=10.0 --nb_run=5
python train.py --dataset=google --model=gravity_gcn_ae --task=task_3 --epochs=200 --learning_rate=0.2 --hidden=64 --dimension=33 --lamb=10.0 --nb_run=5
```

Notes:
 - Set `--nb_run=100` to report mean AUC and AP, along with standard errors, over 100 runs, as in the paper
 - We recommend GPU usage for faster learning

## Cite

Please cite our paper if you use this code in your own work:

```BibTeX
@inproceedings{salha2019gravity,
  title={Gravity-Inspired Graph Autoencoders for Directed Link Prediction},
  author={Salha, Guillaume and Limnios, Stratis and Hennequin, Romain and Tran, Viet Anh and Vazirgiannis, Michalis},
  booktitle={ACM International Conference on Information and Knowledge Management (CIKM)},
  year={2019}
}
```
