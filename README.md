# Differentially Private link prediction with protected connections

Note that re-running the scripts will generate new training test splits and new instances of different randomizations. Therefore, successive runs may give slightly different numbers.
However, we uploaded all intermediate files which would lead to the numbers reported in the paper, at URL https://rebrand.ly/dplp 
Due to the upper limit on upload size, we are not uploading those intermediate pickle files with this material.

## Pre-requisites

This code depends on the following packages:

 1. `numpy`
 2. `scipy`
 3. `matplotlib`
 4. `torch`
 5. `Keras (for preprocessing embeddings of deep methods)`
 6. `Tensorflow (for preprocessing embeddings of deep methods)`
 7. `Networkx`

Use `requirements.txt` to prepare a suitable pip environment.

## Code structure (classes and methods without execution harness)

 - `graph_process.py` contains the implementation of pre processing of the graph--- splitting in training, test and validation set, marking the protected edges.
 - `DPLP_plus.py` contains implementation of preparing data-structures containing scores and noises, so that the subsequent training algorithm can easily process it.
- `neural_modules.py` contains DPLP (dplp_mnn), DPLP-Lin (dplp_lin), DPLP-UMNN (dplp_mnn_no_pre_trained); torch driven AUC computation for different circumstances (e.g. probing during training, debugging, final evaluation, etc.); pairwise ranking loss implementations
 - `optim.py` trains DPLP-LIN (def train() module), DPLP (def train_umnn() module)
 - `all_embeddings.py` creates embeddings for All Deep methods.
 - `baselines.py` evaluates all methods--- DPLPs and other state-of-the-art methods

## Code structure (entry points for executable scripts)
 - `runfile_process_graph.py` is a *script* that processes graphs, implements base LP algorithm that return base scores, prepares data structures containing scores and noises, so that the subsequent training algorithm for several variations of protected edges and triad based algorithms can easily process it. Essentially, it is wrapper calling different modules for `graph_process.py` and `DPLP_plus.py`.

 - `train_runfile.py` is the training script for only triad-based protocols.
 - `deep_graph_score_prepare.py` is the script computing embeddings for different deep methods.
 - `emb_train_runfile.py` is the training script for only embeddings-based protocols.

## Execution

`python3 runfile_process_graph.py --dataset [dataset_name] --for_score --gen_graph --dplp_prepare`
`python3 train_runfile.py  --dataset [dataset_name] --device [dev]`   ## trains DPLP-Lin for AA, CN, JC, PA
`python3 train_runfile.py  --dataset [dataset_name] --device [dev] --umnn` ## trains DPLP for AA, CN, JC, PA

`python3 deep_graph_score_prepare.py --dataset [dataset_name] --score_matrix_deep --for_score --dplp_prepare --device [dev]` ## prepares score matrix
`python3 emb_train_runfile.py  --dataset [dataset_name] --umnn --device [dev]` ## trains DPLP-Lin for embedding based methods
`python3 emb_train_runfile.py  --dataset [dataset_name] --umnn --device [dev] --umnn` ## trains DPLP for embedding based methods
`python3 util_variation.py  --dataset [dataset_name]  ## prepares results for Figure 2 in the paper.
dev is "cuda" if GPU is preferred, otherwise "cpu".

## Evaluation

See ResultNotebook.ipynb for details. 
