# Supplementary Github repository for experiment 2 in our paper: On the Use of Steady-State Detection for Process Mining: Achieving More Accurate Insights
In this repository, we explain how we train three different models: Dummy, DALSTM, and PGTNet for remaining time prediction based on steady-state detection results. This repository is a forked version of [PGTNet repository](https://github.com/keyvan-amiri/PGTNet) with minor changes. For more details, you can refer to the original repository.

We applied all three models on 9 publicly available event logs. Each time we train the models first for all traces in the event log, and then for only traces that are included in steady-state periods. 

**<a name="part1">1. Set up a Python environement to work with GPS Graph Transformers:</a>**

GPS Graph Transformers recipe is implemented based on the PyTorch machine learning framework, and it utlizes [PyG](https://pytorch-geometric.readthedocs.io/en/latest/) (PyTorch Geometric) library. In order to be able to work with GPS Graph Transformers, you need to set up a Python environement with Conda as suggested [here](https://github.com/rampasek/GraphGPS#python-environment-setup-with-conda). To set up such an environement:
```
conda create -n graphgps python=3.10
conda activate graphgps

conda install pytorch=1.13 torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
conda install pyg=2.2 -c pyg -c conda-forge
pip install pyg-lib -f https://data.pyg.org/whl/torch-1.13.0+cu117.html

pip install pytorch-lightning yacs torchmetrics
pip install performer-pytorch
pip install tensorboardX
pip install ogb
pip install wandb

pip install pm4py
#pip install PyYAML #Requirement already satisfied
#pip install numpy  #Requirement already satisfied
#pip install scikit-learn #Requirement already satisfied

conda clean --all
```
Note that, we also included pip install for [pm4py](https://pm4py.fit.fraunhofer.de/) library to facilitate working with event log data. 

**<a name="part2">2. Clone repositories and download event logs:</a>**

Once you setup a conda environement, clone the [GPS Graph Transformer repository](https://github.com/rampasek/GraphGPS) using the following command:
```
git clone https://github.com/rampasek/GraphGPS
```
This repository is called **GPS repository** in the remaining of this README file. Now, Navigate to the root directory of **GPS repository**, and clone the current repository (i.e., the **PGTNet repository**). By doing so, the **PGTNet repository** will be placed in the root directory of **GPS repository** meaning that the latter is the parent directory of the former.
```
cd GraphGPS
git clone https://github.com/keyvan-amiri/PGTNet
```
Now, we are ready to download all event logs that are used in our experiments. In priniciple, downloading event logs and converting them to graph datasets are not mandatory steps for training PGTNet because we already uploaded the resultant graph datasets [here](https://github.com/keyvan-amiri/PGTNet/tree/main/transformation). In case you want to start with [training](https://github.com/keyvan-amiri/PGTNet#part4) a PGTNet, you can skip this step and the next one. In this case, generated graph dataset are automatically downloaded and will be used to train and evaluate PGTNet for remaining time prediction.

To download all event logs, navigate to the root directory of **PGTNet repository** and run `data-acquisition.py` script:
```
cd PGTNet
python data-acquisition.py
```
All event logs utilized in our experiments are publicly available at [the 4TU Research Data repository](https://data.4tu.nl/categories/13500?categories=13503). The **data-acquisition.py** script download all event logs, and convert them into .xes format. It also generates additional event logs (BPIC12C, BPIC12W, BPIC12CW, BPIC12A, BPIC12O) from BPIC12 event log.  

**<a name="part3">3. Converting an event log into a graph dataset:</a>**

In order to convert an event log into its corresponding graph dataset, you need to run the same python script with specific arguments:
```
python GTconvertor.py conversion_configs bpic15m1.yaml --overwrite true
```
The first argument (i.e., conversion_configs) is a name of directory in which all required configuration files are located. The second argument (i.e., bpic15m1.yaml) is the name of configuration file that defines parameters used for converting the event log into its corresponding graph dataset. All conversion configuration files used in our experiment are collected [here](https://github.com/keyvan-amiri/PGTNet/tree/main/conversion_configs). The last argument called overwrite is a Boolean variable which provides some flexibility. If it is set to false, and you have already converted the event log into its corresponding graph dataset the script simply skip repeating the task. 

**Graph dataset structure:** The resultant graph dataset will be saved in a seperate folder which is located in the **datasets** folder in the root directory for **GPS repository**. Each graph dataset is a [PyG data object](https://pytorch-geometric.readthedocs.io/en/latest/modules/data.html) and represents a set of event prefixes. For each graph dataset, three separate files are generated for the training, validation, and test sets. 

**<a name="part4">4. Training a PGTNet for remaining time prediction:</a>**

To train and evaluate PGTNet, we employ the implementation of [GraphGPS: General Powerful Scalable Graph Transformers](https://github.com/rampasek/GraphGPS). However, in order to use it for remaining time prediction of business process instances, you need to adjust some part of the original implementation. This can be achieved by running the following command:
```
python file_transfer.py
```
This script copies 5 important python scripts which take care of all necessary adjustments to the original implementation of GPS Graph Transformer recipe:

a. In **main.py** , the most important change is that a train mode called 'event-inference' is added to customize the inference step.

b. In **master_loader.py**, the most important change is that several new dataset classes are added to handle graph representation of event logs.

c. The python script **GTeventlogHandler.py** includes multiple **InMemoryDataset** Pytorch Geometric classes. We created one seperate class for each event log.

d. The python scripts **linear_edge_encoder.py** and **two_layer_linear_edge_encoder.py** are specifically designed for edge embedding in the remaining cycle time prediction problem.

Once abovementioned adjustments are done, training PGTNet is straightforward. Training is done using the relevant .yml configuration file which specifies all hyperparameters and training parameters. All configuration files required to train PGTNet based on the event logs used in our experiments are collected [here](https://github.com/keyvan-amiri/PGTNet/tree/main/training_configs). The **file_transfer.py** script also copy all required configuration files for training and evaluation of PGTNet to the relevant folder (i.e., configs/GPS) in **GPS repository**.

For training PGTNet, you need to navigate to the root directory of **GPS repository** and run **main.py** script:
```
cd ..
python main.py --cfg configs/GPS/bpic2015m1-GPS+LapPE+RWSE-ckptbest.yaml run_multiple_splits [0,1,2,3,4] seed 42
```
As we mentioned in our paper, to evaluate robustness of our approach we trained and evaluated PGTNet using three different random seeds. These random seeds are 42, 56, 89. Each time you want to train PGTNet for specific event log and specific seed number, you should adjust the **training configuration file** name, and the seed number in this command.

The [**training configuration files**](https://github.com/keyvan-amiri/PGTNet/tree/main/training_configs) include all required training hyperparameters. Following table briefly discusses the most important parameters:

| Parameter name | Parameter description |
|----------|----------|
| out_dir  | Name of the directory in which the results will be saved (e.g., results)| 
| metric_best | The metric that is used for results. In our case, it is always **"mae"** (Mean Absolute Error). There is another parameter called "metric_agg" which determines whether the metric should be maximized or minimized (in our case it is always set to "argmin".)| 
| dataset.format | Name of the PyG data object class that is used (e.g., PyG-EVENTBPIC15M1)| 
| dataset.task | Specifies the task level. In our case, it is always set to **"graph"** since we always have a graph-level prediction task at hand.| 
| dataset.task_type | Specifies the task type. In our case, it is always set to **"regression"**.| 
| dataset.split_mode | while **"cv-kfold-5"** specifies cross-fold validation data split, **"standard"** can be used for holdout data split.| 
| node_encoder_name | Specifies the encoding that will be employed for nodes. For instance, in **"TypeDictNode+LapPE+RWSE"**, "TypeDictNode" refers to embedding layer, and "LapPE+RWSE" refers to the type of PE/SEs that are used. There is another parameter called **node_encoder_num_types** which should be set the number of activity classes in the event log. For instance, node_encoder_num_types: 396 for the BPIC15-1 event log.| 
| edge_encoder_name | Specifies the encoding that will be employed for edges. For instance, "TwoLayerLinearEdge" refers to two linear layers.| 
| PE/SE parameters | Depending of type of PE/SEs that are used, all relevant hyperparameter can be defined. For instance if "LapPE+RWSE" is used, hyperparameters can be defined using "posenc_LapPE" and "posenc_RWSE". These hyperparameters include a wide range of options for instance the size of PE can be defined using "dim_pe", and the model that is used for processing it can be defined using "model" (for instance, model: DeepSet).|
| train | Specify the most important training hyperparameters including the training mode (i.e., **train.mode**) and batch size (i.e., **train.batch_size**). We always use the **custom** mode for training. |
| model | Specifies the most important global design options. For instance, **model.type** defines type of the model and in our case is always a **GPSModel**. The **model.loss_fun** defines the loss fucntion which in our case is always **l1** (the L1 loss function is equivalent to Mean Absolute Error). The **model.graph_pooling** specifies type of graph pooling and for instance can be set to "mean".|
| gt | Specifies the most important design options with respect to Graph Transformer that will be employed. For instance, **gt.layer_type** defines type of MPNN and Transformer blocks within each GPS layer, and in our case is always set to **GINE+Transformer**. The **gt.layers** and **gt.n_heads** define the number of GPS layers and number of heads in each layer. The **gt.dim_hidden** defines the hidden dimentsion that is used for both node and edge features. Note that, this size also include PE/SEs that are incorporated into node and edge features. The **gt.dropout** and **gt.attn_dropout** define the dropout value for MPNN and Transformer blocks, respectively.|
| optim | Specifies the most important design options with respect to the optimizer. For instance, **optim.optimizer** specifies the optimizer type and in our case is always set to **adamW**. The **optim.base_lr** and **optim.weight_decay** define base learning rate and weight decay, respectively. The **optim.max_epoch** specifies number of training epochs, while **optim.scheduler** and **optim.num_warmup_epochs** specify type of schedule (in our case always **cosine_with_warmup**) and number of warmup epochs. |
<!-- This is not remaining of the table. -->

Training results are saved in a seperate folder which is located in the **results** folder in the root directory of **GPS repository**. Name of this folder is always equivalent to the name of the configuration file that is used for training. For instance running the previous command produces this folder: **bpic2015m1-GPS+LapPE+RWSE-ckptbest**

This folder contains the best models (i.e., checkpoints) for each of 5 different folds (best checkpoints are selected based on the minimum MAE over validation set). The checkpint files can be used for inference with PGTNet as it is discussed in the [next section](#part5). You can also visualize learning curve using tensorboard:
```
tensorboard --logdir=work/kamiriel/GraphGPS/results/bpic2015m1-GPS+LapPE+RWSE-ckptbest/agg #the location of the training results on your system 
```

<a name="part5">**5. Inference with PGTNet:**</a>

The inference (i.e., get prediction of PGTNet for all examples in the test set) can be done similar to the training step. To do so, run commands like: 
```
python main.py --cfg configs/GPS/bpic2015m1-GPS+LapPE+RWSE-ckptbest-eventinference.yaml run_multiple_splits [0,1,2,3,4] seed 42
```
All **inference configuration files** that are used in our experiments are collected [here](https://github.com/keyvan-amiri/PGTNet/tree/main/evaluation_configs).

In principle, the inference configuration files are similar to the training configuration files. The most important difference is that, the **"train.mode"** parameter is set to **"event-inference"** instead of "custom". The inference configuration files additionally include another parameter called **"pretrained.dir"** by which we specify the folder that contais training results. For instance, it can be something like this:
```
pretrained:
  dir: /work/kamiriel/GraphGPS/results/bpic2015m1-GPS+LapPE+RWSE-ckptbest #the location of the training results on your system
```

Running the inference script results in one dataframe including predictions for all event prefixes (We call it **prediction dataframe**). In prediction dataframe, each row represents a graph representation of an event prefix for which number of nodes, number of edges, real remaining time (normalized),  predicted remaining time (normalized), and mean absolute error (in days) are provided thorugh different columns.

While the mean of the **"MAE-days"** column is the average of mean absolute error for all 5 folds, prediction dataframe still needs to be processed for further performance analysis (e.g., earliness analysis). This further processing is required because it is not clear that each row of the prediction dataframe belongs to which trace (the prediction dataframe does not include case id in its columns), and what is the number of events of the prefix (remember that edges have weights, and therefore knowing number of nodes, and edges is not sufficient for inference about length of the prefix). Therefore, we need to use a matching algorithm (from rows of prediction dataframe to event prefixes in the original event log). This can be achieved by navigating to the root directory of **PGTNet repository** and running the following script:
```
cd PGTNet
python ResultHandler.py --dataset_name BPIC15_1 --seed_number 42 --inference_config 'bpic2015m1-GPS+LapPE+RWSE-ckptbest-eventinference'
```
Note that the dataset_name should match the relevant part of the name of prediction dataframe, meaning that it should be one of the elements of this set: {"BPIC15_1", "BPIC15_2", "BPIC15_3", "BPIC15_4", "BPIC15_5", "BPI_Challenge_2012", "BPI_Challenge_2012A", "BPI_Challenge_2012O", "BPI_Challenge_2012W",  "BPI_Challenge_2012C", "BPI_Challenge_2012CW", "BPI_Challenge_2013C", "BPI_Challenge_2013I", "BPIC20_DomesticDeclarations", "BPIC20_InternationalDeclarations", "env_permit", "HelpDesk",  "Hospital", "Sepsis", "Traffic_Fines"}

The result of this matching process is saved as a dataframe in the **PGTNet results** folder in the root directory of **PGTNet repository**, and can be used for further performance evaluation analysis.

**<a name="part6">6. Miscellaneous:</a>**

**Earliness of PGTNet's predictions:**

We are interested in models that not only have smaller MAE but also can make accurate predictions earlier, allowing more time for corrective actions. We used the method proposed in [Predictive Business Process Monitoring with LSTM Neural Networks](https://link.springer.com/chapter/10.1007/978-3-319-59536-8_30), which evaluates MAE across different event prefix lengths. In our paper, we have provided the predcition earliness analysis (i.e., MAE trends at different prefix lengths) only for BPIC15-4, Sepsis, Helpdesk, and BPIC12A event logs. Similar analysis for other event logs used in our experiments can be found [here](https://github.com/keyvan-amiri/PGTNet/tree/main/earliness_analysis).

**Ablation study:**

As it is discussed in our paper, we conducted an ablation study for which we trained a minimal PGTNet model, relying solely on edge weights (i.e., control-flow) and temporal features, thus omitting data attributes from consideration. To replicate our ablation study, you need to adjust the conversion script and use different configuration files which you can find [here](https://github.com/keyvan-amiri/PGTNet/tree/main/ablation_study). For the quantitative analysis of the contribution of PGTNet's architecture and the contribution of incorporating additional features to the remarkabel performance of PGTNet see this [plot](https://github.com/keyvan-amiri/PGTNet/blob/main/ablation_study/ablation_plot.pdf). 

**PGTNet's results for holdout data split:**

While we chose a 5-fold cross-validation strategy (CV=5) in our experiments, we also report the [results](https://github.com/keyvan-amiri/PGTNet/blob/main/holdout_results/README.md) obtained using holdout data splits for the sake of completeness. Note that, we used different training configuration files for holdout data split which can be found on the same [folder](https://github.com/keyvan-amiri/PGTNet/tree/main/holdout_results).

**Implementation of the baselines:**
As it is discussed in our paper, we compare our approach against three others:
1. DUMMY : A simple baseline that predicts the average remaining time of all training prefixes with the same length k as a given prefix.
2. [DALSTM](https://ieeexplore.ieee.org/abstract/document/8285184): An LSTM-based approach that was recently shown to have superior results among LSTMs used for remaining time prediction. To implement this baseline, we used the [**pmdlcompararator**](https://gitlab.citius.usc.es/efren.rama/pmdlcompararator) github repository of a recently published [benchamrk](https://ieeexplore.ieee.org/abstract/document/9667311). We adjusted this implementation as per follows: a) the original implementation was based on Keras, we have implemented the same model in Pytorch. b) Unlike the original implementation we excluded prefixes of length 1 to have a fair comparison (remember that our graph representation of event prefixes requires at least two events). c) We extended the original implemetation utilizing more event logs that were not part of the benchmark. Our implementation of DALSTM can be find [here](https://github.com/keyvan-amiri/PGTNet/tree/main/baselines/dalstm).
3. [ProcessTransformer](https://arxiv.org/abs/2104.00721): A transformer-based approach designed to overcome LSTM’s limitations that generally outperforms DALSTM. To implement this baseline, we used [**ProcessTransformer**](https://github.com/Zaharah/processtransformer) github repository. However, we adjusted the original implementation as per follows: a) the original implementation is based on Keras, we have provided the implementation of the same model in Pytorch framework. b) the original implementation does not include cross-validation data split. We extended it by considering both holdout and cross-validation data split. c) The performance evaluation of ProcessTransformer is conducted differently. To ensure having a fair comparison, predictions of event prefixes of length 1, and length n (where n is the number of events in the trace) are not included in our performance evaluation. More importantly, the metric that is reported in ProcessTransformer paper is not MAE (mean absolute error) as authers computed the average of errors for different prefix lengths while they did not account for different frequencies of different prefix lengths in the test set. However, MAE should reflect frequencies and can be considered as the weighted average of errors for different prefix lengths. We used this weighted average in our implementation to ensure having a fair comparison. Our implementation of ProcessTransformer model can be find [here](https://github.com/keyvan-amiri/PGTNet/tree/main/baselines/processtransformer).
4. [GGNN](https://dl.acm.org/doi/abs/10.1145/3589883.3589897): A graph-based approach which utilizes Gated Graph Neural Network (GGNN) for remaining time prediction. To implement this baseline we used [**GGNN**](https://github.com/duongtoan261196/RemainingCycleTimePrediction) github repository. We have extended the original implementation which was limited to two publicly available event logs. Furthermore, we have provided a modular python implementation which can be easily extended (the original implementation was in jupyter notebook). Finally, we improved the original implementation in multiple ways. For instance, the original search space for the widths of Gated Graph Convolution layers could not handle event logs with distinct activities more than 100. Like ProcessTransformer model, the provided metric is not MAE as it is averaged across different prefix lengths without considering the frequency of prefixes for each length. Our implementation for this model, can be find here.
