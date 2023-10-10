# GT-Remaining-CycleTime
GPS graph transformers can be used to predict remaining cyle time of business process instances in a two-step approach:

**1. Data preparation:**
Converting event logs into graph datasets. For more information, see: [conversion directory](https://github.com/keyvan-amiri/GT-Remaining-CycleTime/tree/main/conversion). We already uploaded generated graph datasets in [conversion/transformation directory](https://github.com/keyvan-amiri/GT-Remaining-CycleTime/tree/main/conversion/transformation) in this repository. Therefore, this step can be skipped if you are not intreseted in conducting more experiments with feature engineering. In this case, generated graph dataset are directly used in the second step to train and evaluaate GPS graph transformers.

**2. Training and evaluation of GPS graph transformers:**
Our porposal is based on the original implementation of [GraphGPS: General Powerful Scalable Graph Transformers](https://github.com/rampasek/GraphGPS). Therefore, the first step is to clone the original repository and follow the [instructions](https://github.com/rampasek/GraphGPS#python-environment-setup-with-conda) for setting up a Python environement with Conda.
