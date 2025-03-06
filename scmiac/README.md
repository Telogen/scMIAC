# Installation

```
mamba create -n scMIAAC python=3.8
mamba activate scMIAAC

# install pytorch depending on your cuda version
# https://pytorch.org/get-started/previous-versions/
pip install torch==1.10.1+cu111 torchvision==0.11.2+cu111 torchaudio==0.10.1 -f https://download.pytorch.org/whl/cu111/torch_stable.html

# install other
pip install scanpy


pip install cosg


pip install scib 
# https://github.com/theislab/scib/blob/main/scib/knn_graph/README.md
# cd home/txm/miniforge3/envs/scMIAAC/lib/python3.8/site-packages/scib/knn_graph/
# g++ -std=c++11 -O3 knn_graph.cpp -o knn_graph.o
```


