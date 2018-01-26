# Deep Learning for Decentralized Parking Lot Occupancy Detection

This repo contains code to reproduce the experiments presented in [Deep Learning for Decentralized Parking Lot Occupancy Detection](https://www.sciencedirect.com/science/article/pii/S095741741630598X).

Visit the [project website](http://cnrpark.it/) for more info and resources (dataset, pre-trained models).

## Requirements

 - Caffe with Python interface (PyCaffe)

## Steps to reproduce experiments

 1. Clone this repo together with its submodules:

    ```bash
    git clone --recursive https://github.com/fabiocarrara/deep-parking.git
    ```

 2. Download the datasets using the following links and extract them somewhere.

    | Dataset | Link | Size | 
    | ------- | ---- | ---: |
    | CNRPark | http://cnrpark.it/dataset/CNRPark-Patches-150x150.zip | 36.6 MB |
    | CNR-EXT | http://cnrpark.it/dataset/CNR-EXT-Patches-150x150.zip | 449.5 MB |
    | PKLot   | visit [PKLot webpage](https://web.inf.ufpr.br/vri/databases/parking-lot-database/) | 4.6 GB |


 3. Get the dataset splits and extract them in the repo folder
    ```bash
    # Listfile containing dataset splits
    wget http://cnrpark.it/dataset/splits.zip
    unzip splits.zip
    ```

 4. Add a `config.py` files inside each folder in `splits/` to tell `pyffe` where the images are.
    The content of the files should be like this (adjust the `root_dir` attribute to the absolute path of the extracted datasets):
    ```python
    config = dict(root_folder = '/path/to/dataset/dir/')
    ```
    This path will be prepended to each line in the list files defining the various splits.


 5. Train and evaluate all the models by running:
    ```bash
    python main.py
    ```
    Modify `main.py` to select the experiments you want to reproduce.
    Run `pklot.py` if you want to train and evaluate our architecture on the PKLot splits only.
    
## Citation

```
@article{amato2017deep,
  title={Deep learning for decentralized parking lot occupancy detection},
  author={Amato, Giuseppe and Carrara, Fabio and Falchi, Fabrizio and Gennaro, Claudio and Meghini, Carlo and Vairo, Claudio},
  journal={Expert Systems with Applications},
  volume={72},
  pages={327--334},
  year={2017},
  publisher={Pergamon}
}
```

