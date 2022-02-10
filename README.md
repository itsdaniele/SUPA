Generates point cloud showers in batches (one file for each batch) and optionally converts them to cell grid (images).

## Requirements

* torch
* networkx
* numpy
* tqdm
* h5py


## Usage

```[bash]
usage: main_discrete.py [-h] [-N NUM_SHOWERS] [-E INITIAL_ENERGY] [-grids] [-l] [-p SAVE_PATH] [-s SEED] [-bs BATCH_SIZE]

Sythnetic Calorimeter Simulator with particle propagation and scattering

optional arguments:
  -h, --help            show this help message and exit
  -N NUM_SHOWERS, --num-showers NUM_SHOWERS
                        Number of showers to generate. (default: 50)
  -E INITIAL_ENERGY, --initial-energy INITIAL_ENERGY
                        Initial energy of incoming particle (in GeV) (default: 65.0)
  -grids, --save-grids  Whether to convert and save at grid granularity (default: False)
  -l, --load            No generation. Retrieve showers from save_path (default: False)
  -p SAVE_PATH, --save-path SAVE_PATH
                        Path where to save the generated showers (default: None)
  -s SEED, --seed SEED  Numpy Random Seed (default: 1234)
  -bs BATCH_SIZE, --batch-size BATCH_SIZE
                        Number of showers to store in a single file (default: 10000)
```

## Example

```python main_discrete.py -p config_v3 -g -N 100 -bs 50 -E 65 -s 1122```

## Cell Grid Conversion

Need to specify the bin width and number of bins (currently hardcoded).


