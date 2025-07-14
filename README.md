# NeuroSTF: Neuroplastic Sparse Attention for Time-Series Forecasting

NeuroSTF is a novel deep learning model for multivariate time-series forecasting. It uses neuroplastic sparse attention combined with temporal convolution and MLP layers to efficiently capture long-range dependencies.

## Installation

Make sure you have Python 3.7 or higher installed. Then install the required packages with:

```bash
pip install -r requirements.txt
````

## Dataset

This project uses the ETTh1 and ETTh2 datasets, which are publicly available:

* Download from: [https://github.com/zhouhaoyi/ETDataset](https://github.com/zhouhaoyi/ETDataset)
* Specifically, download the files `ETTh1.csv` and `ETTh2.csv` from the repository.
* Place the CSV files in a local folder and provide the file path when running the script.

## Usage

To train and evaluate the NeuroSTF model on the ETTh1 dataset, run:

```bash
python neurostf.py path/to/ETTh1.csv
```

Replace `path/to/ETTh1.csv` with the actual path to the downloaded dataset file.

You can similarly run the code on ETTh2 by specifying its path.

The script will automatically handle data loading, training, validation, testing, and saving results.

## Citation

If you use this software in your research, please cite:

```bibtex
@article{pal2025neurostf,
  title={NeuroSTF: Neuroplastic Sparse Attention for Time-Series Forecasting},
  author={Ovi Pal},
  journal={Journal of Open Source Software},
  year={2025},
  volume={},
  number={},
  pages={},
  doi={}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
