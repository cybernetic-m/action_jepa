# TITLE


<img src="./images/image.png" alt="Description" width="400" height = "300" />

## 📖 Description

---
## 🔧 Installation 
Firstly you need to install all the dependencies needed to run inference with the VLA ActionJEPA or train a new ActionJEPA VLA.

1. **Clone and install requirements of this repository:**
 ```sh 
 git clone "https://github.com/cybernetic-m/action_jepa.git"
 cd action_jepa
 pip install -r requirements.txt
 ```
2. **Clone and install requirements of the LIBERO repository:**
 ```sh 
 git clone "https://github.com/Lifelong-Robot-Learning/LIBERO.git"  
 cd LIBERO
 pip install -r requirements.txt
 pip install -e .
 ```
3. **Clone and install requirements of the JEPA World Models repository:**
```sh
 git clone "https://github.com/facebookresearch/jepa-wms.git" 
 pip install -U git+https://github.com/huggingface/transformers
 ```

4. **Fix for Pytorch 2.6+:**

In the original LIBERO env I have changed the 164 line of the file:
LIBERO/libero/libero/benchmark/__init__.py:

BEFORE

_init_states = torch.load(init_states_path)_ 

AFTER

_init_states = torch.load(init_states_path, weights_only=False)_

To solve _UnpicklingError: Weights only load failed_ error.

Then run:

```sh
cp ./action_jepa/fix/__init__.py LIBERO/libero/libero/benchmark/__init__.py
 ```

### Download LIBERO datasets (optional)
If you want to retrain the model, you need to download the [**LIBERO dataset**](https://huggingface.co/datasets/yifengzhu-hf/LIBERO-datasets/tree/main):. 

**[INFO]**
- _libero_10_ size: 13.7 GB
- _libero_90_ size: 66.7 GB
- _libero_goal_ size: 6.37 GB
- _libero_object_ size: 7.44 GB
- _libero_spatial_ size: 6.24 GB
-----------------------------
- TOTAL DATASET SIZE: **100 GB**
You can download it in the old (**OPTION 1**) or if the previous did not work using the modern (**OPTION 2**) way:

1. **OPTION 1**
You can download all the datasets using:
```sh
 cd LIBERO
 python benchmark_scripts/download_libero_datasets.py
```
or alternatively only a subset (where ``` DATASET ``` can be ``` libero_spatial, libero_object, libero_100, libero_goal```):
```sh
 cd LIBERO
 python benchmark_scripts/download_libero_datasets.py --datasets DATASET
```

2. **OPTION 2 (Hugging Face)**
If the previous method did not work, you can use the Hugging Face method:
```sh
 cd LIBERO
 python benchmark_scripts/download_libero_datasets.py --use-huggingface
 python benchmark_scripts/download_libero_datasets.py --datasets libero_100
```

### Download pre-trained models

## 🔧 Instructions (Colab)

### LIBERO environment setup
In order to 
### Download LIBERO datasets (optional)
If you want to retrain the model, you need to download the LIBERO trajectories following these steps:

### V-JEPA 2 

### Download pre-trained models



---
## 📽️ Presentation

CLICK HERE [<img src="https://ssl.gstatic.com/docs/doclist/images/mediatype/icon_1_presentation_x128.png" alt="Google Slides" width= "30" height="30"/>]() 

---
## 📚 Datasets

---
## 📊 Experiment and results


---
## 👤 Author

**Massimo Romano**  

GitHub: [@cybernetic-m](https://github.com/cybernetic-m)  

LinkedIn: [Massimo Romano](https://www.linkedin.com/in/massimo-romano-01/)

Website: [Massimo Romano](https://www.massimo-romano.com)


## 📄 License

This project is licensed under the MIT License.  
See the [LICENSE](./LICENSE) file for details.

