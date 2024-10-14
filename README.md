<h1 align="center">
  <br>
MedEdit: Counterfactual Diffusion-based Image Editing on Brain MRI
  <br>
</h1>
</h1>
  <p align="center">
    <a href="https://www.linkedin.com/in/malek-ben-alaya/">Malek Ben Alaya</a> •
    <a href="https://compai-lab.github.io/author/daniel-m.-lang/">Daniel M. Lang</a> •
    <a href="https://www.neurokopfzentrum.med.tum.de/neuroradiologie/mitarbeiter-profil-wiestler.html">Benedikt Wiestler</a> •
    <a href="https://compai-lab.github.io/author/julia-a.-schnabel/">Julia A. Schnabel </a> •
    <a href="https://cosmin-bercea.com">Cosmin Bercea</a> 
  </p>
<h4 align="center">Official repository of the paper</h4>
<h4 align="center">MICCAI 2024 Simulation and Synthesis in Medical Imaging (SASHIMI) Workshop</h4>
<h4 align="center"><a href="https://link.springer.com/chapter/10.1007/978-3-031-73281-2_16">Proceedings</a>  • <a href="https://arxiv.org/pdf/2407.15270">Preprint</a> </h4>

<p align="center">
<img src="https://github.com/Malekba98/MedEdit/blob/main/assets/method_animation.gif">
</p>

## Citation

If you find our work useful, please cite our paper:
```
@InProceedings{alaya2024mededit,
author="Alaya, Malek Ben and Lang, Daniel M. and Wiestler, Benedikt and Schnabel, Julia A. and Bercea, Cosmin I.",
title="MedEdit: Counterfactual Diffusion-Based Image Editing on Brain MRI",
booktitle="MICCAI workshop on Simulation and Synthesis in Medical Imaging",
year="2025",
publisher="Springer Nature Switzerland",
pages="167--176",
}
```
```
@misc{alaya2024mededitcounterfactualdiffusionbasedimage,
      title={MedEdit: Counterfactual Diffusion-based Image Editing on Brain MRI}, 
      author={Malek Ben Alaya and Daniel M. Lang and Benedikt Wiestler and Julia A. Schnabel and Cosmin I. Bercea},
      year={2024},
      eprint={2407.15270},
      archivePrefix={arXiv},
      primaryClass={eess.IV},
      url={https://arxiv.org/abs/2407.15270}, 
}
```

> **Abstract:** *Denoising diffusion probabilistic models enable high fidelity image synthesis and editing. In biomedicine, these models facilitate counterfactual image editing, producing pairs of images where one is edited to simulate hypothetical conditions. For example, they can model the progression of specific diseases, such as stroke lesions. However, current image editing techniques often fail to generate realistic biomedical counterfactuals, either by inadequately modeling indirect pathological effects like brain atrophy or by excessively altering the scan, which disrupts correspondence to the original images.*
>
> *Here, we propose MedEdit, a conditional diffusion model for medical image editing. MedEdit induces pathology in specific areas while balancing the modeling of disease effects and preserving the original scan’s integrity. We evaluated MedEdit on the Atlas v2.0 stroke dataset using Frechet Inception Distance and Dice scores, outperforming state-of-the-art diffusion-based methods such as Palette (by 45%) and SDEdit (by 61%). Additionally, clinical evaluations by a board-certified neuroradiologist confirmed that MedEdit generated realistic stroke scans indistinguishable from real ones. We believe this work will enable counterfactual image editing research to further advance the development of realistic and clinically useful imaging tools.*

 
## Setup and Run

The code is based on the deep learning framework from the Institute of Machine Learning in Biomedical Imaging: https://github.com/compai-lab/iml-dl

### Framework Overview 

<p align="center">
<img src="https://github.com/Malekba98/MedEdit/blob/main/assets/iml_dl.png">
</p>

#### 1). Set up wandb (https://docs.wandb.ai/quickstart)

Sign up for a free account and login to your wandb account.
```bash
wandb login
```
Paste the API key from https://wandb.ai/authorize when prompted.

#### 2). Clone repository

```bash
git clone https://github.com/Malekba98/MedEdit.git
cd mededit
```

#### 3). Install requirements
*Optional* create virtual env:
```bash
conda create --name mededit python=3.9.19
conda activate mededit
```

```bash
pip install -r pip_requirements.txt
```

#### 4). Install PyTorch 

> Example installation: 
* *with cuda*: 
```
pip3 install torch==1.9.1+cu111 torchvision==0.10.1+cu111 -f https://download.pytorch.org/whl/torch_stable.html
```
* *w/o cuda*:
```
pip3 install torch==1.9.1 torchvision==0.10.1 -f https://download.pytorch.org/whl/torch_stable.html
```

#### 5). Download datasets 

<h4 align="center"><a href="https://fcon_1000.projects.nitrc.org/indi/retro/atlas.html">Atlas (Stroke) </a> </h4>

> Move the datasets to the target locations. You can find detailed information about the expected files and locations in the corresponding *.csv files under data/$DATASET/splits.

> *Alternatively you can use your own mid-axial slices of T1w brain scans with our <a href="https://www.dropbox.com/scl/fi/6m0zic01q53riu1ydyny8/model_pretraining_1500.pt?rlkey=ct6wdhuffollokrd5gigb1qsb&e=1&st=q3p9l105&dl=0"> pre-trained weights</a> or train from scratch on other anatomies and modalities.*

#### 6). Run the pipeline

Run the scripts with the corresponding config like this:

```bash
python core/Main.py --config_path ./projects/mededit/configs/mededit.yaml
python core/Main.py --config_path ./projects/baselines/configs/sdedit.yaml
python core/Main.py --config_path ./projects/baselines/configs/naive_repaint.yaml
python core/Main.py --config_path ./projects/baselines/configs/palette.yaml
```

Refer to the mededit.yaml for the default configuration. Store the pre-trained model from <a href="https://www.dropbox.com/scl/fi/6m0zic01q53riu1ydyny8/model_pretraining_1500.pt?rlkey=ct6wdhuffollokrd5gigb1qsb&e=1&st=q3p9l105&dl=0"> HERE</a> into the specified directory to skip the training part.


# That's it, enjoy! :rocket:



