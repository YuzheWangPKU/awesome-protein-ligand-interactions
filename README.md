<h1 align="center">
💊📝 Awesome Protein-Ligand Interactions
</h1>

This is the official repository for the review article *"Modeling Protein–Ligand Interactions for Drug Discovery in the Era of Deep Learning"* published in *Chemical Society Reviews*. [[link]](https://doi.org/10.1039/D5CS00415B). We curate papers, tools, and resources related to deep learning-based
modeling of protein–ligand interactions for drug discovery.

<p align="center">
  <img src="overview.png" width="512">
</p>

**Note**: This repository is not intended to be an exhaustive or definitive collection. The included papers and resources were selected based on the authors’ perspectives and interests, and we sincerely apologize for any important work that may have been unintentionally omitted. We warmly welcome community contributions and suggestions to help improve and expand this resource.

## Table of Contents
- [Table of Contents](#table-of-contents)
- [1. Deep Learning-Augmented Molecular Dynamics](#1)
  - [1.1 MD simulations of protein-ligand complexes with MLFFs](#1.1)
  - [1.2 Learning protein-ligand interactions from MD trajectories](#1.2)
  - [1.3 Generative modeling of MD trajectories for protein-ligand complexes](#1.3)
  - [1.4 Benchmarks, datasets, and tools](#1.4)
- [2. Deep Learning-Enhanced Molecular Docking and Virtual Screening](#2)
  - [2.1 Deep learning-based docking methods](#2.1)
  - [2.2 Deep learning scoring functions and binding affinity prediction](#2.2)
  - [2.3 Deep learning-accelerated virtual screening](#2.3)
  - [2.4 Benchmarks, datasets, and tools](#2.4)
- [3. End-to-End Structural Modeling](#3)
  - [3.1 Protein structure prediction with applications in drug discovery](#3.1)
  - [3.2 Generative modeling of protein–ligand complexes and equilibrium ensembles](#3.2)
- [4. Structure-Based *De Novo* Drug Design](#4)
  - [4.1 Structure-based de novo drug design methods](#4.1)
  - [4.2 Ligand-based design and lead optimization methods](#4.2)
  - [4.3 Benchmarks, datasets, and tools](#4.3)
- [5. Sequence-Based Methods for Drug Discovery](#5)

## 1. Deep Learning-Augmented Molecular Dynamics <a name="1"></a>

### 1.1 MD simulations of protein-ligand complexes with MLFFs <a name="1.1"></a>

| Name             | Paper Title                                                  | Year | **Venue**  | Resources                                                    | Notes       |
| ---------------- | ------------------------------------------------------------ | ---- | ---------- | ------------------------------------------------------------ | ----------- |
| /                | Simulating protein-ligand binding with neural network potentials [[paper]](https://doi.org/10.1039/C9SC06017K) | 2020 | Chem. Sci. | /                                                            | NNP/MM      |
| qmlify           | Towards chemical accuracy for alchemical free energy calculations with hybrid physics-based machine learning/molecular mechanics potentials [[preprint]](https://doi.org/10.1101/2020.07.29.227959) | 2020 | bioRxiv    | [[code]](https://github.com/choderalab/qmlify)               | NNP/MM      |
| /                | NNP/MM: accelerating molecular dynamics simulations with machine learning potentials and molecular mechanics [[paper]](https://doi.org/10.1021/acs.jcim.3c00773) | 2023 | JCIM       | [[code]](https://github.com/openmm/nnpops)                   | NNP/MM      |
| /                | Enhancing protein-ligand binding affinity predictions using neural network potentials [[paper]](https://doi.org/10.1021/acs.jcim.3c02031) | 2024 | JCIM       | [[data]](https://github.com/compsciencelab/ATM_benchmark/tree/main/ATM_With_NNPs) | NNP/MM      |
| AP-Net           | A physics-aware neural network for protein-ligand interactions with quantum chemical accuracy [[paper]](https://doi.org/10.1039/D4SC01029A) | 2024 | Chem. Sci. | [[code]](https://github.com/zachglick/apnet)                 | Native MLFF |
| espaloma-0.3     | Machine-learned molecular mechanics force fields from large-scale quantum chemical data [[paper]](https://doi.org/10.1039/D4SC00690A) | 2024 | Chem. Sci. | [[code]](https://github.com/choderalab/refit-espaloma)       | Native MLFF |
| QuantumBind-RBFE | QuantumBind-RBFE: accurate relative binding free energy calculations using neural network potentials [[paper]](https://doi.org/10.1021/acs.jcim.5c00033) | 2025 | JCIM       | [[code]](https://github.com/Acellera/quantumbind_rbfe)       | NNP/MM      |

### 1.2 Learning protein-ligand interactions from MD trajectories <a name="1.2"></a>

| Name       | Paper Title                                                  | Year | **Venue**    | Resources                                        |
| ---------- | ------------------------------------------------------------ | ---- | ------------ | ------------------------------------------------ |
| NRI-MD     | Neural relational inference to learn long-range allosteric interactions in proteins from molecular dynamics simulations [[paper]](https://doi.org/10.1038/s41467-022-29331-3) | 2022 | Nat. Commun. | [[code]](https://github.com/juexinwang/NRI-MD)   |
| ProtMD     | Pre-training of equivariant graph matching networks with conformation flexibility for drug binding [[paper]](https://doi.org/10.1002/advs.202203796) | 2022 | Adv. Sci.    | [[code]](https://github.com/smiles724/ProtMD)    |
| Dynaformer | From static to dynamic structures: improving binding affinity prediction with graph-based deep learning [[paper]](https://doi.org/10.1002/advs.202405404) | 2024 | Adv. Sci.    | [[code]](https://github.com/Minys233/Dynaformer) |
| MDbind | Spatio-temporal learning from molecular dynamics simulations for protein–ligand binding affinity prediction [[paper]](https://doi.org/10.1093/bioinformatics/btaf429) | 2025 | Bioinformatics | [[code]](https://github.com/ICOA-SBC/MD_DL_BA) |

### 1.3 Generative modeling of MD trajectories for protein-ligand complexes <a name="1.3"></a>

| Name       | Paper Title                                                  | Year | **Venue**    | Resources                                        |
| ---------- | ------------------------------------------------------------ | ---- | ------------ | ------------------------------------------------ |
| BioMD     | BioMD: all-atom generative model for biomolecular dynamics simulation [[paper]](https://openreview.net/forum?id=LQDeJk6NOr) | 2026 | ICLR | /   |
| BioKinema     | Physically grounded generative modeling of all-atom biomolecular dynamics [[preprint]](https://www.biorxiv.org/content/10.64898/2026.02.15.705956) | 2026 | bioRxiv | [[code]](https://github.com/IDEA-XL/BioKinema)   |

### 1.4 Benchmarks, datasets, and tools <a name="1.4"></a>

| Name   | Paper Title                                                  | Year | Venue             | Resources                                                    | Notes                             |
| ------ | ------------------------------------------------------------ | ---- | ----------------- | ------------------------------------------------------------ | --------------------------------- |
| SPICE  | SPICE, a dataset of drug-like molecules and peptides for training machine learning potentials [[paper]](https://doi.org/10.1038/s41597-022-01882-6) | 2023 | Sci. Data         | [[data]](https://zenodo.org/records/7338495) [[code]](https://github.com/openmm/spice-dataset) | QM calculations                   |
| MISATO | MISATO: machine learning dataset of protein-ligand complexes for structure-based drug discovery [[paper]](https://doi.org/10.1038/s43588-024-00627-2) | 2024 | Nat. Comput. Sci. | [[data]](https://zenodo.org/records/7711953) [[code]](https://github.com/t7morgen/misato-dataset/) | QM calculations & MD trajectories |
| PLAS-20k | PLAS-20k: extended dataset of protein-ligand affinities from MD simulations for machine learning applications [[paper]](https://doi.org/10.1038/s41597-023-02872-y) | 2024 | Sci. Data | [[data]](https://doi.org/10.6084/m9.figshare.c.6742521.v2) | MD trajectories & MMPBSA calculations |
| OMol25 | The open molecules 2025 (OMol25) dataset, evaluations, and models [[preprint]](https://arxiv.org/abs/2505.08762) | 2025 | arXiv             | [[data]](https://huggingface.co/facebook/OMol25) [[code]](https://github.com/facebookresearch/fairchem) [[blog]](https://ai.meta.com/blog/meta-fair-science-new-open-source-releases/) | QM calculations                   |
| DD-13M | A Novel 4-D Dataset Paradigm for Studying Complete Ligand-Protein Dissociation Dynamics [[preprint]](https://arxiv.org/abs/2504.18367) | 2025 | arXiv             | [[data]](https://huggingface.co/SZBL-IDEA) [[webpage]](https://aimm.szbl.ac.cn/database/ddd/#/home)                   | MD trajectories                   |
| qcMol | qcMol: a large-scale dataset of 1.2 million molecules with high-quality quantum chemical annotations for molecular representation learning [[preprint]](https://doi.org/10.1101/2025.09.07.674462) | 2025 | bioRxiv             | [[webpage]](https://structpred.life.tsinghua.edu.cn/qcmol/)                   | QM calculations                   |
| BMS25 | Biomolecular multiscale simulation (BMS25) dataset to train neural network potentials for QM/MM settings with electrostatic embedding [[preprint]](https://doi.org/10.26434/chemrxiv-2025-xzp6k) | 2025 | ChemRxiv             | [[data]](https://doi.org/10.3929/ethz-c-000788484)                   | QM/MM calculations                   |
| AnewFEP | Physics-Based vs AI-Based Free Energy Prediction for Protein-Ligand Potency: Public Benchmarks and Internal Project Evidence [[preprint]](https://doi.org/10.26434/chemrxiv.15002526/v1) | 2026 | ChemRxiv             | / | RBFE benchmark & prospective evidence                   |

## 2. Deep Learning-Enhanced Molecular Docking and Virtual Screening <a name="2"></a>

### 2.1 Deep learning-based docking methods <a name="2.1"></a>

| Name        | Paper Title                                                  | Year | **Venue**          | Resources                                               | Notes               |
| ----------- | ------------------------------------------------------------ | ---- | ------------------ | ------------------------------------------------------- | ------------------- |
| DeepDock    | A geometric deep learning approach to predict binding conformations of bioactive molecules [[paper]](https://doi.org/10.1038/s42256-021-00409-9) | 2021 | Nat. Mach. Intell. | [[code]](https://github.com/OptiMaL-PSE-Lab/DeepDock)   | Rigid receptor      |
| TankBind    | TankBind: trigonometry-aware neural networks for drug-protein binding structure prediction [[paper]](https://proceedings.neurips.cc/paper_files/paper/2022/hash/2f89a23a19d1617e7fb16d4f7a049ce2-Abstract-Conference.html) | 2022 | NeurIPS            | [[code]](https://github.com/luwei0917/TankBind)         | Rigid receptor      |
| EquiBind    | EquiBind: geometric deep learning for drug binding structure prediction [[paper]](https://proceedings.mlr.press/v162/stark22b.html) | 2022 | ICML               | [[code]](https://github.com/HannesStark/EquiBind)       | Rigid receptor      |
| E3Bind      | E3Bind: an end-to-end equivariant network for protein-ligand docking [[paper]](https://openreview.net/forum?id=sO1QiAftQFv) | 2022 | ICLR               | /                                                       | Rigid receptor      |
| DiffDock    | DiffDock: diffusion steps, twists, and turns for molecular docking [[paper]](https://openreview.net/forum?id=kKF8_K-mBbS) | 2022 | ICLR               | [[code]](https://github.com/gcorso/DiffDock)            | Rigid receptor      |
| Uni-Mol     | Uni-Mol: a universal 3D molecular representation learning framework [[paper]](https://openreview.net/forum?id=6K2RM6wVqKu) | 2023 | ICLR               | [[code]](https://github.com/deepmodeling/Uni-Mol)       | Rigid receptor      |
| KarmaDock   | Efficient and accurate large library ligand docking with KarmaDock [[paper]](https://doi.org/10.1038/s43588-023-00511-5) | 2023 | Nat. Comput. Sci.  | [[code]](https://github.com/schrojunzhang/KarmaDock)    | Rigid receptor      |
| FABind      | FABind: fast and accurate protein-ligand binding [[paper]](https://proceedings.neurips.cc/paper_files/paper/2023/hash/aee1de5f335558b546b7e58c380be087-Abstract-Conference.html) | 2023 | NeurIPS            | [[code]](https://github.com/QizhiPei/FABind)            | Rigid receptor      |
| FlexPose    | Equivariant flexible modeling of the protein-ligand binding pose with geometric deep learning [[paper]](https://doi.org/10.1021/acs.jctc.3c00273) | 2023 | JCTC               | [[code]](https://github.com/tiejundong/FlexPose)        | Side-chain flexible |
| CarsiDock   | CarsiDock: a deep learning paradigm for accurate protein-ligand docking and screening based on large-scale pre-training [[paper]](https://doi.org/10.1039/D3SC05552C) | 2024 | Chem. Sci.         | [[code]](https://github.com/carbonsilicon-ai/CarsiDock) | Rigid receptor      |
| DeltaDock   | DeltaDock: a unified framework for accurate, efficient, and physically reliable molecular docking [[paper]](https://proceedings.neurips.cc/paper_files/paper/2024/hash/cd97da5366de69250442901abcdd4c0a-Abstract-Conference.html) | 2024 | NeurIPS            | [[code]](https://github.com/jiaxianyan/DeltaDock)       | Rigid receptor      |
| DiffBindFR  | DiffBindFR: an SE (3) equivariant network for flexible protein-ligand docking [[paper]](https://doi.org/10.1039/D3SC06803J) | 2024 | Chem. Sci.         | [[code]](https://github.com/HBioquant/DiffBindFR)       | Side-chain flexible |
| DynamicBind | DynamicBind: predicting ligand-specific protein-ligand complex structure with a deep equivariant generative model [[paper]](https://doi.org/10.1038/s41467-024-45461-2) | 2024 | Nat. Commun.       | [[code]](https://github.com/luwei0917/DynamicBind)      | Fully flexible      |
| PackDock | Flexible protein–ligand docking with diffusion-based side-chain packing [[paper]](https://doi.org/10.1073/pnas.2511925122) | 2025 | PNAS       | [[code]](https://github.com/Zhang-Runze/PackDock)      | Side-chain flexible      |
| Matcha | Matcha: Multi-Stage Riemannian Flow Matching for Accurate and Physically Valid Molecular Docking [[preprint]](https://arxiv.org/abs/2510.14586) | 2025 | arXiv       | [[code]](https://github.com/LigandPro/Matcha)      | Rigid receptor      |

### 2.2 Deep learning scoring functions and binding affinity prediction <a name="2.2"></a>

| Name      | Paper Title                                                  | Year | **Venue**          | Resources                                         | Notes                     |
| --------- | ------------------------------------------------------------ | ---- | ------------------ | ------------------------------------------------- | ------------------------- |
| OnionNet  | OnionNet: a multiple-layer intermolecular-contact-based convolutional neural network for protein-ligand binding affinity prediction [[paper]](https://doi.org/10.1021/acsomega.9b01997) | 2019 | ACS Omega          | [[code]](https://github.com/zhenglz/onionnet)     | /                         |
| RTMScore  | Boosting protein-ligand binding pose prediction and virtual screening based on residue-atom distance likelihood potential and graph transformer [[paper]](https://doi.org/10.1021/acs.jmedchem.2c00991) | 2022 | JMC                | [[code]](https://github.com/sc8668/RTMScore)      | /                         |
| PIGNet    | PIGNet: a physics-informed deep learning model toward generalized drug-target interaction predictions [[paper]](https://doi.org/10.1039/D1SC06946B) | 2022 | Chem. Sci.         | [[code]](https://github.com/ACE-KAIST/PIGNet)     | Physical terms            |
| GenScore  | A generalized protein-ligand scoring framework with balanced scoring, docking, ranking and screening powers [[paper]](https://doi.org/10.1039/D3SC02044D) | 2023 | Chem. Sci.         | [[code]](https://github.com/sc8668/GenScore)      | /                         |
| PBCNet    | Computing the relative binding affinity of ligands based on a pairwise binding comparison network [[paper]](https://doi.org/10.1038/s43588-023-00529-9) | 2023 | Nat. Comput. Sci.  | [[code]](https://doi.org/10.24433/CO.1095515.v2)  | Relative binding affinity |
| PLANET    | PLANET: a multi-objective graph neural network model for protein-ligand binding affinity prediction [[paper]](https://doi.org/10.1021/acs.jcim.3c00253) | 2023 | JCIM               | [[code]](https://github.com/ComputArtCMCG/PLANET) | Binding pose-free         |
| EquiScore | Generic protein-ligand interaction scoring by integrating physical prior knowledge and data augmentation modelling [[paper]](https://doi.org/10.1038/s42256-024-00849-z) | 2024 | Nat. Mach. Intell. | [[code]](https://github.com/CAODH/EquiScore)      | Interaction fingerprints  |
| DeepRLI   | DeepRLI: a multi-objective framework for universal protein-ligand interaction prediction [[paper]](https://doi.org/10.1039/D4DD00403E) | 2025 | Digit. Discov.     | [[code]](https://github.com/fairydance/DeepRLI)   | /                         |
| PBCNet2.0 | Atomic-level protein–ligand recognition with PBCNet2.0 for probe discovery [[paper]](https://doi.org/10.1038/s41589-026-02241-x) | 2026 | Nat. Chem. Biol.   | [[code]](https://github.com/YuJie-0202/PBCNet2.0) | Relative binding affinity, probe discovery |
| BioScore | BioScore: a foundational scoring function for diverse biomolecular complexes [[preprint]](https://arxiv.org/abs/2507.10877) | 2025 | arXiv            | /                                                 | A unified scoring function |


### 2.3 Deep learning-accelerated virtual screening <a name="2.3"></a>

| Name         | Paper Title                                                  | Year | **Venue**         | Resources                                                    | Notes             |
| ------------ | ------------------------------------------------------------ | ---- | ----------------- | ------------------------------------------------------------ | ----------------- |
| Deep Docking | Deep docking: a deep learning platform for augmentation of structure based drug discovery [[paper]](https://doi.org/10.1021/acscentsci.0c00229) | 2020 | ACS Cent. Sci.    | [[code]](https://github.com/jamesgleave/DD_protocol)         | /                 |
| DrugCLIP     | DrugCLIP: contrastive protein-molecule representation learning for virtual screening [[paper]](https://proceedings.neurips.cc/paper_files/paper/2023/hash/8bd31288ad8e9a31d519fdeede7ee47d-Abstract-Conference.html) | 2023 | NeurIPS           | [[code]](https://github.com/bowen-gao/DrugCLIP) [[project page]](https://drug-the-whole-genome.yanyanlan.com/) | CLIP architecture |
| OpenVS       | An artificial intelligence accelerated virtual screening platform for drug discovery [[paper]](https://doi.org/10.1038/s41467-024-52061-7) | 2024 | Nat. Commun.      | [[code]](https://github.com/gfzhou/OpenVS)                   | Active learning   |
| /            | Rapid traversal of vast chemical space using machine learning-guided docking screens [[paper]](https://doi.org/10.1038/s43588-025-00777-x) | 2025 | Nat. Comput. Sci. | [[code]](https://github.com/carlssonlab/conformalpredictor)  | /                 |
| FragmentScope            | FragmentScope - exploring the fragment space with learned surface representations [[preprint]](https://doi.org/10.64898/2025.12.16.694391) | 2025 | bioRxiv | [[code]](https://github.com/LPDI-EPFL/FragmentScope)  | Surface-based fragment screening |
| BoltzMol-1   | BoltzMol-1: Towards reliable virtual screening for fast and cost-effective hit discovery [[technical report]](https://boltz.bio/boltzmol1-technical-report.pdf) | 2026 | Technical Report  | [[platform]](https://boltz.bio/) | AI-driven prospective hit discovery |
| CombiDOCK & MINT-Dock | Combinatorial docking and molecular generation to navigate over 100-billion molecules for prospective ligand discovery [[preprint]](https://doi.org/10.64898/2026.06.07.730716) | 2026 | bioRxiv | / | 100B-scale make-on-demand screening |


### 2.4 Benchmarks, datasets, and tools <a name="2.4"></a>

| Name               | Paper Title                                                  | Year | **Venue**          | Resources                                                    | Notes                              |
| ------------------ | ------------------------------------------------------------ | ---- | ------------------ | ------------------------------------------------------------ | ---------------------------------- |
| BindingDB          | BindingDB: a web-accessible database of experimentally determined protein-ligand binding affinities [[paper]](https://doi.org/10.1093/nar/gkl999) | 2007 | NAR                | [[webpage]](https://www.bindingdb.org/rwd/bind/index.jsp)    | Database                           |
| ChEMBL             | ChEMBL: a large-scale bioactivity database for drug discovery [[paper]](https://doi.org/10.1093/nar/gkr777) | 2012 | NAR                | [[webpage]](https://www.ebi.ac.uk/chembl/) [[code]](https://github.com/chembl/ChEMBL_Structure_Pipeline) | Database                           |
| ZINC Database      | ZINC 15-ligand discovery for everyone [[paper]](https://doi.org/10.1021/acs.jcim.5b00559) | 2015 | JCIM               | [[webpage]](https://cartblanche22.docking.org/)    | Database                           |
| Enamine REAL Space | Generating multibillion chemical space of readily accessible screening compounds [[paper]](https://doi.org/10.1016/j.isci.2020.101681) | 2020 | iScience           | [[webpage]](https://enamine.net/compound-collections/real-compounds) | Make-on-demand database            |
| PoseBusters        | PoseBusters: AI-based docking methods fail to generate physically valid poses or generalise to novel sequences [[paper]](https://doi.org/10.1039/D3SC04185A) | 2024 | Chem. Sci.         | [[code]](https://github.com/maabuu/posebusters) [[doc]](https://posebusters.readthedocs.io/en/latest/) | Benchmark, pose quality check      |
| Leak Proof PDBBind | Leak Proof PDBBind: A Reorganized Data Set of Protein-Ligand Complexes for More Generalizable Binding Affinity Prediction [[paper]](https://doi.org/10.1021/acs.jpcb.5c08598) | 2026 | J. Phys. Chem. B | [[code]](https://github.com/THGLab/LP-PDBBind)               | Dataset split                      |
| SPECTRA            | Evaluating generalizability of artificial intelligence models for molecular datasets [[paper]](https://doi.org/10.1038/s42256-024-00931-6) | 2024 | Nat. Mach. Intell. | [[code]](https://github.com/mims-harvard/SPECTRA)            | Framework for model evaluation     |
| SAIR               | SAIR: Enabling Deep Learning for Protein-Ligand Interactions with a Synthetic Structural Dataset [[paper]](https://openreview.net/forum?id=qgk2F6jxH4) | 2026 | ICLR | [[data]](https://pub.sandboxaq.com/data/ic50-dataset)        | Database, AF3-predicted structures |
| QUID               | Extending quantum-mechanical benchmark accuracy to biological ligand-pocket interactions [[paper]](https://doi.org/10.1038/s41467-025-63587-9) | 2025 | Nat. Commun.            | [[data]](https://github.com/MirelaVP/QUID)        | Benchmark, QM-level pocket-ligand interaction systems  |
| BindFlow               | BindFlow: A Free, User-Friendly Pipeline for Absolute Binding Free Energy Calculations Using Free Energy Perturbation or MM(PB/GB)SA [[paper]](https://doi.org/10.1021/acs.jctc.5c02026) | 2026 | JCTC | [[code]](https://github.com/ale94mleon/BindFlow)        | Tool, FEP & MM(PB/GB)SA pipeline |

## 3. End-to-End Structural Modeling <a name="3"></a>

### 3.1 Protein structure prediction with applications in drug discovery <a name="3.1"></a>

| Paper Title                                                  | Year | **Venue**    | Resources                                                    | Notes                  |
| ------------------------------------------------------------ | ---- | ------------ | ------------------------------------------------------------ | ---------------------- |
| AlphaFold2 structures guide prospective ligand discovery [[paper]](https://doi.org/10.1126/science.adn6354) | 2024 | Science      | /                                                            | GPCR targets           |
| AlphaFold accelerates artificial intelligence powered drug discovery: efficient discovery of a novel CDK20 small molecule inhibitor [[paper]](https://doi.org/10.1039/D2SC05709C) | 2023 | Chem. Sci.   | [[PandaOmics]](https://pharma.ai/pandaomics) [[Chemistry42]](https://www.chemistry42.com/) | CDK20                  |
| AlphaFold accelerated discovery of psychotropic agonists targeting the trace amine-associated receptor 1 [[paper]](https://doi.org/10.1126/sciadv.adn1524) | 2024 | Sci. Adv.    | /                                                            | TAAR1                  |
| Virtual library docking for cannabinoid-1 receptor agonists with reduced side effects [[paper]](https://doi.org/10.1038/s41467-025-57136-7) | 2025 | Nat. Commun. | [[data]](https://lsd.docking.org/targets/CB1R)               | Cannabinoid-1 receptor |

### 3.2 Generative modeling of protein–ligand complexes and equilibrium ensembles <a name="3.2"></a>

| Name                 | Paper Title                                                  | Year | **Venue**          | Resources                                                    | Notes                                     |
| -------------------- | ------------------------------------------------------------ | ---- | ------------------ | ------------------------------------------------------------ | ----------------------------------------- |
| NeuralPLexer         | State-specific protein-ligand complex structure prediction with a multiscale deep generative model [[paper]](https://doi.org/10.1038/s42256-024-00792-z) | 2024 | Nat. Mach. Intell. | [[code]](https://github.com/zrqiao/NeuralPLexer)             | /                                         |
| RoseTTAFold All-Atom | Generalized biomolecular modeling and design with RoseTTAFold All-Atom [[paper]](https://doi.org/10.1126/science.adl2528) | 2024 | Science            | [[code]](https://github.com/baker-laboratory/RoseTTAFold-All-Atom) | /                                         |
| AlphaFold3           | Accurate structure prediction of biomolecular interactions with AlphaFold 3 [[paper]](https://doi.org/10.1038/s41586-024-07487-w) | 2024 | Nature             | [[code]](https://github.com/google-deepmind/alphafold3) [[server]](https://alphafoldserver.com/welcome) | Non-commercial usage                      |
| Umol                 | Structure prediction of protein-ligand complexes from sequence information with Umol [[paper]](https://doi.org/10.1038/s41467-024-48837-6) | 2024 | Nat. Commun.       | [[code]](https://github.com/patrickbryant1/Umol)             | /                                         |
| Chai-1               | Chai-1: decoding the molecular interactions of life [[preprint]](https://doi.org/10.1101/2024.10.10.615955) | 2024 | bioRxiv            | [[code]](https://github.com/chaidiscovery/chai-lab) [[server]](https://lab.chaidiscovery.com/dashboard) | /                                         |
| Protenix             | Protenix - Advancing Structure Prediction Through a Comprehensive AlphaFold3 Reproduction [[preprint]](https://doi.org/10.1101/2025.01.08.631967) | 2025 | bioRxiv            | [[code]](https://github.com/bytedance/Protenix) [[server]](https://protenix-server.com/login) | /                                         |
| Boltz-1              | Boltz-1: democratizing biomolecular interaction modeling [[preprint]](https://doi.org/10.1101/2024.11.19.624167) | 2024 | bioRxiv            | [[code]](https://github.com/jwohlwend/boltz) [[blog]](https://jclinic.mit.edu/boltz-1/) | /                                         |
| Boltz-2              | Boltz-2: Towards accurate and efficient binding affinity prediction [[preprint]](https://doi.org/10.1101/2025.06.14.659707) | 2025 | bioRxiv            | [[code]](https://github.com/jwohlwend/boltz) [[design-code]](https://github.com/recursionpharma/synflownet-boltz) | FEP-level affinity prediciton             |
| GeoFlow-V2              | GeoFlow-V2: a unified atomic diffusion model for protein structure prediction and de novo design [[preprint]](https://doi.org/10.1101/2025.05.06.652551) | 2025 | bioRxiv            | [[server]](https://prot.design/) | Protein Binder & antibody design                                         |
| Chai-2               | Zero-shot antibody design in a 24-well plate [[preprint]](https://doi.org/10.1101/2025.07.05.663018) | 2025 | bioRxiv                  | [[blog]](https://www.chaidiscovery.com/news/introducing-chai-2) | Minibinder & antibody design |
| RF3               | Accelerating biomolecular modeling with AtomWorks and RF3 [[preprint]](https://doi.org/10.1101/2025.08.14.670328) | 2025 | bioRxiv                  | [[code]](https://github.com/RosettaCommons/atomworks) | AtomWorks framework |
| Pearl               | Pearl: a foundation model for placing every atom in the right location [[preprint]](https://arxiv.org/abs/2510.24670) | 2025 | arXiv                  | / | Synthetic data, SO(3)-equivariance |
| SeedFold               | SeedFold: scaling biomolecular structure prediction [[preprint]](https://arxiv.org/abs/2512.24354) | 2025 | arXiv                  | [[project page]](https://seedfold.github.io/) | Synthetic data, linear triangular attention |
| Protenix-v1               | Protenix-v1: toward high-accuracy open-source biomolecular structure prediction [[preprint]](https://doi.org/10.64898/2026.02.05.703733) | 2026 | bioRxiv                  | [[code]](https://github.com/bytedance/Protenix) [[server]](https://protenix-server.com/) | Open-source structure prediction model with superior performance to AlphaFold3 |
| Protenix-v2               | Protenix-v2: Broadening the Reach of Structure Prediction and Biomolecular Design [[preprint]](https://doi.org/10.64898/2026.04.10.717613) | 2026 | bioRxiv                  | [[code]](https://github.com/bytedance/Protenix) [[server]](https://protenix-server.com/) | Structure prediction and biomolecular design |
| IsoDDE               | Accurate predictions of novel biomolecular interactions with IsoDDE [[technical report]](https://doi.org/10.5281/zenodo.19699685) | 2026 | Technical Report | / | State-of-the-art performance in challenging structure modeling and affinity prediction, implementation details not disclosed |
| OpenFold3               | OpenFold3-preview2 technical report [[technical report]](https://portal.openfold.omsf.io/reports/of3p2_technical_report.pdf) | 2026 | Technical Report | [[code]](https://github.com/aqlaboratory/openfold-3) | MGnify 13M-sequence distillation dataset openly available |
| AnewSampling               | Learning the all-atom equilibrium distribution of biomolecular interactions at scale [[preprint]](https://doi.org/10.64898/2026.03.10.710952) | 2026 | bioRxiv                  | [[project page]](https://anewbt.com/research-anewsampling/) | Dynamic equilibrium sampling of protein–ligand complexes |


## 4. Structure-Based *De Novo* Drug Design with Deep Generative Models <a name="4"></a>

### 4.1 Structure-based de novo drug design methods <a name="4.1"></a>

| Name            | Paper Title                                                  | Year | **Venue**          | Resources                                                    | Notes                                            |
| --------------- | ------------------------------------------------------------ | ---- | ------------------ | ------------------------------------------------------------ | ------------------------------------------------ |
| /               | A 3D generative model for structure-based drug design [[paper]](https://proceedings.neurips.cc/paper/2021/hash/314450613369e0ee72d0da7f6fee773c-Abstract.html) | 2021 | NeurIPS            | [[code]](https://github.com/luost26/3D-Generative-SBDD)      | Autoregressive                                   |
| DeepLigBuilder  | Structure-based de novo drug design using 3D deep generative models [[paper]](https://doi.org/10.1039/D1SC04444C) | 2021 | Chem. Sci.         | [[data]](https://disk.pku.edu.cn/link/FA7AD4D5E57C4134EC7869225DB4063F) | Autoregressive                                   |
| GraphBP         | Generating 3D molecules for target protein binding [[paper]](https://proceedings.mlr.press/v162/liu22m.html) | 2022 | ICML               | [[code]](https://github.com/divelab/GraphBP)                 | Autoregressive                                   |
| Pocket2Mol      | Pocket2Mol: efficient molecular sampling based on 3D protein pockets [[paper]](https://proceedings.mlr.press/v162/peng22b.html) | 2022 | ICML               | [[code]](https://github.com/pengxingang/Pocket2Mol)          | Autoregressive                                   |
| DeepLigBuilder+ | Synthesis-driven design of 3D molecules for structure-based drug discovery using geometric transformers [[preprint]](https://arxiv.org/abs/2301.00167) | 2022 | arXiv              | /                                                            | Autoregressive, synthon, pharmacophore           |
| TargetDiff      | 3D equivariant diffusion for target-aware molecule generation and affinity prediction [[paper]](https://openreview.net/forum?id=kJqXEPXMsE0) | 2023 | ICLR               | [[code]](https://github.com/guanjq/targetdiff)               | Non-autoregressive                               |
| FLAG            | Molecule generation for target protein binding with structural motifs [[paper]](https://openreview.net/forum?id=Rq13idF0F73) | 2023 | ICLR               | [[code]](https://github.com/zaixizhang/FLAG)                 | Autoregressive, fragment                         |
| ResGen          | ResGen is a pocket-aware 3D molecular generation model based on parallel multiscale modelling [[paper]](https://doi.org/10.1038/s42256-023-00712-7) | 2023 | Nat. Mach. Intell. | [[code]](https://github.com/OdinZhang/ResGen)                | Autoregressive                                   |
| DrugGPS         | Learning subpocket prototypes for generalizable structure-based drug design [[paper]](https://proceedings.mlr.press/v202/zhang23z.html) | 2023 | ICML               | [[code]](https://github.com/zaixizhang/DrugGPS_ICML23)       | Autoregressive, fragment                         |
| DecompDiff      | DecompDiff: diffusion models with decomposed priors for structure-based drug design [[paper]](https://proceedings.mlr.press/v202/guan23a.html) | 2023 | ICML               | [[code]](https://github.com/bytedance/DecompDiff)            | Non-autoregressive, fragment                     |
| D3FG            | Functional-group-based diffusion for pocket-specific molecule generation and elaboration [[paper]](https://proceedings.neurips.cc/paper_files/paper/2023/hash/6cdd4ce9330025967dd1ed0bed3010f5-Abstract-Conference.html) | 2023 | NeurIPS            | [[code]](https://github.com/EDAPINENUT/CBGBench)             | Non-autoregressive, fragment                     |
| SurfGen         | Learning on topological surface and geometric structure for 3D molecular generation [[paper]](https://doi.org/10.1038/s43588-023-00530-2) | 2023 | Nat. Comput. Sci.  | [[code]](https://github.com/OdinZhang/SurfGen)               | Autoregressive, explicit pocket surface features |
| PocketFlow      | PocketFlow is a data-and-knowledge-driven structure-based molecular generative model [[paper]](https://doi.org/10.1038/s42256-024-00808-8) | 2024 | Nat. Mach. Intell. | [[code]](https://github.com/Saoge123/PocketFlow)             | Autoregressive                                   |
| DiffSBDD        | Structure-based drug design with equivariant diffusion models [[paper]](https://doi.org/10.1038/s43588-024-00737-x) | 2024 | Nat. Comput. Sci.  | [[code]](https://github.com/arneschneuing/DiffSBDD)          | Non-autoregressive                               |
| PMDM            | A dual diffusion model enables 3D molecule generation and lead optimization based on target pockets [[paper]](https://doi.org/10.1038/s41467-024-46569-1) | 2024 | Nat. Commun.       | [[code]](https://github.com/Layne-Huang/PMDM)                | Non-autoregressive                               |
| MolCRAFT        | MolCRAFT: structure-based drug design in continuous parameter space [[paper]](https://proceedings.mlr.press/v235/qu24a.html) | 2024 | ICML               | [[code]](https://github.com/AlgoMole/MolCRAFT)               | Non-autoregressive                               |
| Lingo3DMol      | Generation of 3D molecules in pockets via a language model [[paper]](https://doi.org/10.1038/s42256-023-00775-6) | 2024 | Nat. Mach. Intell. | [[code]](https://github.com/stonewiseAIDrugDesign/Lingo3DMol) | Autoregressive LM, fragment                      |
| KGDiff          | KGDiff: towards explainable target-aware molecule generation with knowledge guidance [[paper]](https://doi.org/10.1093/bib/bbad435) | 2024 | Brief. Bioinform.  | [[code]](https://github.com/CMACH508/KGDiff)                 | Non-autoregressive, Vina scoring guidance        |
| FlexSBDD        | FlexSBDD: structure-based drug design with flexible protein modeling [[paper]](https://openreview.net/forum?id=4AB54h21qG) | 2024 | NeurIPS            | /                                                            | Non-autoregressive, dynamic pocket               |
| DynamicFlow     | Integrating protein dynamics into structure-based drug design via full-atom stochastic flows [[paper]](https://openreview.net/forum?id=9qS3HzSDNv) | 2025 | ICLR               | /                                                            | Non-autoregressive, dynamic pocket               |
| DrugFlow & FlexFlow     | Multi-domain distribution learning for de novo drug design [[paper]](https://openreview.net/forum?id=g3VCIM94ke) | 2025 | ICLR               | [[code]](https://github.com/LPDI-EPFL/DrugFlow)                                                            | Non-autoregressive, flexible side-chains               |
| RxnFlow     | Generative flows on synthetic pathway for drug design [[paper]](https://openreview.net/forum?id=pB1XSj2y4X) | 2025 | ICLR               | [[code]](https://github.com/SeonghwanSeo/RxnFlow)                                                            | Autoregressive, codesigns synthetic routes               |
| SynGFN     | SynGFN: learning across chemical space with generative flow-based molecular discovery [[paper]](https://doi.org/10.1038/s43588-025-00902-w) | 2025 | Nat. Comput. Sci.               | [[code]](https://github.com/ChemloverYuchen/SynGFN)                                                            | Autoregressive, codesigns synthetic routes             |
| PocketXMol        | Unified modeling of 3D molecular generation via atomic interactions with PocketXMol [[paper]](https://doi.org/10.1016/j.cell.2026.01.003) | 2026 | Cell            | [[code]](https://github.com/pengxingang/PocketXMol)                                                            | Non-autoregressive, atom-level generative foundation model               |
| AnewOmni     | Programming biomolecular interactions with all-atom generative model [[preprint]](https://doi.org/10.64898/2026.03.12.711044) | 2026 | bioRxiv               | [[project page]](https://anewbt.com/research-anewomni/)                                                            | Non-autoregressive, all-atom latent diffusion, unified generation of peptides, antibodies, and small molecules |

### 4.2 Ligand-based design and lead optimization methods <a name="4.2"></a>

| Name         | Paper Title                                                  | Year | **Venue**          | Resources                                               | Notes                                                        |
| ------------ | ------------------------------------------------------------ | ---- | ------------------ | ------------------------------------------------------- | ------------------------------------------------------------ |
| Delinker     | Deep generative models for 3D linker design [[paper]](https://doi.org/10.1021/acs.jcim.9b01120) | 2020 | JCIM               | [[code]](https://github.com/oxpig/DeLinker)             | Linker                                                       |
| DEVELOP      | Deep generative design with 3D pharmacophoric constraints [[paper]](https://doi.org/10.1039/D1SC02436A) | 2021 | Chem. Sci.         | [[code]](https://github.com/oxpig/DEVELOP)              | Linker, pharmacophore                                        |
| DeepHop      | Deep scaffold hopping with multimodal transformer neural networks [[paper]](https://doi.org/10.1186/s13321-021-00565-5) | 2021 | J. Cheminform.     | [[code]](https://github.com/prokia/deepHops)            | Scaffold hopping                                             |
| DRLinker     | DRLinker: deep reinforcement learning for optimization in fragment linking design [[paper]](https://doi.org/10.1021/acs.jcim.2c00982) | 2022 | JCIM               | [[code]](https://github.com/biomed-AI/DRlinker)         | Linker                                                       |
| SILVR        | SILVR: guided diffusion for molecule generation [[paper]](https://doi.org/10.1021/acs.jcim.3c00667) | 2023 | JCIM               | [[code]](https://github.com/meyresearch/SILVR)          | Training-free                                                |
| FFLOM        | FFLOM: a flow-based autoregressive model for fragment-to-lead optimization [[paper]](https://doi.org/10.1021/acs.jmedchem.3c01009) | 2023 | JMC                | [[code]](https://github.com/JenniferKim09/FFLOM)        | Linker                                                       |
| LinkerNet    | LinkerNet: fragment poses and linker co-design with 3D equivariant diffusion [[paper]](https://openreview.net/forum?id=6EaLIw3W7c) | 2023 | NeurIPS            | [[code]](https://github.com/guanjq/LinkerNet)           | Linker                                                       |
| PGMG         | A pharmacophore-guided deep learning approach for bioactive molecular generation [[paper]](https://doi.org/10.1038/s41467-023-41454-9) | 2023 | Nat. Commun.       | [[code]](https://github.com/CSUBioGroup/PGMG)           | Scaffold hopping, pharmacophore                              |
| DiffLinker   | Equivariant 3D-conditional diffusion model for molecular linker design [[paper]](https://doi.org/10.1038/s42256-024-00815-9) | 2024 | Nat. Mach. Intell. | [[code]](https://github.com/igashov/DiffLinker)         | Linker                                                       |
| ShEPhERD     | ShEPhERD: diffusing shape, electrostatics, and pharmacophores for bioisosteric drug design [[paper]](https://openreview.net/forum?id=KSLkFYHlYg) | 2025 | ICLR               | [[code]](https://github.com/coleygroup/shepherd-score)  | Bioisosteric lignd design, shape & electrostatics & pharmacophore |
| Delete       | Deep lead optimization enveloped in protein pocket and its application in designing potent and selective ligands targeting LTK protein [[paper]](https://doi.org/10.1038/s42256-025-00997-w) | 2025 | Nat. Mach. Intell. | [[code]](https://github.com/OdinZhang/Delete)           | /                                                            |
| TransPharmer | Accelerating discovery of bioactive ligands with pharmacophore-informed generative models [[paper]](https://doi.org/10.1038/s41467-025-56349-0) | 2025 | Nat. Commun.       | [[code]](https://github.com/iipharma/transpharmer-repo) | Scaffold elaboration, pharmacophore                          |
| PhoreGen | Pharmacophore-oriented 3D molecular generation toward efficient feature-customized drug discovery [[paper]](https://doi.org/10.1038/s43588-025-00850-5) | 2025 | Nat. Comput. Sci.       | [[code]](https://github.com/ppjian19/PhoreGen) | Pharmacophore                          |
| ED2Mol | Electron-density-informed effective and reliable de novo molecular design and optimization with ED2Mol [[paper]](https://doi.org/10.1038/s42256-025-01095-7) | 2025 | Nat. Mach. Intell.       | [[code]](https://github.com/pineappleK/ED2Mol) | Electron density                         |


### 4.3 Benchmarks, datasets, and tools <a name="4.3"></a>

| Name            | Paper Title                                                  | Year | **Venue**         | Resources                                                    | Notes              |
| --------------- | ------------------------------------------------------------ | ---- | ----------------- | ------------------------------------------------------------ | ------------------ |
| GuacaMol        | GuacaMol: benchmarking models for de novo molecular design [[paper]](https://doi.org/10.1021/acs.jcim.8b00839) | 2019 | JCIM              | [[code]](https://github.com/BenevolentAI/guacamol)           | Benchmark          |
| MOSES           | Molecular sets (MOSES): a benchmarking platform for molecular generation models [[paper]](https://doi.org/10.3389/fphar.2020.565644) | 2020 | Front. Pharmacol. | [[code]](https://github.com/molecularsets/moses)             | Benchmark          |
| CrossDocked2020 | Three-dimensional convolutional neural networks and a cross-docked data set for structure-based drug design [[paper]](https://doi.org/10.1021/acs.jcim.0c00411) | 2020 | JCIM              | [[data]](https://bits.csb.pitt.edu/files/crossdock2020/) [[instruction]](https://github.com/gnina/models/tree/master/data/CrossDocked2020) | Benchmark, dataset |
| POKMOL-3D       | How good are current pocket-based 3D generative models?: the benchmark set and evaluation of protein pocket-based 3D molecular generative models [[paper]](https://doi.org/10.1021/acs.jcim.4c01598) | 2024 | JCIM              | [[code]](https://github.com/haoyang9688/POKMOL3D)            | Benchmark          |
| Durian          | Durian: a comprehensive benchmark for structure-based 3D molecular generation [[paper]](https://doi.org/10.1021/acs.jcim.4c02232) | 2024 | JCIM              | [[code]](https://github.com/19990210nd/Durian)               | Benchmark          |
| CBGBench        | CBGBench: fill in the blank of protein-molecule complex binding graph [[paper]](https://openreview.net/forum?id=mOpNrrV2zH) | 2025 | ICLR              | [[code]](https://github.com/EDAPINENUT/CBGBench)             | Benchmark          |
| MolGenBench        | Benchmarking real-world applicability of molecular generative models from de novo design to lead optimization with MolGenBench [[preprint]](https://doi.org/10.1101/2025.11.03.686215) | 2025 | bioRxiv              | [[code]](https://github.com/CAODH/MolGenBench)             | Benchmark          |
| TarPass        | Revisiting target-aware de novo molecular generation with TarPass: between rational design and Texas Sharpshooter [[paper]](https://doi.org/10.1002/advs.75411) | 2026 | Advanced Science | [[code]](https://github.com/sorui-qin/TarPass)             | Benchmark          |

## 5. Sequence-Based Methods for Drug Discovery <a name="5"></a>

| Name               | Paper Title                                                  | Year | **Venue**          | Resources                                                  | Notes                                           |
| ------------------ | ------------------------------------------------------------ | ---- | ------------------ | ---------------------------------------------------------- | ----------------------------------------------- |
| DrugBAN            | Interpretable bilinear attention network with domain adaptation improves drug--target prediction [[paper]](https://doi.org/10.1038/s42256-022-00605-1) | 2023 | Nat. Mach. Intell. | [[code]](https://github.com/peizhenbai/DrugBAN)            | DTI prediction                                  |
| DeepTarget         | Deep generative model for drug design from protein target sequence [[paper]](https://doi.org/10.1186/s13321-023-00702-2) | 2023 | J. Cheminform.     | [[code]](https://github.com/viko-3/TargetGAN)              | Target-conditioned generation                   |
| ConPLex            | Contrastive learning in protein language space predicts interactions between drugs and protein targets [[paper]](https://doi.org/10.1073/pnas.2220778120) | 2023 | PNAS               | [[code]](https://github.com/samsledje/ConPLex)             | ULVS                                            |
| TransformerCPI 2.0 | Sequence-based drug design as a concept in computational drug design [[paper]](https://doi.org/10.1038/s41467-023-39856-w) | 2023 | Nat. Commun.       | [[code]](https://github.com/myzhengSIMM/transformerCPI2.0) | DTI prediction                                  |
| CogMol             | Accelerating drug target inhibitor discovery with a deep generative foundation model [[paper]](https://doi.org/10.1126/sciadv.adg7865) | 2023 | Sci. Adv.          | [[code]](https://zenodo.org/records/7863805)               | Target-conditioned generation                   |
| AI-Bind            | Improving the generalizability of protein-ligand binding predictions with AI-Bind [[paper]](https://doi.org/10.1038/s41467-023-37572-z) | 2023 | Nat. Commun.       | [[code]](https://github.com/Barabasi-Lab/AI-Bind)          | DTI prediction, interaction network             |
| DRAGONFLY          | Prospective de novo drug design with deep interactome learning [[paper]](https://doi.org/10.1038/s41467-024-47613-w) | 2024 | Nat. Commun.       | [[code]](https://github.com/atzkenneth/dragonfly_gen)      | DTI prediction, generation, interaction network |
| PSICHIC            | Physicochemical graph neural network for learning protein-ligand interaction fingerprints from sequence data [[paper]](https://doi.org/10.1038/s42256-024-00847-1) | 2024 | Nat. Mach. Intell. | [[code]](https://github.com/huankoh/PSICHIC)               | Interaction fingerprints                        |
| DeepBlock          | A deep learning approach for rational ligand generation with toxicity control via reactive building blocks [[paper]](https://doi.org/10.1038/s43588-024-00718-0) | 2024 | Nat. Comput. Sci.  | [[code]](https://github.com/BioChemAI/DeepBlock)           | Target-conditioned generation, fragment         |
| LaMGen             | LaMGen: LLM-based 3D molecular generation for multi-target drug design [[paper]](https://doi.org/10.1038/s41467-026-71737-w) | 2026 | Nat. Commun.       | [[code]](https://github.com/cholin01/LaMGen) [[software]](https://doi.org/10.5281/zenodo.19050125) | Multi-target 3D generation from protein sequences |


# Contributing
We welcome contributions from the community. To contribute, please fork this repository, apply your modifications, and open a pull request to the main branch.

# License
This repository is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

# Citation
If you find this repository useful for your research, please consider citing the following paper:

```bibtex
@article{wang2025modeling,
  title={Modeling protein–ligand interactions for drug discovery in the era of deep learning},
  author={Wang, Yuzhe and Li, Yibo and Chen, Jiaxiao and Lai, Luhua},
  journal={Chemical Society Reviews},
  year={2025},
  publisher={The Royal Society of Chemistry},
  doi={10.1039/D5CS00415B}
}
```
