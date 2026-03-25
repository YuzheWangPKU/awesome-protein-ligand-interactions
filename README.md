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
| /                | Simulating protein-ligand binding with neural network potentials | 2020 | Chem. Sci. | /                                                            | NNP/MM      |
| qmlify           | Towards chemical accuracy for alchemical free energy calculations with hybrid physics-based machine learning/molecular mechanics potentials | 2020 | bioRxiv    | [[code]](https://github.com/choderalab/qmlify)               | NNP/MM      |
| /                | NNP/MM: accelerating molecular dynamics simulations with machine learning potentials and molecular mechanics | 2023 | JCIM       | [[code]](https://github.com/openmm/nnpops)                   | NNP/MM      |
| /                | Enhancing protein-ligand binding affinity predictions using neural network potentials | 2024 | JCIM       | [[data]](https://github.com/compsciencelab/ATM_benchmark/tree/main/ATM_With_NNPs) | NNP/MM      |
| AP-Net           | A physics-aware neural network for protein-ligand interactions with quantum chemical accuracy | 2024 | Chem. Sci. | [[code]](https://github.com/zachglick/apnet)                 | Native MLFF |
| espaloma-0.3     | Machine-learned molecular mechanics force fields from large-scale quantum chemical data | 2024 | Chem. Sci. | [[code]](https://github.com/choderalab/refit-espaloma)       | Native MLFF |
| QuantumBind-RBFE | QuantumBind-RBFE: accurate relative binding free energy calculations using neural network potentials | 2025 | JCIM       | [[code]](https://github.com/Acellera/quantumbind_rbfe)       | NNP/MM      |

### 1.2 Learning protein-ligand interactions from MD trajectories <a name="1.2"></a>

| Name       | Paper Title                                                  | Year | **Venue**    | Resources                                        |
| ---------- | ------------------------------------------------------------ | ---- | ------------ | ------------------------------------------------ |
| NRI-MD     | Neural relational inference to learn long-range allosteric interactions in proteins from molecular dynamics simulations | 2022 | Nat. Commun. | [[code]](https://github.com/juexinwang/NRI-MD)   |
| ProtMD     | Pre-training of equivariant graph matching networks with conformation flexibility for drug binding | 2022 | Adv. Sci.    | [[code]](https://github.com/smiles724/ProtMD)    |
| Dynaformer | From static to dynamic structures: improving binding affinity prediction with graph-based deep learning | 2024 | Adv. Sci.    | [[code]](https://github.com/Minys233/Dynaformer) |
| MDbind | Spatio-temporal learning from molecular dynamics simulations for protein–ligand binding affinity prediction | 2025 | Bioinfomatics    | [[code]](https://github.com/ICOA-SBC/MD_DL_BA) |

### 1.3 Generative modeling of MD trajectories for protein-ligand complexes <a name="1.3"></a>

| Name       | Paper Title                                                  | Year | **Venue**    | Resources                                        |
| ---------- | ------------------------------------------------------------ | ---- | ------------ | ------------------------------------------------ |
| BioMD     | BioMD: all-atom generative model for biomolecular dynamics simulation | 2026 | ICLR | /   |
| BioKinema     | Physically grounded generative modeling of all-atom biomolecular dynamics | 2026 | bioRxiv | [[code]](https://github.com/IDEA-XL/BioKinema)   |

### 1.4 Benchmarks, datasets, and tools <a name="1.4"></a>

| Name   | Paper Title                                                  | Year | Venue             | Resources                                                    | Notes                             |
| ------ | ------------------------------------------------------------ | ---- | ----------------- | ------------------------------------------------------------ | --------------------------------- |
| SPICE  | SPICE, a dataset of drug-like molecules and peptides for training machine learning potentials | 2023 | Sci. Data         | [[data]](https://zenodo.org/records/7338495) [[code]](https://github.com/openmm/spice-dataset) | QM calculations                   |
| MISATO | MISATO: machine learning dataset of protein-ligand complexes for structure-based drug discovery | 2024 | Nat. Comput. Sci. | [[data]](https://zenodo.org/records/7711953) [[code]](https://github.com/t7morgen/misato-dataset/) | QM calculations & MD trajectories |
| PLAS-20k | PLAS-20k: extended dataset of protein-ligand affinities from MD simulations for machine learning applications | 2024 | Sci. Data | [[data]](https://doi.org/10.6084/m9.figshare.c.6742521.v2) | MD trajectories & MMPBSA calculations |
| OMol25 | The open molecules 2025 (OMol25) dataset, evaluations, and models | 2025 | arXiv             | [[data]](https://huggingface.co/facebook/OMol25) [[code]](https://github.com/facebookresearch/fairchem) [[blog]](https://ai.meta.com/blog/meta-fair-science-new-open-source-releases/) | QM calculations                   |
| DD-13M | Enhanced sampling, public dataset and generative model for drug-protein dissociation dynamics | 2025 | arXiv             | [[data]](https://huggingface.co/SZBL-IDEA) [[webpage]](https://aimm.szbl.ac.cn/database/ddd/#/home)                   | MD trajectories                   |
| qcMol | qcMol: a large-scale dataset of 1.2 million molecules with high-quality quantum chemical annotations for molecular representation learning | 2025 | bioRxiv             | [[webpage]](https://structpred.life.tsinghua.edu.cn/qcmol/)                   | QM calculations                   |
| BMS25 | Biomolecular multiscale simulation (BMS25) dataset to train neural network potentials for QM/MM settings with electrostatic embedding | 2025 | ChemRxiv             | [[data]](https://doi.org/10.3929/ethz-c-000788484)                   | QM/MM calculations                   |

## 2. Deep Learning-Enhanced Molecular Docking and Virtual Screening <a name="2"></a>

### 2.1 Deep learning-based docking methods <a name="2.1"></a>

| Name        | Paper Title                                                  | Year | **Venue**          | Resources                                               | Notes               |
| ----------- | ------------------------------------------------------------ | ---- | ------------------ | ------------------------------------------------------- | ------------------- |
| DeepDock    | A geometric deep learning approach to predict binding conformations of bioactive molecules | 2021 | Nat. Mach. Intell. | [[code]](https://github.com/OptiMaL-PSE-Lab/DeepDock)   | Rigid receptor      |
| TankBind    | TankBind: trigonometry-aware neural networks for drug-protein binding structure prediction | 2022 | NeurIPS            | [[code]](https://github.com/luwei0917/TankBind)         | Rigid receptor      |
| EquiBind    | EquiBind: geometric deep learning for drug binding structure prediction | 2022 | ICML               | [[code]](https://github.com/HannesStark/EquiBind)       | Rigid receptor      |
| E3Bind      | E3Bind: an end-to-end equivariant network for protein-ligand docking | 2022 | ICLR               | /                                                       | Rigid receptor      |
| DiffDock    | DiffDock: diffusion steps, twists, and turns for molecular docking | 2022 | ICLR               | [[code]](https://github.com/gcorso/DiffDock)            | Rigid receptor      |
| Uni-Mol     | Uni-Mol: a universal 3D molecular representation learning framework | 2023 | ICLR               | [[code]](https://github.com/deepmodeling/Uni-Mol)       | Rigid receptor      |
| KarmaDock   | Efficient and accurate large library ligand docking with KarmaDock | 2023 | Nat. Comput. Sci.  | [[code]](https://github.com/schrojunzhang/KarmaDock)    | Rigid receptor      |
| FABind      | FABind: fast and accurate protein-ligand binding             | 2023 | NeurIPS            | [[code]](https://github.com/QizhiPei/FABind)            | Rigid receptor      |
| FlexPose    | Equivariant flexible modeling of the protein-ligand binding pose with geometric deep learning | 2023 | JCTC               | [[code]](https://github.com/tiejundong/FlexPose)        | Side-chain flexible |
| CarsiDock   | CarsiDock: a deep learning paradigm for accurate protein-ligand docking and screening based on large-scale pre-training | 2024 | Chem. Sci.         | [[code]](https://github.com/carbonsilicon-ai/CarsiDock) | Rigid receptor      |
| DeltaDock   | DeltaDock: a unified framework for accurate, efficient, and physically reliable molecular docking | 2024 | NeurIPS            | [[code]](https://github.com/jiaxianyan/DeltaDock)       | Rigid receptor      |
| DiffBindFR  | DiffBindFR: an SE (3) equivariant network for flexible protein-ligand docking | 2024 | Chem. Sci.         | [[code]](https://github.com/HBioquant/DiffBindFR)       | Side-chain flexible |
| DynamicBind | DynamicBind: predicting ligand-specific protein-ligand complex structure with a deep equivariant generative model | 2024 | Nat. Commun.       | [[code]](https://github.com/luwei0917/DynamicBind)      | Fully flexible      |
| PackDock | Flexible protein–ligand docking with diffusion-based side-chain packing | 2025 | PNAS       | [[code]](https://github.com/Zhang-Runze/PackDock)      | Side-chain flexible      |

### 2.2 Deep learning scoring functions and binding affinity prediction <a name="2.2"></a>

| Name      | Paper Title                                                  | Year | **Venue**          | Resources                                         | Notes                     |
| --------- | ------------------------------------------------------------ | ---- | ------------------ | ------------------------------------------------- | ------------------------- |
| OnionNet  | OnionNet: a multiple-layer intermolecular-contact-based convolutional neural network for protein-ligand binding affinity prediction | 2019 | ACS Omega          | [[code]](https://github.com/zhenglz/onionnet)     | /                         |
| RTMScore  | Boosting protein-ligand binding pose prediction and virtual screening based on residue-atom distance likelihood potential and graph transformer | 2022 | JMC                | [[code]](https://github.com/sc8668/RTMScore)      | /                         |
| PIGNet    | PIGNet: a physics-informed deep learning model toward generalized drug-target interaction predictions | 2022 | Chem. Sci.         | [[code]](https://github.com/ACE-KAIST/PIGNet)     | Physical terms            |
| GenScore  | A generalized protein-ligand scoring framework with balanced scoring, docking, ranking and screening powers | 2023 | Chem. Sci.         | [[code]](https://github.com/sc8668/GenScore)      | /                         |
| PBCNet    | Computing the relative binding affinity of ligands based on a pairwise binding comparison network | 2023 | Nat. Comput. Sci.  | [[code]](https://doi.org/10.24433/CO.1095515.v2)  | Relative binding affinity |
| PLANET    | PLANET: a multi-objective graph neural network model for protein-ligand binding affinity prediction | 2023 | JCIM               | [[code]](https://github.com/ComputArtCMCG/PLANET) | Binding pose-free         |
| EquiScore | Generic protein-ligand interaction scoring by integrating physical prior knowledge and data augmentation modelling | 2024 | Nat. Mach. Intell. | [[code]](https://github.com/CAODH/EquiScore)      | Interaction fingerprints  |
| DeepRLI   | DeepRLI: a multi-objective framework for universal protein-ligand interaction prediction | 2025 | Digit. Discov.     | [[code]](https://github.com/fairydance/DeepRLI)   | /                         |
| PBCNet2.0 | Advancing ligand binding affinity prediction with cartesian tensor-based deep learning | 2025 | bioRxiv            | /                                                 | Relative binding affinity |
| BioScore | BioScore: a foundational scoring function for diverse biomolecular complexes | 2025 | arXiv            | /                                                 | A unified scoring function |


### 2.3 Deep learning-accelerated virtual screening <a name="2.3"></a>

| Name         | Paper Title                                                  | Year | **Venue**         | Resources                                                    | Notes             |
| ------------ | ------------------------------------------------------------ | ---- | ----------------- | ------------------------------------------------------------ | ----------------- |
| Deep Docking | Deep docking: a deep learning platform for augmentation of structure based drug discovery | 2020 | ACS Cent. Sci.    | [[code]](https://github.com/jamesgleave/DD_protocol)         | /                 |
| DrugCLIP     | DrugCLIP: contrastive protein-molecule representation learning for virtual screening | 2023 | NeurIPS           | [[code]](https://github.com/bowen-gao/DrugCLIP) [[project page]](https://drug-the-whole-genome.yanyanlan.com/) | CLIP architecture |
| OpenVS       | An artificial intelligence accelerated virtual screening platform for drug discovery | 2024 | Nat. Commun.      | [[code]](https://github.com/gfzhou/OpenVS)                   | Active learning   |
| /            | Rapid traversal of vast chemical space using machine learning-guided docking screens | 2025 | Nat. Comput. Sci. | [[code]](https://github.com/carlssonlab/conformalpredictor)  | /                 |
| FragmentScope            | FragmentScope - exploring the fragment space with learned surface representations | 2025 | bioRxiv | [[code]](https://github.com/LPDI-EPFL/FragmentScope)  | Surface-based fragment screening |


### 2.4 Benchmarks, datasets, and tools <a name="2.4"></a>

| Name               | Paper Title                                                  | Year | **Venue**          | Resources                                                    | Notes                              |
| ------------------ | ------------------------------------------------------------ | ---- | ------------------ | ------------------------------------------------------------ | ---------------------------------- |
| BindingDB          | BindingDB: a web-accessible database of experimentally determined protein-ligand binding affinities | 2007 | NAR                | [[webpage]](https://www.bindingdb.org/rwd/bind/index.jsp)    | Database                           |
| ChEMBL             | ChEMBL: a large-scale bioactivity database for drug discovery | 2012 | NAR                | [[webpage]](https://www.ebi.ac.uk/chembl/) [[code]](https://github.com/chembl/ChEMBL_Structure_Pipeline) | Database                           |
| ZINC Database      | ZINC 15-ligand discovery for everyone                        | 2015 | JCIM               | [[webpage]](https://cartblanche22.docking.org/)    | Database                           |
| Enamine REAL Space | Generating multibillion chemical space of readily accessible screening compounds | 2020 | iScience           | [[webpage]](https://enamine.net/compound-collections/real-compounds) | Make-on-demand database            |
| PoseBusters        | PoseBusters: AI-based docking methods fail to generate physically valid poses or generalise to novel sequences | 2024 | Chem. Sci.         | [[code]](https://github.com/maabuu/posebusters) [[doc]](https://posebusters.readthedocs.io/en/latest/) | Benchmark, pose quality check      |
| Leak Proof PDBBind | Leak proof PDBBind: a reorganized dataset of protein-ligand complexes for more generalizable binding affinity prediction | 2024 | arXiv              | [[code]](https://github.com/THGLab/LP-PDBBind)               | Dataset split                      |
| SPECTRA            | Evaluating generalizability of artificial intelligence models for molecular datasets | 2024 | Nat. Mach. Intell. | [[code]](https://github.com/mims-harvard/SPECTRA)            | Framework for model evaluation     |
| SAIR               | SAIR: enabling deep learning for protein-ligand lnteractions with a synthetic structural dataset | 2025 | bioRxiv            | [[data]](https://pub.sandboxaq.com/data/ic50-dataset)        | Database, AF3-predicted structures |
| QUID               | Extending quantum-mechanical benchmark accuracy to biological ligand-pocket interactions | 2025 | Nat. Commun.            | [[data]](https://github.com/MirelaVP/QUID)        | Benchmark, QM-level pocket-ligand interaction systems  |
| BindFlow               | BindFlow: a free, user-friendly pipeline for absolute binding free energy calculations using free energy perturbation or MM(PB/GB)SA | 2025 | bioRxiv            | [[code]](https://github.com/ale94mleon/BindFlow)        | Tool, FEP & MM(PB/GB)SA pipeline |

## 3. End-to-End Structural Modeling <a name="3"></a>

### 3.1 Protein structure prediction with applications in drug discovery <a name="3.1"></a>

| Paper Title                                                  | Year | **Venue**    | Resources                                                    | Notes                  |
| ------------------------------------------------------------ | ---- | ------------ | ------------------------------------------------------------ | ---------------------- |
| AlphaFold2 structures guide prospective ligand discovery     | 2024 | Science      | /                                                            | GPCR targets           |
| AlphaFold accelerates artificial intelligence powered drug discovery: efficient discovery of a novel CDK20 small molecule inhibitor | 2023 | Chem. Sci.   | [[PandaOmics]](https://pharma.ai/pandaomics) [[Chemistry42]](https://www.chemistry42.com/) | CDK20                  |
| AlphaFold accelerated discovery of psychotropic agonists targeting the trace amine-associated receptor 1 | 2024 | Sci. Adv.    | /                                                            | TAAR1                  |
| Virtual library docking for cannabinoid-1 receptor agonists with reduced side effects | 2025 | Nat. Commun. | [[data]](https://lsd.docking.org/targets/CB1R)               | Cannabinoid-1 receptor |

### 3.2 Generative modeling of protein–ligand complexes and equilibrium ensembles <a name="3.2"></a>

| Name                 | Paper Title                                                  | Year | **Venue**          | Resources                                                    | Notes                                     |
| -------------------- | ------------------------------------------------------------ | ---- | ------------------ | ------------------------------------------------------------ | ----------------------------------------- |
| NeuralPLexer         | State-specific protein-ligand complex structure prediction with a multiscale deep generative model | 2024 | Nat. Mach. Intell. | [[code]](https://github.com/zrqiao/NeuralPLexer)             | /                                         |
| RoseTTAFold All-Atom | Generalized biomolecular modeling and design with RoseTTAFold All-Atom | 2024 | Science            | [[code]](https://github.com/baker-laboratory/RoseTTAFold-All-Atom) | /                                         |
| AlphaFold3           | Accurate structure prediction of biomolecular interactions with AlphaFold 3 | 2024 | Nature             | [[code]](https://github.com/google-deepmind/alphafold3) [[server]](https://alphafoldserver.com/welcome) | Non-commercial usage                      |
| Umol                 | Structure prediction of protein-ligand complexes from sequence information with Umol | 2024 | Nat. Commun.       | [[code]](https://github.com/patrickbryant1/Umol)             | /                                         |
| Chai-1               | Chai-1: decoding the molecular interactions of life          | 2024 | bioRxiv            | [[code]](https://github.com/chaidiscovery/chai-lab) [[server]](https://lab.chaidiscovery.com/dashboard) | /                                         |
| Protenix             | Protenix-advancing structure prediction through a comprehensive AlphaFold3 reproduction | 2024 | bioRxiv            | [[code]](https://github.com/bytedance/Protenix) [[server]](https://protenix-server.com/login) | /                                         |
| Boltz-1              | Boltz-1: democratizing biomolecular interaction modeling     | 2024 | bioRxiv            | [[code]](https://github.com/jwohlwend/boltz) [[blog]](https://jclinic.mit.edu/boltz-1/) | /                                         |
| Boltz-2              | Towards accurate and efficient binding affinity prediction   | 2025 | bioRxiv            | [[code]](https://github.com/jwohlwend/boltz) [[design-code]](https://github.com/recursionpharma/synflownet-boltz) | FEP-level affinity prediciton             |
| GeoFlow-V2              | GeoFlow-V2: a unified atomic diffusion model for protein structure prediction and de novo design     | 2025 | bioRxiv            | [[server]](https://prot.design/) | Protein Binder & antibody design                                         |
| Chai-2               | Zero-shot antibody design in a 24-well plate                 | 2025 | bioRxiv                  | [[blog]](https://www.chaidiscovery.com/news/introducing-chai-2) | Minibinder & antibody design |
| RF3               | Accelerating biomolecular modeling with AtomWorks and RF3                 | 2025 | bioRxiv                  | [[code]](https://github.com/RosettaCommons/atomworks) | AtomWorks framework |
| Pearl               | Pearl: a foundation model for placing every atom in the right location                 | 2025 | arXiv                  | / | Synthetic data, SO(3)-equivariance |
| SeedFold               | SeedFold: scaling biomolecular structure prediction                 | 2026 | arXiv                  | [[project page]](https://seedfold.github.io/) | Synthetic data, linear triangular attention |
| Protenix-v1               | Protenix-v1: toward high-accuracy open-source biomolecular structure prediction                 | 2026 | bioRxiv                  | [[code]](https://github.com/bytedance/Protenix) [[server]](https://protenix-server.com/) | Open-source structure prediction model with superior performance to AlphaFold3 |
| IsoDDE               | Accurate predictions of novel biomolecular interactions with IsoDDE                 | 2026 | /                  | [[technical report]](https://storage.googleapis.com/isomorphiclabs-website-public-artifacts/isodde_technical_report.pdf) | State-of-the-art performance in challenging structure modeling and affinity prediction, implementation details not disclosed |
| OpenFold3               | OpenFold3-preview2 technical report                 | 2026 | /                  | [[code]](https://github.com/aqlaboratory/openfold-3) | MGnify 13M-sequence distillation dataset openly available |
| AnewSampling               | Learning the all-atom equilibrium distribution of biomolecular interactions at scale                 | 2026 | bioRxiv                  | [[project page]](https://anewbt.com/research-anewsampling/) | Dynamic equilibrium sampling of protein–ligand complexes |


## 4. Structure-Based *De Novo* Drug Design with Deep Generative Models <a name="4"></a>

### 4.1 Structure-based de novo drug design methods <a name="4.1"></a>

| Name            | Paper Title                                                  | Year | **Venue**          | Resources                                                    | Notes                                            |
| --------------- | ------------------------------------------------------------ | ---- | ------------------ | ------------------------------------------------------------ | ------------------------------------------------ |
| /               | A 3D generative model for structure-based drug design        | 2021 | NeurIPS            | [[code]](https://github.com/luost26/3D-Generative-SBDD)      | Autoregressive                                   |
| DeepLigBuilder  | Structure-based de novo drug design using 3D deep generative models | 2021 | Chem. Sci.         | [[data]](https://disk.pku.edu.cn/link/FA7AD4D5E57C4134EC7869225DB4063F) | Autoregressive                                   |
| GraphBP         | Generating 3D molecules for target protein binding           | 2022 | ICML               | [[code]](https://github.com/divelab/GraphBP)                 | Autoregressive                                   |
| Pocket2Mol      | Pocket2Mol: efficient molecular sampling based on 3D protein pockets | 2022 | ICML               | [[code]](https://github.com/pengxingang/Pocket2Mol)          | Autoregressive                                   |
| DeepLigBuilder+ | Synthesis-driven design of 3D molecules for structure-based drug discovery using geometric transformers | 2022 | arXiv              | /                                                            | Autoregressive, synthon, pharmacophore           |
| TargetDiff      | 3D equivariant diffusion for target-aware molecule generation and affinity prediction | 2023 | ICLR               | [[code]](https://github.com/guanjq/targetdiff)               | Non-autoregressive                               |
| FLAG            | Molecule generation for target protein binding with structural motifs | 2023 | ICLR               | [[code]](https://github.com/zaixizhang/FLAG)                 | Autoregressive, fragment                         |
| ResGen          | ResGen is a pocket-aware 3D molecular generation model based on parallel multiscale modelling | 2023 | Nat. Mach. Intell. | [[code]](https://github.com/OdinZhang/ResGen)                | Autoregressive                                   |
| DrugGPS         | Learning subpocket prototypes for generalizable structure-based drug design | 2023 | ICML               | [[code]](https://github.com/zaixizhang/DrugGPS_ICML23)       | Autoregressive, fragment                         |
| DecompDiff      | DecompDiff: diffusion models with decomposed priors for structure-based drug design | 2023 | ICML               | [[code]](https://github.com/bytedance/DecompDiff)            | Non-autoregressive, fragment                     |
| D3FG            | Functional-group-based diffusion for pocket-specific molecule generation and elaboration | 2023 | NeurIPS            | [[code]](https://github.com/EDAPINENUT/CBGBench)             | Non-autoregressive, fragment                     |
| SurfGen         | Learning on topological surface and geometric structure for 3D molecular generation | 2023 | Nat. Comput. Sci.  | [[code]](https://github.com/OdinZhang/SurfGen)               | Autoregressive, explicit pocket surface features |
| PocketFlow      | PocketFlow is a data-and-knowledge-driven structure-based molecular generative model | 2024 | Nat. Mach. Intell. | [[code]](https://github.com/Saoge123/PocketFlow)             | Autoregressive                                   |
| DiffSBDD        | Structure-based drug design with equivariant diffusion models | 2024 | Nat. Comput. Sci.  | [[code]](https://github.com/arneschneuing/DiffSBDD)          | Non-autoregressive                               |
| PMDM            | A dual diffusion model enables 3D molecule generation and lead optimization based on target pockets | 2024 | Nat. Commun.       | [[code]](https://github.com/Layne-Huang/PMDM)                | Non-autoregressive                               |
| MolCRAFT        | MolCRAFT: structure-based drug design in continuous parameter space | 2024 | ICML               | [[code]](https://github.com/AlgoMole/MolCRAFT)               | Non-autoregressive                               |
| Lingo3DMol      | Generation of 3D molecules in pockets via a language model   | 2024 | Nat. Mach. Intell. | [[code]](https://github.com/stonewiseAIDrugDesign/Lingo3DMol) | Autoregressive LM, fragment                      |
| KGDiff          | KGDiff: towards explainable target-aware molecule generation with knowledge guidance | 2024 | Brief. Bioinform.  | [[code]](https://github.com/CMACH508/KGDiff)                 | Non-autoregressive, Vina scoring guidance        |
| FlexSBDD        | FlexSBDD: structure-based drug design with flexible protein modeling | 2024 | NeurIPS            | /                                                            | Non-autoregressive, dynamic pocket               |
| DynamicFlow     | Integrating protein dynamics into structure-based drug design via full-atom stochastic flows | 2025 | ICLR               | /                                                            | Non-autoregressive, dynamic pocket               |
| DrugFlow & FlexFlow     | Multi-domain distribution learning for de novo drug design | 2025 | ICLR               | [[code]](https://github.com/LPDI-EPFL/DrugFlow)                                                            | Non-autoregressive, flexible side-chains               |
| RxnFlow     | Generative flows on synthetic pathway for drug design | 2025 | ICLR               | [[code]](https://github.com/SeonghwanSeo/RxnFlow)                                                            | Autoregressive, codesigns synthetic routes               |
| SynGFN     | SynGFN: learning across chemical space with generative flow-based molecular discovery | 2025 | Nat. Comput. Sci.               | [[code]](https://github.com/ChemloverYuchen/SynGFN)                                                            | Autoregressive, codesigns synthetic routes             |
| PocketXMol        | Unified modeling of 3D molecular generation via atomic interactions with PocketXMol | 2026 | Cell            | [[code]](https://github.com/pengxingang/PocketXMol)                                                            | Non-autoregressive, atom-level generative foundation model               |
| AnewOmni     | Programming biomolecular interactions with all-atom generative model | 2026 | bioRxiv               | [[project page]](https://anewbt.com/research-anewomni/)                                                            | Non-autoregressive, all-atom latent diffusion, unified generation of peptides, antibodies, and small molecules |

### 4.2 Ligand-based design and lead optimization methods <a name="4.2"></a>

| Name         | Paper Title                                                  | Year | **Venue**          | Resources                                               | Notes                                                        |
| ------------ | ------------------------------------------------------------ | ---- | ------------------ | ------------------------------------------------------- | ------------------------------------------------------------ |
| Delinker     | Deep generative models for 3D linker design                  | 2020 | JCIM               | [[code]](https://github.com/oxpig/DeLinker)             | Linker                                                       |
| DEVELOP      | Deep generative design with 3D pharmacophoric constraints    | 2021 | Chem. Sci.         | [[code]](https://github.com/oxpig/DEVELOP)              | Linker, pharmacophore                                        |
| DeepHop      | Deep scaffold hopping with multimodal transformer neural networks | 2021 | J. Cheminform.     | [[code]](https://github.com/prokia/deepHops)            | Scaffold hopping                                             |
| DRLinker     | DRLinker: deep reinforcement learning for optimization in fragment linking design | 2022 | JCIM               | [[code]](https://github.com/biomed-AI/DRlinker)         | Linker                                                       |
| SILVR        | SILVR: guided diffusion for molecule generation              | 2023 | JCIM               | [[code]](https://github.com/meyresearch/SILVR)          | Training-free                                                |
| FFLOM        | FFLOM: a flow-based autoregressive model for fragment-to-lead optimization | 2023 | JMC                | [[code]](https://github.com/JenniferKim09/FFLOM)        | Linker                                                       |
| LinkerNet    | LinkerNet: fragment poses and linker co-design with 3D equivariant diffusion | 2023 | NeurIPS            | [[code]](https://github.com/guanjq/LinkerNet)           | Linker                                                       |
| PGMG         | A pharmacophore-guided deep learning approach for bioactive molecular generation | 2023 | Nat. Commun.       | [[code]](https://github.com/CSUBioGroup/PGMG)           | Scaffold hopping, pharmacophore                              |
| DiffLinker   | Equivariant 3D-conditional diffusion model for molecular linker design | 2024 | Nat. Mach. Intell. | [[code]](https://github.com/igashov/DiffLinker)         | Linker                                                       |
| ShEPhERD     | ShEPhERD: diffusing shape, electrostatics, and pharmacophores for bioisosteric drug design | 2025 | ICLR               | [[code]](https://github.com/coleygroup/shepherd-score)  | Bioisosteric lignd design, shape & electrostatics & pharmacophore |
| Delete       | Deep lead optimization enveloped in protein pocket and its application in designing potent and selective ligands targeting LTK protein | 2025 | Nat. Mach. Intell. | [[code]](https://github.com/OdinZhang/Delete)           | /                                                            |
| TransPharmer | Accelerating discovery of bioactive ligands with pharmacophore-informed generative models | 2025 | Nat. Commun.       | [[code]](https://github.com/iipharma/transpharmer-repo) | Scaffold elaboration, pharmacophore                          |
| PhoreGen | Pharmacophore-oriented 3D molecular generation toward efficient feature-customized drug discovery | 2025 | Nat. Comput. Sci.       | [[code]](https://github.com/ppjian19/PhoreGen) | Pharmacophore                          |
| ED2Mol | Electron-density-informed effective and reliable de novo molecular design and optimization with ED2Mol | 2025 | Nat. Mach. Intell.       | [[code]](https://github.com/pineappleK/ED2Mol) | Electron density                         |


### 4.3 Benchmarks, datasets, and tools <a name="4.3"></a>

| Name            | Paper Title                                                  | Year | **Venue**         | Resources                                                    | Notes              |
| --------------- | ------------------------------------------------------------ | ---- | ----------------- | ------------------------------------------------------------ | ------------------ |
| GuacaMol        | GuacaMol: benchmarking models for de novo molecular design   | 2019 | JCIM              | [[code]](https://github.com/BenevolentAI/guacamol)           | Benchmark          |
| MOSES           | Molecular sets (MOSES): a benchmarking platform for molecular generation models | 2020 | Front. Pharmacol. | [[code]](https://github.com/molecularsets/moses)             | Benchmark          |
| CrossDocked2020 | Three-dimensional convolutional neural networks and a cross-docked data set for structure-based drug design | 2020 | JCIM              | [[data]](https://bits.csb.pitt.edu/files/crossdock2020/) [[instruction]](https://github.com/gnina/models/tree/master/data/CrossDocked2020) | Benchmark, dataset |
| POKMOL-3D       | How good are current pocket-based 3D generative models?: the benchmark set and evaluation of protein pocket-based 3D molecular generative models | 2024 | JCIM              | [[code]](https://github.com/haoyang9688/POKMOL3D)            | Benchmark          |
| Durian          | Durian: a comprehensive benchmark for structure-based 3D molecular generation | 2024 | JCIM              | [[code]](https://github.com/19990210nd/Durian)               | Benchmark          |
| CBGBench        | CBGBench: fill in the blank of protein-molecule complex binding graph | 2025 | ICLR              | [[code]](https://github.com/EDAPINENUT/CBGBench)             | Benchmark          |
| MolGenBench        | Benchmarking real-world applicability of molecular generative models from de novo design to lead optimization with MolGenBench | 2025 | bioRxiv              | [[code]](https://github.com/CAODH/MolGenBench)             | Benchmark          |
| TarPass        | Revisiting target-aware de novo molecular generation with TarPass: between rational design and Texas Sharpshooter | 2026 | ChemRxiv              | [[code]](https://github.com/sorui-qin/TarPass)             | Benchmark          |

## 5. Sequence-Based Methods for Drug Discovery <a name="5"></a>

| Name               | Paper Title                                                  | Year | **Venue**          | Resources                                                  | Notes                                           |
| ------------------ | ------------------------------------------------------------ | ---- | ------------------ | ---------------------------------------------------------- | ----------------------------------------------- |
| DrugBAN            | Interpretable bilinear attention network with domain adaptation improves drug--target prediction | 2023 | Nat. Mach. Intell. | [[code]](https://github.com/peizhenbai/DrugBAN)            | DTI prediction                                  |
| DeepTarget         | Deep generative model for drug design from protein target sequence | 2023 | J. Cheminform.     | [[code]](https://github.com/viko-3/TargetGAN)              | Target-conditioned generation                   |
| ConPLex            | Contrastive learning in protein language space predicts interactions between drugs and protein targets | 2023 | PNAS               | [[code]](https://github.com/samsledje/ConPLex)             | ULVS                                            |
| TransformerCPI 2.0 | Sequence-based drug design as a concept in computational drug design | 2023 | Nat. Commun.       | [[code]](https://github.com/myzhengSIMM/transformerCPI2.0) | DTI prediction                                  |
| CogMol             | Accelerating drug target inhibitor discovery with a deep generative foundation model | 2023 | Sci. Adv.          | [[code]](https://zenodo.org/records/7863805)               | Target-conditioned generation                   |
| AI-Bind            | Improving the generalizability of protein-ligand binding predictions with AI-Bind | 2023 | Nat. Commun.       | [[code]](https://github.com/Barabasi-Lab/AI-Bind)          | DTI prediction, interaction network             |
| DRAGONFLY          | Prospective de novo drug design with deep interactome learning | 2024 | Nat. Commun.       | [[code]](https://github.com/atzkenneth/dragonfly_gen)      | DTI prediction, generation, interaction network |
| PSICHIC            | Physicochemical graph neural network for learning protein-ligand interaction fingerprints from sequence data | 2024 | Nat. Mach. Intell. | [[code]](https://github.com/huankoh/PSICHIC)               | Interaction fingerprints                        |
| DeepBlock          | A deep learning approach for rational ligand generation with toxicity control via reactive building blocks | 2024 | Nat. Comput. Sci.  | [[code]](https://github.com/BioChemAI/DeepBlock)           | Target-conditioned generation, fragment         |


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
