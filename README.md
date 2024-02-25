# Medical Image Analyzer(MIA) Project
- Analyzing Chest Radiography(CR) images utilizing Pytorch and OpenCV, with insights from relevant research papers.

## Project Directory Layout

```bash
< PROJECT ROOT >
   |
   |-- Analysis_Visualization/                    # Drawing initial insights from data
   |    |
   |    |-- exploratory.py                        # Exploratory data analysis
   |    |-- img_scan.py                           # Scanning images with openCV
   |    |-- test_train_make.py                    # Creating the test train dataset
   |
   |-- Data/                                      # Data directory
   |    |
   |    |-- img/                                  # Holds the CXR images in png format  
   |    |
   |    |-- sliced.csv                            # Cleaned and processed from tsv
   |    |-- test_dataset.csv                      # Test dataset
   |    |-- train_dataset.csv                     # Train dataset
   |
   |-- Deep_Learning/                             # Deep Learning directory
   |    |
   |    |-- torch_train.py                        # Pytorch model creation
   |
   |-- Research Papers/                           # Research directory
   |    |
   |    |-- AnalysisReport.md                     # Insights from Research papers
   |
   |-- README.md                                  # Standard readme documentation
   |-- requirements.txt                           # Development modules
   |
   |-- ************************************************************************
```

<br />

## Dataset Description
- BIMCV-COVID19+ dataset is a large dataset with chest X-ray images CXR (CR, DX) and computed tomography (CT) imaging of COVID-19 patients along with their radiographic findings, pathologies, polymerase chain reaction (PCR), immunoglobulin G (IgG) and immunoglobulin M (IgM) diagnostic antibody tests and radiographic reports from Medical Imaging Databank in Valencian Region Medical Image Bank (BIMCV).
- The findings are mapped onto standard Unified Medical Language System (UMLS) terminology and they cover a wide spectrum of thoracic entities, contrasting with the much more reduced number of entities annotated in previous datasets.
- Images are stored in high resolution and entities are localized with anatomical labels in a Medical Imaging Data Structure (MIDS) format.
- In addition, 23 images were annotated by a team of expert radiologists to include semantic segmentation of radiographic findings.
- Moreover, extensive information is provided, including the patient’s demographic information, type of projection and acquisition parameters for the imaging study, among others.
- These iterations of the database include 21342 CR, 34829 DX and 7918 CT studies.
- Link: https://github.com/BIMCV-CSUSP/BIMCV-COVID-19
## Insights and Conclusions

- AI, CAD, DL systems alone seem to have a higher Sensitivity but lower Specificity than Raidologists

- General Practitioners benefit most from CADs

- CADs lower Turn around Time and boost overall accuracy when used by doctors

- Explainable AI and Causal Inference will need to be worked on as DL systems are usually black boxes

- Could add different parameters to model such as biomarkers, NLR (neutrophil-to-lymphocyte ratio), Basic Fibroblast Growth Factor (bFGF), Insulin-like Growth Factor (IGF-R), age, blood pH, to boost accuracy

## Research Citations

An, J. Y., Hwang, E. J., Nam, G., Lee, S. H., Park, C. M., Goo, J. M., & Choi, Y. R. (2024). Artificial Intelligence for assessment of endotracheal tube position on chest radiographs: Validation in patients from two institutions. American Journal of Roentgenology, 222(1). https://doi.org/10.2214/ajr.23.29769 

Shin, H. J., Kim, M. H., Son, N., Han, K., Kim, M. J., Kim, Y. C., Park, Y. S., Lee, E. H., & Kyong, T. (2023). Clinical implication and prognostic value of Artificial-Intelligence-Based results of chest radiographs for assessing clinical outcomes of COVID-19 patients. Diagnostics, 13(12), 2090. https://doi.org/10.3390/diagnostics13122090

Hwang, E. J., Kim, K. B., Kim, J. Y., Lim, J., Nam, J. G., Choi, H., Kim, H., Yoon, S. H., Goo, J. M., & Park, C. M. (2021). COVID-19 pneumonia on chest X-rays: Performance of a deep learning-based computer-aided detection system. PLOS ONE, 16(6), e0252440. https://doi.org/10.1371/journal.pone.0252440

Jang, S. B., Lee, S. H., Lee, D., Park, S., Kim, J. K., Cho, J. W., Cho, J., Kim, K. B., Park, B., Park, J., & Lim, J. (2020). Deep-learning algorithms for the interpretation of chest radiographs to aid in the triage of COVID-19 patients: A multicenter retrospective study. PLOS ONE, 15(11), e0242759. https://doi.org/10.1371/journal.pone.0242759

Pan, Y., Chen, Q., Chen, T., Wang, H., Zhu, X., Fang, Z., & Lü, Y. (2019). Evaluation of a computer-aided method for measuring the Cobb angle on chest X-rays. European Spine Journal, 28(12), 3035–3043. https://doi.org/10.1007/s00586-019-06115-w

Hong, S., Hwang, E. J., Kim, S., Song, J., Lee, T., Jo, G. D., Choi, Y., Park, C. M., & Goo, J. M. (2023). Methods of Visualizing the results of an Artificial-Intelligence-Based Computer-Aided Detection System for chest radiographs: Effect on the diagnostic performance of radiologists. Diagnostics, 13(6), 1089. https://doi.org/10.3390/diagnostics13061089

Lee, J. H., Ahn, J. S., Chung, M. J., Jeong, Y. J., Kim, J. H., Lim, J., Kim, J. Y., Kim, Y. J., Lee, J. E., & Kim, E. Y. (2022). Development and validation of a Multimodal-Based Prognosis and Intervention Prediction model for COVID-19 patients in a multicenter cohort. Sensors, 22(13), 5007. https://doi.org/10.3390/s22135007

Hwang, E. J., Kim, H., Yoon, S. H., Goo, J. M., & Park, C. M. (2020). Implementation of a Deep Learning-Based Computer-Aided detection system for the interpretation of chest radiographs in patients suspected for COVID-19. Korean Journal of Radiology, 21(10), 1150. https://doi.org/10.3348/kjr.2020.0536

## Libraries and Frameworks Used
Pytorch | SKlearn | Statsmodels | Pandas | OpenCV 

#### To-be added
GCP | Statistics | Multithreading | Causal Inference | PGMs | Explainable AI | Prefect
AUC | Specificity | Sensitivity | NPV | PPV | Confusion Matrix
