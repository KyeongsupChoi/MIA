# Alpha_AIAssessmentEndotrachealTubePositionChestRadiographs.pdf

## Summary

- Timely and accurate interpretation of chest radiographs obtained to evaluate endotracheal tube (ETT) position is important for facilitating prompt adjustment if needed.

- The purpose of this study was to evaluate the performance of a (DL)-based artificial intelligence (AI) system for detecting ETT presence and position on chest radiographs.

- This retrospective study included 539 chest radiographs obtained immediately after ETT insertion from January 1 to March 31, 2020, in 505 patients (293 men, 212 women; mean age, 63 years) from institution A (sample A); 637 chest radiographs obtained from January 1 to January 3, 2020, in 302 patients (157 men, 145 women; mean age, 66 years) in the ICU (with or without an ETT) from institution A (sample B);
and 546 chest radiographs obtained from January 1 to January 20, 2020, in 83 patients
(54 men, 29 women; mean age, 70 years) in the ICU (with or without an ETT) from institution B (sample C). 

- A commercial DL-based AI system was used to identify ETT presence and measure ETT tip-to-carina distance (TCD). 

- The reference standard for proper ETT position was TCD between greater than 3 cm and less than 7 cm, determined by human
readers. 

- Critical ETT position was separately defined as ETT tip below the carina or TCD
of 1 cm or less. 

- AI had sensitivity and specificity for identification of ETT presence of 100.0% and 98.7% (sample B) and 99.2% and 94.5% (sample C). 

- AI had sensitivity and specificity for identification of improper ETT position of 72.5% and 92.0% (sample A),
78.9% and 100.0% (sample B), and 83.7% and 99.1% (sample C). 

- At a threshold y-axis TCD of 2 cm or less, AI had sensitivity and specificity for critical ETT position of 100.0% and 96.7% (sample A), 100.0% and 100.0% (sample B), and 100.0% and 99.2% (sample C).

- AI identified improperly positioned ETTs on chest radiographs obtained after ETT insertion as well as on chest radiographs obtained of patients in the ICU at two institutions.

- Automated AI identification of improper ETT position on chest
radiographs may allow earlier repositioning and thereby reduce complications.

## Thoughts

- Improper ETT placement had the most glaring low sensitivity(73%-84), meaning the AI could not detect some cases. 

- More or better reference points and inclusion of other factors such as angle of ETT could improve sensitivity.

# Beta_AIBasedChestRadiographsAssessingClinicalOutcomesCOVID-19.pdf

## Summary

- This study aimed to investigate the clinical implications and prognostic value of (AI)-based results for chest radiographs (CXR) in COVID-19 patients. 

- A commercial AI-based software(Lunit INSIGHT CXR, version 3) was used to assess CXR data for consolidation and pleural effusion scores.

- Clinical data, including laboratory results, were analyzed for possible prognostic factors. Total O2 supply period, the last SpO2 result, and deterioration were evaluated as prognostic indicators of treatment outcome. 

- Generalized linear mixed model and regression tests were used to examine the prognostic value of CXR results. 

- Among a total of 228 patients, consolidation scores had a significant association with erythrocyte sedimentation rate and C-reactive protein changes. 

- Initial consolidation scores were associated with the last SpO2 result (estimate −0.018, p = 0.024). 

- All consolidation scores during admission showed significant association with the total O2 supply period and the last SpO2 result.

- Early changing degree of consolidation score showed an association with deterioration (odds ratio 1.017, 95% confidence interval 1.005–1.03). 

- In conclusion, AI-based CXR results for consolidation have potential prognostic value for predicting treatment outcomes in COVID-19 patients.

- Overfitting has been a major obstacle to the actual clinical application of AI algorithms.

- Pleural effusion score did not show significant results.

## Thoughts

- Diverse data sources could potentially address overfitting.

- Patients lacking followup exams was a weakpoint in the data.

- AI was a component in this study, but the fact that there were patterns between patient factors, means an AI for prognosis could potentially be developed.

# Delta_COVID-19PneumoniaCXRDLComputerAidedSystem.pdf


## Summary
- Chest X-rays (CXRs) can help triage for (COVID-19) patients in resource-constrained environments, and a computer-aided detection system (CAD) that can identify pneumonia on CXR may help the triage of patients

- In this study, CXRs of patients with and without COVID-19 confirmed by reverse transcriptase polymerase chain reaction (RT-PCR) were retrospectively collected from four and one institution, respectively, 

- A commercialized, regulatory-approved CAD that can identify various abnormalities including pneumonia was used to analyze each CXR. 

- Performance of the CAD was evaluated using area under the receiver operating characteristic curves (AUCs), with reference standards of the RT-PCR results and the presence of findings of pneumonia on chest CTs obtained within 24 hours from the CXR. 

- The performance of CAD (AUCs, 0.714 and 0.790 against RT-PCR and chest CT, respectively hereinafter) were similar with those of thoracic radiologists (AUCs, 0.701 and 0.784), and higher than those of non-radiologist physicians (AUCs, 0.584 and 0.650). 

- Non-radiologist physicians showed significantly improved performance when assisted with the CAD (AUCs, 0.584 to 0.664 and 0.650 to 0.738). In addition, inter-reader agreement among physicians was also improved in the CAD-assisted interpretation (Fleiss’ kappa coefficient, 0.209 to 0.322).

## Thoughts

- Non-radiologist doctors seem to benefit most and resource strained locations would obviously benefit but I personally wonder how this CAD can be tuned to outperform radiologists

- Lab result and Radiologist confirmation on wider dataset could potentially boost the algorithm's accuracy


# Epsilon_DLAlgorithmsChestRadiographsTriageCOVID-19.pdf

## Summary

- The aim of this study was to evaluate the efficacy of the DL algorithm for detecting COVID-19 pneumonia on CR compared with formal radiology reports. 

- This is a retrospective study of adult patients that were diagnosed as positive COVID-19 cases based on the reverse transcription polymerase chain reaction test

- The overall sensitivity and specificity of the DL algorithm for detecting COVID-19 pneumonia on CR were 95.6%, and 88.7%, respectively. 

- The area under the curve value of the DL algorithm for the detection of COVID-19 with pneumonia was 0.921. 

- The DL algorithm demonstrated a satisfactory diagnostic performance comparable with that of formal radiology reports in the CR-based diagnosis of pneumonia in COVID-19 patients.

## Thoughts

- There were 11 false-positive results obtained with the DL algorithm: normal vascular marking(n = 6), increased vascular marking (n = 2), emphysematous lung (n = 1), interstitial thickening (n = 1), and subsegmental atelectasis (n = 1).

- Targeted fine-tuning to avoid the most common false-positive(normal vascular marking)

- DL system had a higher Sensitivity but lower Specificity than formal radiology report

# Eta_DLComputerAidedAlgorithmChestRadiographs.pdf

## Summary

- The study evaluates a deep learning-based Computer-Aided Diagnosis (CAD) algorithm for detecting and localizing three major thoracic abnormalities visible on chest radiographs (CRs).

- A subset of 244 subjects, with 60% having abnormal CRs, was analyzed. 

- The performance of physicians with and without the assistance of the algorithm was compared using observer performance tests.

- The algorithm demonstrated high accuracy in detecting abnormalities, with area under the receiver operating characteristic (ROC) curve (AUC) values of 0.9883 for nodules/mass, 1.000 for consolidation, and 0.9997 for pneumothorax. 

- When physicians were assisted by the algorithm, their overall performance in image classification and lesion detection improved, as indicated by higher AUC values and a weighted jackknife alternative free-response ROC (wJAFROC) figure of merit (FOM). 

- Specifically, the overall AUC for image classification increased from 0.8679 without the CAD algorithm to 0.9112 with it, and the wJAFROC FOM for lesion detection improved from 0.8426 to 0.9112.

- These findings suggest that the deep learning-based CAD algorithm can enhance physicians' performance in detecting major thoracic abnormalities on chest radiographs.

- CAD systems powered by artificial intelligence have shown promise in improving the detection and measurement of various diseases, including breast cancer, brain tumors, liver lesions, vessel borders, and blood flow dynamics. 

- In the realm of thoracic diseases, automated detection algorithms have been developed for conditions like tuberculosis, as well as for distinguishing between normal and abnormal chest radiographs. 

- A DL algorithm  used in this study was designed to classify chest radiographs of patients with 3 major abnormal findings, including nodule/mass, consolidation, and pneumothorax, and enhance the performance of human readers.

- Board-certified radiologists 0.9313 (0.9097–0.9529) 0.9586 (0.9399–0.9730) 
Non-radiology physicians 0.9153 (0.8917–0.9389) 0.9440 (0.9425–0.9747)
General practitioners 0.7686 (0.7311–0.8062) 0.8940 (0.8663–0.9218)

## Thoughts

- General practitioners benefited most from CADs

- Different data such as patient blood pH, blood cell count, age could be utilized to enhance diagnosis accuracy.

- Opacity in lower areas could be used as a general parameter for project

# Gamma_AIBasedDetectionDiagnosticPerformance.pdf

## Summary

- This study aimed to evaluate the accuracy of radiologists’ interpretations of chest radiographs using different visualization methods for the same AI-CAD. 

- A commercialized AI-CAD using three different methods of visualizing was applied: (a) closed-line method, (b) heat map method, and (c) combined method.

- A reader test was conducted with five trainee radiologists over three interpretation sessions. In each
session, the chest radiographs were interpreted using AI-CAD with one of the three visualization
methods in random order. 

- Examination-level sensitivity and accuracy, and lesion-level detection rates
for clinically significant abnormalities were evaluated for the three visualization methods. 

- The sensitivity (p = 0.007) and accuracy (p = 0.037) of the combined method are significantly higher than that
of the closed-line method. 

- Detection rates using the heat map method (p = 0.043) and the combined
method (p = 0.004) are significantly higher than those using the closed-line method. 

- The methods
for visualizing AI-CAD results for chest radiographs influenced the performance of radiologists’
interpretations. 

- Combining the closed-line and heat map methods for visualizing AI-CAD results led
to the highest sensitivity and accuracy of radiologists.

## Thoughts

- Heat map method seems to be better for visual indications compared to the closed line method

- Maybe an isolation of the heatmap with additional info in the UI with patient data could improve the process

# Theta_PrognosisInterventionPredictionModelCOVID-19.pdf

## Summary

- The aim of the study is to develop and validate a multimodal artificial intelligence (AI) system using clinical findings, laboratory data and AI-interpreted
features of chest X-rays (CXRs), and to predict the prognosis and the required interventions for
patients diagnosed with COVID-19, using multi-center data. 
- In total, 2282 real-time reverse transcriptase polymerase chain reaction-confirmed COVID-19 patients’ initial clinical findings, laboratory
data and CXRs were retrospectively collected from 13 medical centers in South Korea, between
January 2020 and June 2021. 
- The prognostic outcomes collected included intensive care unit (ICU)
admission and in-hospital mortality. 
- Intervention outcomes included the use of oxygen (O2
) supplementation, mechanical ventilation and extracorporeal membrane oxygenation (ECMO). 
- A deep
learning algorithm detecting 10 common CXR abnormalities (DLAD-10) was used to infer the initial
CXR taken. 
- A random forest model with a quantile classifier was used to predict the prognostic
and intervention outcomes, using multimodal data. 
- The area under the receiver operating curve
(AUROC) values for the single-modal model, using clinical findings, laboratory data and the outputs
from DLAD-10, were 0.742 (95% confidence interval [CI], 0.696–0.788), 0.794 (0.745–0.843) and 0.770
(0.724–0.815), respectively. 
- The AUROC of the combined model, using clinical findings, laboratory
data and DLAD-10 outputs, was significantly higher at 0.854 (0.820–0.889) than that of all other
models (p < 0.001, using DeLong’s test). 
- In the order of importance, age, dyspnea, consolidation
and fever were significant clinical variables for prediction. 
- The most predictive DLAD-10 output
was consolidation. 

## Thoughts

- Predicting prognosis and required interventions can be predicted but patient welfare and advocacy should be considered for any applications in real life scenarios

- Most likely a doctor utilizing an AI system for accountability purposes

# Zeta_DLBasedComputerAidedDetectionCOVID-19.pdf

## Summary

## Thoughts