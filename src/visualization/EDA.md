E:\MIA\venv\Scripts\python.exe E:/MIA/src/visualization/exploratory.py
                                              ImageID  StudyDate_DICOM                                  StudyID                                PatientID  PatientBirth PatientSex_DICOM ViewPosition_DICOM     Projection               MethodProjection Pediatric Modality_DICOM     Manufacturer_DICOM PhotometricInterpretation_DICOM  PixelRepresentation_DICOM PixelAspectRatio_DICOM  SpatialResolution_DICOM  BitsStored_DICOM  WindowCenter_DICOM  WindowWidth_DICOM  Rows_DICOM  Columns_DICOM  XRayTubeCurrent_DICOM  Exposure_DICOM  ExposureInuAs_DICOM  ExposureTime  RelativeXRayExposure_DICOM                                                                                                            Labels group
0   14968019924555248865694512726537711769_3r162a.png         20150622   14968019924555248865694512726537711769   32588016314042952172500681022764831222        1993.0                F                NaN  AP_horizontal  Manual review of DICOM fields        No             CR  PhilipsMedicalSystems                     MONOCHROME2                          0                    NaN                    0.200                12              2047.0             4095.0        1760           2140                    NaN             NaN                  NaN           NaN                       419.0                                                                ['interstitial pattern', 'unchanged', 'pneumonia']     N
1   68031809687808465969241796432987316467_wir071.png         20160918   68031809687808465969241796432987316467  206034364315266209909106199297767779230        1958.0                F                NaN  AP_horizontal  Manual review of DICOM fields        No             CR  PhilipsMedicalSystems                     MONOCHROME2                          0                    NaN                    0.200                12              2047.0             4095.0        1760           2140                    NaN             NaN                  NaN           NaN                       737.0                                                                                    ['consolidation', 'pneumonia']     N
2  240901793831868002350241513124822033145_clgbhs.png         20140407  240901793831868002350241513124822033145   27469857597493924763355022866351961988        1959.0                M                NaN              L  Manual review of DICOM fields        No             CR  PhilipsMedicalSystems                     MONOCHROME2                          0             ['1', '1']                    0.200                12              2047.0             4095.0        2140           1760                    NaN             NaN                  NaN           NaN                      1575.0                                                  ['bronchovascular markings', 'emphysema', 'pneumonia', 'bullas']     N
3  240901793831868002350241513124822033145_bnzkfq.png         20140407  240901793831868002350241513124822033145   27469857597493924763355022866351961988        1959.0                M                NaN             PA  Manual review of DICOM fields        No             CR  PhilipsMedicalSystems                     MONOCHROME2                          0             ['1', '1']                    0.200                12              2047.0             4095.0        1760           2140                    NaN             NaN                  NaN           NaN                       186.0                                                  ['bronchovascular markings', 'emphysema', 'pneumonia', 'bullas']     N
4  303749711587233420279581812693876233202_comg8i.png         20140827  303749711587233420279581812693876233202  159094138658868928674447347956484819685        1948.0                F                NaN              L  Manual review of DICOM fields        No             CR  PhilipsMedicalSystems                     MONOCHROME2                          0             ['1', '1']                    0.143                12              2047.0             4095.0        3000           2262                    0.0             0.0                  NaN           0.0                       398.0  ['aortic elongation', 'alveolar pattern', 'vertebral degenerative changes', 'interstitial pattern', 'pneumonia']     N
StudyDate_DICOM
20150120    59
20170517    56
20140424    55
20150217    53
20140505    51
            ..
20100420     1
20120510     1
20111106     1
20090207     1
20150801     1
Name: count, Length: 2338, dtype: int64
StudyID
126022968388682456059208259745221627283          12
245322425398888579493818370201383775169          10
45619611488718541787082738746754182782            8
183884193889618288521329954021747777031           8
222903195543054630998217816896348187815           8
                                                 ..
216840111366964012339356563862009047164422280     1
216840111366964012339356563862009073093218222     1
216840111366964012373310883942009090091422766     1
216840111366964012339356563862009047184708648     1
216840111366964012768025509942010196113158161     1
Name: count, Length: 15622, dtype: int64
       PatientBirth
count  26020.000000
mean    1957.756841
std       22.427642
min     1911.000000
25%     1940.000000
50%     1955.000000
75%     1971.000000
max     2016.000000
(26020, 9)
Index(['ImageID', 'PatientID', 'PatientBirth', 'Projection', 'Pediatric',
       'Modality_DICOM', 'Manufacturer_DICOM', 'Labels', 'group'],
      dtype='object')
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 26020 entries, 0 to 26019
Data columns (total 9 columns):
 #   Column              Non-Null Count  Dtype  
---  ------              --------------  -----  
 0   ImageID             26020 non-null  object 
 1   PatientID           26020 non-null  object 
 2   PatientBirth        26020 non-null  float64
 3   Projection          26020 non-null  object 
 4   Pediatric           26020 non-null  object 
 5   Modality_DICOM      26020 non-null  object 
 6   Manufacturer_DICOM  26020 non-null  object 
 7   Labels              26020 non-null  object 
 8   group               26020 non-null  object 
dtypes: float64(1), object(8)
memory usage: 1.8+ MB
None
                                              ImageID                                PatientID  PatientBirth     Projection Pediatric Modality_DICOM     Manufacturer_DICOM                                                                                                            Labels group
0   14968019924555248865694512726537711769_3r162a.png   32588016314042952172500681022764831222        1993.0  AP_horizontal        No             CR  PhilipsMedicalSystems                                                                ['interstitial pattern', 'unchanged', 'pneumonia']     N
1   68031809687808465969241796432987316467_wir071.png  206034364315266209909106199297767779230        1958.0  AP_horizontal        No             CR  PhilipsMedicalSystems                                                                                    ['consolidation', 'pneumonia']     N
2  240901793831868002350241513124822033145_clgbhs.png   27469857597493924763355022866351961988        1959.0              L        No             CR  PhilipsMedicalSystems                                                  ['bronchovascular markings', 'emphysema', 'pneumonia', 'bullas']     N
3  240901793831868002350241513124822033145_bnzkfq.png   27469857597493924763355022866351961988        1959.0             PA        No             CR  PhilipsMedicalSystems                                                  ['bronchovascular markings', 'emphysema', 'pneumonia', 'bullas']     N
4  303749711587233420279581812693876233202_comg8i.png  159094138658868928674447347956484819685        1948.0              L        No             CR  PhilipsMedicalSystems  ['aortic elongation', 'alveolar pattern', 'vertebral degenerative changes', 'interstitial pattern', 'pneumonia']     N
Projection
PA               13825
L                 8491
AP_horizontal     2772
AP                 901
COSTAL              31
Name: count, dtype: int64
Labels
['normal']                            9004
['pneumonia']                          783
['infiltrates']                        739
['infiltrates', ' pneumonia']          478
['consolidation', ' pneumonia']        210
['COPD signs']                         156
['alveolar pattern', ' pneumonia']     156
['scoliosis']                          137
['infiltrates', 'unchanged']           134
['alveolar pattern', 'pneumonia']      125
Name: count, dtype: int64
                                                 ImageID  ... group
1412   216840111366964013590140476722013029082557414_...  ...     N
1419   216840111366964013590140476722013029082557414_...  ...     N
1474   216840111366964013686042548532013284111324978_...  ...     N
1478   216840111366964013686042548532013166121349765_...  ...     N
1483   216840111366964013686042548532013213141553325_...  ...     N
...                                                  ...  ...   ...
26015  312778764617415130915951295243246198713_olsncb...  ...     C
26016  312778764617415130915951295243246198713_pitn54...  ...     C
26017  30417792092053170589867008849778809477_iwa8og.png  ...     C
26018  30417792092053170589867008849778809477_iw7387.png  ...     C
26019  30417792092053170589867008849778809477-2_uiwj4...  ...     C

[9214 rows x 9 columns]

Process finished with exit code 0
