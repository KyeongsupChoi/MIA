E:\MIA\venv\Scripts\python.exe E:/MIA/src/visualization/exploratory.py
                                              ImageID  StudyDate_DICOM                                  StudyID                                PatientID  PatientBirth PatientSex_DICOM ViewPosition_DICOM     Projection               MethodProjection Pediatric Modality_DICOM     Manufacturer_DICOM PhotometricInterpretation_DICOM  PixelRepresentation_DICOM PixelAspectRatio_DICOM  SpatialResolution_DICOM  BitsStored_DICOM  WindowCenter_DICOM  WindowWidth_DICOM  Rows_DICOM  Columns_DICOM  XRayTubeCurrent_DICOM  Exposure_DICOM  ExposureInuAs_DICOM  ExposureTime  RelativeXRayExposure_DICOM                                                                                                            Labels group
0   14968019924555248865694512726537711769_3r162a.png         20150622   14968019924555248865694512726537711769   32588016314042952172500681022764831222        1993.0                F                NaN  AP_horizontal  Manual review of DICOM fields        No             CR  PhilipsMedicalSystems                     MONOCHROME2                          0                    NaN                    0.200                12              2047.0             4095.0        1760           2140                    NaN             NaN                  NaN           NaN                       419.0                                                                ['interstitial pattern', 'unchanged', 'pneumonia']     N
1   68031809687808465969241796432987316467_wir071.png         20160918   68031809687808465969241796432987316467  206034364315266209909106199297767779230        1958.0                F                NaN  AP_horizontal  Manual review of DICOM fields        No             CR  PhilipsMedicalSystems                     MONOCHROME2                          0                    NaN                    0.200                12              2047.0             4095.0        1760           2140                    NaN             NaN                  NaN           NaN                       737.0                                                                                    ['consolidation', 'pneumonia']     N
2  240901793831868002350241513124822033145_clgbhs.png         20140407  240901793831868002350241513124822033145   27469857597493924763355022866351961988        1959.0                M                NaN              L  Manual review of DICOM fields        No             CR  PhilipsMedicalSystems                     MONOCHROME2                          0             ['1', '1']                    0.200                12              2047.0             4095.0        2140           1760                    NaN             NaN                  NaN           NaN                      1575.0                                                  ['bronchovascular markings', 'emphysema', 'pneumonia', 'bullas']     N
3  240901793831868002350241513124822033145_bnzkfq.png         20140407  240901793831868002350241513124822033145   27469857597493924763355022866351961988        1959.0                M                NaN             PA  Manual review of DICOM fields        No             CR  PhilipsMedicalSystems                     MONOCHROME2                          0             ['1', '1']                    0.200                12              2047.0             4095.0        1760           2140                    NaN             NaN                  NaN           NaN                       186.0                                                  ['bronchovascular markings', 'emphysema', 'pneumonia', 'bullas']     N
4  303749711587233420279581812693876233202_comg8i.png         20140827  303749711587233420279581812693876233202  159094138658868928674447347956484819685        1948.0                F                NaN              L  Manual review of DICOM fields        No             CR  PhilipsMedicalSystems                     MONOCHROME2                          0             ['1', '1']                    0.143                12              2047.0             4095.0        3000           2262                    0.0             0.0                  NaN           0.0                       398.0  ['aortic elongation', 'alveolar pattern', 'vertebral degenerative changes', 'interstitial pattern', 'pneumonia']     N
       StudyDate_DICOM  PatientBirth  PixelRepresentation_DICOM  SpatialResolution_DICOM  BitsStored_DICOM  WindowCenter_DICOM  WindowWidth_DICOM    Rows_DICOM  Columns_DICOM  XRayTubeCurrent_DICOM  Exposure_DICOM  ExposureInuAs_DICOM  ExposureTime  RelativeXRayExposure_DICOM
count     2.602000e+04  26020.000000                    26020.0             17769.000000      26020.000000        25830.000000       25830.000000  26020.000000   26020.000000           12495.000000    17194.000000         11508.000000  17194.000000                22839.000000
mean      2.013795e+07   1957.756841                        0.0                 0.166662         11.890392         2023.989721        3453.984204   2374.761453    2420.574673             189.127251        3.012039          4662.061175      9.320461                  462.223094
std       2.512403e+04     22.427642                        0.0                 0.026521          0.455203          375.996496         997.078433    656.842895     673.172578             184.284926        4.509169          4858.262857     11.632941                  638.648915
min       2.007050e+07   1911.000000                        0.0                 0.100000         10.000000          511.500000         765.000000    680.000000     740.000000               0.000000        0.000000           500.000000      0.000000                   -3.700000
25%       2.012021e+07   1940.000000                        0.0                 0.143000         12.000000         2047.000000        2389.000000   1760.000000    1920.000000               0.000000        0.000000          1900.000000      0.000000                  159.000000
50%       2.014103e+07   1955.000000                        0.0                 0.148000         12.000000         2047.000000        4095.000000   2140.000000    2140.000000             250.000000        2.000000          2500.000000      7.000000                  353.000000
75%       2.016032e+07   1971.000000                        0.0                 0.200000         12.000000         2047.500000        4095.000000   2852.000000    2830.000000             320.000000        3.000000          5000.000000     10.000000                  498.000000
max       2.017112e+07   2016.000000                        0.0                 0.200000         12.000000         3617.000000        4095.000000   4280.000000    4280.000000             500.000000       50.000000         50000.000000    153.000000                42839.000000
(26020, 28)
Index(['ImageID', 'StudyDate_DICOM', 'StudyID', 'PatientID', 'PatientBirth',
       'PatientSex_DICOM', 'ViewPosition_DICOM', 'Projection',
       'MethodProjection', 'Pediatric', 'Modality_DICOM', 'Manufacturer_DICOM',
       'PhotometricInterpretation_DICOM', 'PixelRepresentation_DICOM',
       'PixelAspectRatio_DICOM', 'SpatialResolution_DICOM', 'BitsStored_DICOM',
       'WindowCenter_DICOM', 'WindowWidth_DICOM', 'Rows_DICOM',
       'Columns_DICOM', 'XRayTubeCurrent_DICOM', 'Exposure_DICOM',
       'ExposureInuAs_DICOM', 'ExposureTime', 'RelativeXRayExposure_DICOM',
       'Labels', 'group'],
      dtype='object')
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 26020 entries, 0 to 26019
Data columns (total 28 columns):
 #   Column                           Non-Null Count  Dtype  
---  ------                           --------------  -----  
 0   ImageID                          26020 non-null  object 
 1   StudyDate_DICOM                  26020 non-null  int64  
 2   StudyID                          26020 non-null  object 
 3   PatientID                        26020 non-null  object 
 4   PatientBirth                     26020 non-null  float64
 5   PatientSex_DICOM                 26020 non-null  object 
 6   ViewPosition_DICOM               11614 non-null  object 
 7   Projection                       26020 non-null  object 
 8   MethodProjection                 26020 non-null  object 
 9   Pediatric                        26020 non-null  object 
 10  Modality_DICOM                   26020 non-null  object 
 11  Manufacturer_DICOM               26020 non-null  object 
 12  PhotometricInterpretation_DICOM  26020 non-null  object 
 13  PixelRepresentation_DICOM        26020 non-null  int64  
 14  PixelAspectRatio_DICOM           11651 non-null  object 
 15  SpatialResolution_DICOM          17769 non-null  float64
 16  BitsStored_DICOM                 26020 non-null  int64  
 17  WindowCenter_DICOM               25830 non-null  float64
 18  WindowWidth_DICOM                25830 non-null  float64
 19  Rows_DICOM                       26020 non-null  int64  
 20  Columns_DICOM                    26020 non-null  int64  
 21  XRayTubeCurrent_DICOM            12495 non-null  float64
 22  Exposure_DICOM                   17194 non-null  float64
 23  ExposureInuAs_DICOM              11508 non-null  float64
 24  ExposureTime                     17194 non-null  float64
 25  RelativeXRayExposure_DICOM       22839 non-null  float64
 26  Labels                           26020 non-null  object 
 27  group                            26020 non-null  object 
dtypes: float64(9), int64(5), object(14)
memory usage: 5.6+ MB