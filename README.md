#### \- 프로젝트에 있는 데이터는 일부만 첨부하였음
<br></br>

# Bike (EMG)
약 20분 간의 자전거 주행에서 근전도로 근육 신호를 측정하고 피로도를 측정함.
- 측정 도구 : Delsys Wireless EMG
- 부착 위치 : Rectus Femoris (RF), Vastus Lateralis (VL), Biceps Femoris (BF),      Gasctrocnemuius (GM)
- 독립 변인 : 안장 위치, 높이
- 종속 변인 : 중앙 주파수, 적분근전도 등
- 통계 : Repeated Measures ANOVA
- 결과 : [ipynb File](<Bike (EMG)/Python/resampling + sync + filtering + onset.ipynb>)
<br></br>

# Fitness
관성 센서를 이용하여 Cycle, Hiking, Running에 필요한 알고리즘을 도출함.
<br></br>

# Insole
3D 스캔과 3D 프린팅을 통한 맞춤 인솔의 효과를 보기 위함.
- 동작 : Landing & Stop, Single Leg Standing, Landing & Counter Movement Jump
- 독립 변인 : 맞춤 인솔의 착용 유무
- 종속 변인 : COP 움직인 거리, 속도, 면적
- 통계 : Paired-T test, SPM
- 결과 : [Landing](<Insole/Landing Data_procssing.ipynb>), [Stability](<Insole/Stablility Data_processing.ipynb>)
<br></br>

# Shoe Clustering
여러 신발의 기계 테스트 결과를 PCA와 Kmeans 를 이용하여 군집화를 하였음.
![alt text](<Shoe Clustering/Images/1.png>) ![alt text](<Shoe Clustering/Images/2.png>) ![alt text](<Shoe Clustering/Images/3.png>) ![alt text](<Shoe Clustering/Images/4.png>)
<br></br>

# ShoulderROM
PoseEstimation (SmartPhone), Vicon (Marker System), Theia (Markerless System) 각각에서 측정된 어꺠 관절 각도의 비교
<br>

본 내용
- Vicon의 마커로 Trunk와 Right Upper Arm 의 Transformation Matrix를 만들고 내적을 하여 관절각도를 계산
- Theia 에서 제공한 Transformation Matrix 데이터를 이용한 관절 각도 계산
- Vicon의 마커 데이터를 2D 상에서 계산
</br>

![alt text](ShoulderROM/Videos/sample.gif)
<br></br>

# Y-Balance
PoseEstimation (SmartPhone), Vicon (Marker System), Theia (Markerless System) 각각에서 측정된 Y-balance의 각도 비교
![alt text](Y-Balance/Videos/animation.gif)