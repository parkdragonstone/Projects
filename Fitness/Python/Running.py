# 칼만 필터 초기화 단계
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from scipy.spatial.transform import Rotation as R
from detecta import detect_onset, detect_peaks

def initialize_state():
    x = np.zeros(6)  # 상태 벡터: [ax, ay, az, gx, gy, gz]
    P = np.eye(6)  # 공분산 행렬 초기화
    F = np.eye(6)  # 상태 전이 행렬
    H = np.eye(6)  # 측정 행렬
    Q = np.eye(6) * 0.01  # 프로세스 노이즈 공분산
    R1 = np.eye(6) * 0.5  # 관측 노이즈 공분산
    return x, P, F, H, Q, R1

# 칼만 필터
def kalman_filter(z, x, P, F, H, Q, R1):
    # 예측 단계
    x_pred = F @ x
    P_pred = F @ P @ F.T + Q

    # 칼만 이득
    K = P_pred @ H.T @ np.linalg.inv(H @ P_pred @ H.T + R1)

    # 업데이트 단계
    x = x_pred + K @ (z - H @ x_pred)
    P = (np.eye(len(K)) - K @ H) @ P_pred

    return x, P

# 상보 필터
class ComplementaryFilter:
    def __init__(self, alpha=0.96, dt = 1/60):
        self.alpha = alpha
        self.dt = dt
        self.angle_pitch = 0.0
        self.angle_roll = 0.0
        self.angle_yaw = 0.0
        self.Cal_GyX = 0.0
        self.Cal_GyY = 0.0
        self.Cal_GyZ = 0.0
        self.ACC_X_OFFSET = 0.0
        self.ACC_Y_OFFSET = 0.0
        self.ACC_Z_OFFSET = 0.0
        self.GYR_X_OFFSET = 0.0
        self.GYR_Y_OFFSET = 0.0
        self.GYR_Z_OFFSET = 0.0
        
    def update(self, gyro, accel):
        # 자이로스코프 데이터 (각속도)
        gx, gy, gz = np.radians(gyro)  # Convert to radians per second

        # 가속도계 데이터 (가속도)
        ax, ay, az = accel

        # 가속도계로부터 피치와 롤 각도 계산
        acc_pitch = np.arctan2(ay - self.ACC_Y_OFFSET, np.sqrt(ax * ax + az * az)) * 57.29577951
        acc_roll = -np.arctan2(ax - self.ACC_X_OFFSET, np.sqrt(ay * ay + az * az)) * 57.29577951
        acc_yaw = np.arctan2(np.sqrt(ax * ax + az * az), az - self.ACC_Z_OFFSET) * 57.29577951
        
        # 자이로스코프 데이터를 이용한 각도 업데이트
        self.Cal_GyX += (gx - self.GYR_X_OFFSET) * self.dt
        self.Cal_GyY += (gy - self.GYR_Y_OFFSET) * self.dt
        self.Cal_GyZ += (gz - self.GYR_Z_OFFSET) * self.dt

        # 상보 필터 적용 (피치와 롤)
        self.angle_pitch = self.alpha * (((gx - self.GYR_X_OFFSET) * self.dt) + self.angle_pitch)+ (1 - self.alpha) * acc_pitch
        self.angle_roll = self.alpha * (((gy - self.GYR_Y_OFFSET) * self.dt) + self.angle_roll) + (1 - self.alpha) * acc_roll
        self.angle_yaw += (gz - self.GYR_Z_OFFSET) * self.dt
        
        return - self.angle_pitch, self.angle_roll

    def calibration(self, gyro, accel):
        self.ACC_X_OFFSET = accel[0]
        self.ACC_Y_OFFSET = accel[1]
        self.ACC_Z_OFFSET = accel[2]
        self.GYR_X_OFFSET = gyro[0]
        self.GYR_Y_OFFSET = gyro[1]
        self.GYR_Z_OFFSET = gyro[2]
        
def calcualate_heartrate(age, HR_data, sampling_rate=1):
    ### 심박수관련 운동 강도 구하기
    ''' 
    age = 28 # 자신의 나의
    HR_data = 심박수 데이터
    sampling_rate = 심박수 데이터의 초당 데이터 획득 프레임의 수
    '''
    # 두 가지의 공식 존재
    HRmax = 206.9 - (0.67 * age) # 공식에 의한 최대 심박수 1번
    HRmax = 220 - age            # 공식에 의한 최대 심박수 2번

    intensity = []

    for HR_now in HR_data:
        if 100 * HR_now/HRmax < 57:
            intensity.append(0)
            print('매우 가벼운운동')
        elif 57 <= 100 * HR_now/HRmax< 64:
            intensity.append(1)
            print('저강도')
        elif 64 <= 100 * HR_now/HRmax < 77:
            intensity.append(2)
            print('중강도')
        elif 77 <= 100 * HR_now/HRmax < 96:
            intensity.append(3)
            print('고강도')
        elif 96 <= 100 * HR_now/HRmax:
            intensity.append(4)
            print('거의 최대 ~ 최대 강도')

    # 0 = 매우 가벼운 운동, 1 = 저강도, 2 = 중강도, 3 = 고강도, 4 = 최대강도
    intensity = pd.Series(intensity).rename('Intensity')
    
    return intensity
    
def data_load(motion, date_folder):
    # 데이터 불러오기
    basedir = os.getcwd()
    extension = '.csv'

    DATA_FILE = {}
    files = [os.path.join(basedir, "Data", motion, date_folder, i) for i in os.listdir(os.path.join(basedir, "Data",motion, date_folder)) if extension in os.path.splitext(i)]
    DATA_FILE[date_folder] = files
    
    # segment = ['Pelvic','Torso','Thigh-L','Thigh-R']
    usecols = ['Acc_X','Acc_Y','Acc_Z','Gyr_X','Gyr_Y','Gyr_Z']
    rename_cols = ['ax','ay','az','gx','gy','gz']
    
    ## 러닝 & 사이클의 4개의 관성센서 데이터 프레임 형태로 변환
    IMU = {}
    
    for f in DATA_FILE[date_folder]:

        seg = os.path.basename(f).split('_')[0]
        df = pd.read_csv(f, skiprows=7, usecols = usecols)
        
        ## 관성 센서 좌표계 오른쪽 + X, 앞쪽 + Y, 수직 + Z 로 모든 센서를 통일되도록 변경
        if seg in ['Torso','Thigh-L','Thigh_R']:
            df = df[['Acc_Y','Acc_Z','Acc_X','Gyr_Y','Gyr_Z','Gyr_X']]
        
        elif seg in ['Pelvic']:
            df = df[['Acc_Y','Acc_Z','Acc_X','Gyr_Y','Gyr_Z','Gyr_X']]
            df[['Acc_Y','Acc_Z','Gyr_Y','Gyr_Z']] = - df[['Acc_Y','Acc_Z','Gyr_Y','Gyr_Z']]
        
        df.columns = rename_cols
        IMU[seg] = df

    # IMU : Pelvic, Torso, Thigh-L, Thigh-R => ax, ay, az, gx, gy, gz 데이터
    return IMU

if __name__ == "__main__":
    motion = '러닝'
    date_folder = '20240617_165629'
    IMU = data_load(motion, date_folder)

    
    sr = 60 # 관성 센서 샘플링 레이츠 (사용하게 될 관성 센서의 샘플링 레이트로 변경해야 함)
    dt = 1/sr  # 시간 간격
    filter = ComplementaryFilter(alpha = 0.96, dt = dt)
    
    # 초기 상태
    x, P, F, H, Q, R1 = initialize_state()
    
    # 캘리브레이션 신호
    calibration_triggered = False

    angles = []

    PELVIC = IMU['Pelvic']
    TRUNK = IMU['Torso']
    RTHI = IMU['Thigh-R']
    LTHI = IMU['Thigh-L']
    
    ####################
    ### 상체 기울기 각도 ###
    ####################
    angles = []
    for idx, z in enumerate(TRUNK.to_numpy()):
        # 칼만 필터 업데이트
        # Raw Data => z = np.array([ax, ay, az, gx, gy, gz]) 
        x, P = kalman_filter(z, x, P, F, H, Q, R1)

        # x => 칼만 필터 보정
        # 상보 필터 업데이트
        gyro = x[3:]  # 자이로스코프 각속도
        accel = x[:3]  # 가속도
        pitch, roll = filter.update(gyro, accel)
        
        # 캘리브레이션 : 60 프레임 (1초) 을 기준으로 제로잉
        # 실시간으로 받을 때는 실시간 신호를 주는 방식으로 변경
        if idx == 60:
            calibration_triggered = True
            
        # 캘리브레이션 트리거 확인
        # x, P 초기화, gyro, accel 값 제로잉            
        if calibration_triggered:
            print("캘리브레이션 완료")
            filter.calibration(gyro, accel)
            x, P, F, H, Q, R1 = initialize_state()
            calibration_triggered = False
        
        print(f"상체 앞 기울기: {pitch:.2f} degrees, 상체 옆 기울기: {roll:.2f} degrees")
        angles.append([pitch, roll])
    
    angles = pd.DataFrame(angles, columns = ['앞 기울기', '옆 기울기']) # 전체 시간에서의 기울기 데이터
    
    fig, ax = plt.subplots(1,2, figsize=(12,3))
    ax[0].plot(angles.iloc[:,0])
    ax[0].set_title('Forward Tilt')
    ax[1].plot(angles.iloc[:,1])
    ax[1].set_title('Lateral Tilt')
    plt.close()
    
    
    
    #####################################
    ### 운동 시간 & 스텝수 & 보빈도(step/s) ###
    #####################################
    ar = np.sqrt(np.sum(PELVIC[['ax','ay','az']] ** 2, axis=1)) # 합가속도
    threshold = np.std(ar[int(len(ar)/4):int(3*len(ar)/4)])

    running = {}
    plot = False
    work = detect_onset(ar, threshold=2*threshold, n_below= 2*sr, show=plot)
    for i in range(len(work)):
        start, end = work[i]
        time = (end - start + 1) / sr # second
        ar_step = ar[start:end+1]
        step = len(detect_peaks(ar_step, mph = 20,  mpd = sr * 0.1, show=plot))
        running[i] = [round(time,4), step, round(step/time, 3)] # 운동시간, 스텝수, 보빈도
        print(f"러닝 시간 = {running[i][0]}\n스텝 수 = {running[i][1]}\n페이스 = {running[i][2]}" )
    
    ##############################    
    ### 거리 관련은 GPS 로 추정    ###
    ### GPS 거리 = X (이동 거리)  ###
    ### 운동시간 = Time          ###
    ### PACE = Time / X        ###
    ### 스트라이드 길이 = X / Step ###
    ##############################
    
    ############################
    ### 심박수관련 운동 강도 구하기 ###
    ############################
    age = 28 # 자신의 나의
    sampling_rate = 1 # hz 심박수 측정하는 것의 sampling rate 
    HR_data = np.random.randint(60,170,100) # 현재 심박수 / 추후 타임시리즈로 데이터가 찍히면 시간에 따른 심박수의 데이터가 찍힘
    
    intensity = calcualate_heartrate(age, HR_data, sampling_rate=sampling_rate)
    
    time = len(intensity)/sampling_rate # 측정 시간
    # 가벼운 운동 시간
    time_0 = len(intensity[intensity == 0])/sampling_rate
    # 저강도 운동 시간
    time_1 = len(intensity[intensity == 1])/sampling_rate
    # 중강도 운동 시간
    time_2 = len(intensity[intensity == 2])/sampling_rate
    # 고강도 운동 시간
    time_3 = len(intensity[intensity == 3])/sampling_rate
    # 최고 강도 운동 시간
    time_4 = len(intensity[intensity == 4])/sampling_rate

    print(f"총 측정 시간     = {time} 초")
    print(f"가벼운 운동 시간  = {time_0} 초")
    print(f"저강도 운동 시간  = {time_1} 초")
    print(f"중강도 운동 시간  = {time_2} 초")
    print(f"고강도 운동 시간  = {time_3} 초")
    print(f"최고강도 운동 시간 = {time_4} 초")
    
    