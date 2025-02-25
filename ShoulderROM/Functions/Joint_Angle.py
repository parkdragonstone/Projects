import pandas as pd
import numpy as np
import scipy.signal as signal


class ROTATION_to_EULER():
    '''
    Rotation Matrix 를 오일러 각도로 변환하는 과정
    단위 : 라디안
    카르단 시퀀스에 따라 함수 정의
    '''
    def rot2eul_XYZ(Matrix): # XYZ
        alpha = np.arctan2(-Matrix[1,2], Matrix[2,2]) # arctan -R23, R33
        beta  = np.arcsin(Matrix[0,2]) # arcsin R13
        gamma = np.arctan2(-Matrix[0,1], Matrix[0,0]) # arctan -R12, R11
        return alpha, beta, gamma
    
    def rot2eul_XZY(Matrix): # XZY
        alpha = np.arctan2(Matrix[2,1],Matrix[1,1]) # arctan R32, R22
        beta = np.arctan2(Matrix[0,2],Matrix[0,0]) # arctan R13, R11 
        gamma = np.arcsin(-Matrix[0,1]) # arccos -R12
        return alpha, beta, gamma
    
    def rot2eul_YXY(Matrix): # YXY
        alpha = np.arctan2(Matrix[0,1],Matrix[2,1]) # arctan R12, R32
        beta = np.arccos(Matrix[1,1]) # arccos R22
        gamma = np.arctan2(Matrix[1,0],-Matrix[1,2]) # arctan R21, -R23
        return alpha, beta, gamma
    
    def rot2eul_YXZ(Matrix): # YXZ
        alpha = np.arcsin(-Matrix[1,2]) # arcsin -R23
        beta = np.arctan2(Matrix[0,2],Matrix[2,2]) # arctan R13, R33
        gamma = np.arctan2(Matrix[1,0],Matrix[1,1]) # arctan R21, R22
        return alpha, beta, gamma
    
    def rot2eul_YZX(Matrix): # YZX
        alpha = np.arctan2(-Matrix[1,2],Matrix[1,1]) # arctan -R23, R22
        beta = np.arctan2(-Matrix[2,0],Matrix[0,0]) # arctan2 -R31, R11
        gamma = np.arcsin(Matrix[1,0]) # arcsin R21
        return alpha, beta, gamma
    
    def rot2eul_YZY(Matrix): # YZY
        alpha = np.arctan2(Matrix[2,1],-Matrix[0,1]) # arctan R32, -R12
        beta = np.arccos(Matrix[1,1]) # arccos R22
        gamma = np.arctan2(Matrix[1,2],Matrix[1,0]) # arctan R23, R21
        return alpha, beta, gamma        
    
    def rot2eul_ZXY(Matrix): # ZXY
        alpha = np.arcsin(Matrix[2,1]) # arcsin R32
        beta = np.arctan2(-Matrix[2,0],Matrix[2,2]) # arctan2 -R31, R33
        gamma = np.arctan2(-Matrix[0,1],Matrix[1,1]) # arctan -R12, R22
        return alpha, beta, gamma
    
    def rot2eul_ZYZ(Matrix): # ZYZ
        alpha = np.arctan2(Matrix[1,2],Matrix[0,2]) # arctan R23, R13
        beta = np.arccos(Matrix[2,2]) # arccos R33
        gamma = np.arctan2(Matrix[2,1],-Matrix[2,0]) # arctan R32, -R31
        return alpha, beta, gamma    
    
    def rot2eul_ZYX(Matrix): #ZYX
        alpha = np.arctan2(Matrix[2,1],Matrix[2,2]) # arctan R32, R33
        beta = np.arcsin(-Matrix[2,0]) # arccin -R31
        gamma = np.arctan2(Matrix[1,0],Matrix[0,0]) # arctan R21, R11
        return alpha, beta, gamma
    
    def rot2eul_ZXZ(Matrix): # ZXZ
        alpha = np.arctan2(Matrix[0,2],-Matrix[1,2]) # arctan2 R13, -R23
        beta = np.arccos(Matrix[2,2]) # arccos R33
        gamma = np.arctan2(Matrix[2,0],Matrix[2,1]) # arctan R31, R32
        return alpha, beta, gamma



class RotationMatrix():
    def Rx(alpha):
        R = np.array([[1,             0,              0],
                      [0, np.cos(alpha), -np.sin(alpha)],
                      [0, np.sin(alpha),  np.cos(alpha)]])
        return R

    def Ry(beta):
        R = np.array([[np.cos(beta) , 0, np.sin(beta)],
                      [0            , 1,            0],
                      [-np.sin(beta), 0, np.cos(beta)]])
        return R

    def Rz(gamma):
        R = np.array([[np.cos(gamma),-np.sin(gamma),0],
                      [np.sin(gamma), np.cos(gamma),0],
                      [0            ,             0,1]])
        return R



def Euler_to_RotaionMatrix(angle, cardan_sequence):
    '''
    angle : 각도 값 (단위 : Degree), X, Y, Z 값이 포함되어있는 형태
    cardan seqeunce 에 따라 회전 행렬을 추출
    '''
    alpha,beta,gamma = angle
    if cardan_sequence == 'xyz':
        R = np.dot(np.dot(RotationMatrix.Rx(alpha),RotationMatrix.Ry(beta)), RotationMatrix.Rz(gamma))
    elif cardan_sequence == 'xzy':
        R = np.dot(np.dot(RotationMatrix.Rx(alpha),RotationMatrix.Rz(gamma)), RotationMatrix.Ry(beta))
    elif cardan_sequence == 'yxy':
        R = np.dot(np.dot(RotationMatrix.Ry(alpha),RotationMatrix.Rx(beta)), RotationMatrix.Ry(gamma))
    elif cardan_sequence == 'yxz':
        R = np.dot(np.dot(RotationMatrix.Ry(beta),RotationMatrix.Rx(alpha)), RotationMatrix.Rz(gamma))
    elif cardan_sequence == 'yzx':
        R = np.dot(np.dot(RotationMatrix.Ry(beta),RotationMatrix.Rz(gamma)), RotationMatrix.Rx(alpha))
    elif cardan_sequence == 'yzy':
        R = np.dot(np.dot(RotationMatrix.Ry(alpha),RotationMatrix.Rz(beta)), RotationMatrix.Ry(gamma))
    elif cardan_sequence == 'zxy':
        R = np.dot(np.dot(RotationMatrix.Rz(gamma),RotationMatrix.Rx(alpha)), RotationMatrix.Ry(beta))
    elif cardan_sequence == 'zxz':
        R = np.dot(np.dot(RotationMatrix.Rz(alpha),RotationMatrix.Rx(beta)), RotationMatrix.Rz(gamma))
    elif cardan_sequence == 'zyx':
        R = np.dot(np.dot(RotationMatrix.Rz(gamma),RotationMatrix.Ry(beta)), RotationMatrix.Rx(alpha))
    elif cardan_sequence == 'zyz':
        R = np.dot(np.dot(RotationMatrix.Rz(alpha),RotationMatrix.Ry(beta)), RotationMatrix.Rz(gamma))
    return R



def Euler_to_TransformationMatrix(angle, origin, cardan_sequence):
    '''
    X, Y, Z 값의 각도 값을 인풋으로 받음
    각 분절의 Origin 을 인풋으로 받음
    cardan seqeunce를 입력
    분절의 각도값과 원점을 입력받아 변환 행렬을 만듬
    '''
    origin_x, origin_y, origin_z = origin
    R = Euler_to_RotaionMatrix(angle,cardan_sequence)
    T = np.append(np.append(R.T, np.array([[origin_x,origin_y,origin_z]]), axis=0).T, np.array([[0,0,0,1]]),axis=0)
    return T



def global_angle(Matrix, segment_name, cardan_squence):
    '''
    분절의 전역 좌표계를 기준으로한 분절 각도 추출
    cardan_sequence : 회전 순서에 대한 정의
    Matrix : 각 분절의 회전 행렬 또는 변환 행렬
    '''
    N = Matrix.shape[-1]
    angle = np.zeros((3,N))

    # 카르단 시퀀스
    if cardan_squence == 'xyz':
        for i in range(N):
            [angle[0,i], angle[1,i], angle[2,i]] = ROTATION_to_EULER.rot2eul_XYZ(Matrix[:,:,i])
    elif cardan_squence == 'xzy':
        for i in range(N):
            [angle[0,i], angle[1,i], angle[2,i]] = ROTATION_to_EULER.rot2eul_XZY(Matrix[:,:,i])
    elif cardan_squence == 'yxy':
        for i in range(N):
            [angle[0,i], angle[1,i], angle[2,i]] = ROTATION_to_EULER.rot2eul_YXY(Matrix[:,:,i])
    elif cardan_squence == 'yxz':
        for i in range(N):
            [angle[0,i], angle[1,i], angle[2,i]] = ROTATION_to_EULER.rot2eul_YXZ(Matrix[:,:,i])            
    elif cardan_squence == 'yzx': # X-Y-Z
        for i in range(N):
            [angle[0,i], angle[1,i], angle[2,i]] = ROTATION_to_EULER.rot2eul_YZX(Matrix[:,:,i])
    elif cardan_squence == 'yzy': # Z-Y-Z
        for i in range(N):
            [angle[0,i], angle[1,i], angle[2,i]] = ROTATION_to_EULER.rot2eul_YZY(Matrix[:,:,i])
    elif cardan_squence == 'zxy': # X-Y-Z
        for i in range(N):
            [angle[0,i], angle[1,i], angle[2,i]] = ROTATION_to_EULER.rot2eul_ZXY(Matrix[:,:,i])
    elif cardan_squence == 'zxz': # Z-Y-Z
        for i in range(N):
            [angle[0,i], angle[1,i], angle[2,i]] = ROTATION_to_EULER.rot2eul_ZXZ(Matrix[:,:,i])
    elif cardan_squence == 'zyx': # Z-Y-Z
        for i in range(N):
            [angle[0,i], angle[1,i], angle[2,i]] = ROTATION_to_EULER.rot2eul_ZYX(Matrix[:,:,i])
    elif cardan_squence == 'zyz': # Z-Y-Z
        for i in range(N):
            [angle[0,i], angle[1,i], angle[2,i]] = ROTATION_to_EULER.rot2eul_ZYZ(Matrix[:,:,i])
            
    col = [f"{segment_name}_ANGLE_X", f"{segment_name}_ANGLE_Y",f"{segment_name}_ANGLE_Z"]
    angle = np.rad2deg(angle)
    angle = pd.DataFrame(angle.T, columns = col)
    return angle
    

def joint_angle_matrix(Proximal_Matrix, Distal_Matrix):
    '''
    원위 분절과 근위 분절의 회전 행렬 또는 변환 행렬의 관계를 나타나는 행렬 추출
    '''
    N = Proximal_Matrix.shape[-1]
    
    if Proximal_Matrix.shape[0] == 4: # Transformation Matrix
        Matrix = np.zeros((4,4,N))
    elif Proximal_Matrix.shape[0] == 3: # Rotation Matrx
        Matrix = np.zeros((3,3,N))
        
    for i in range(N):
        Matrix[:,:,i] = np.dot(
                        np.linalg.inv(Proximal_Matrix[:,:,i]), # Proximal Seg
                        Distal_Matrix[:,:,i] # Distal Segment
                        )
    return Matrix

def Calculate_Reference_Frame(Segment_matrix, Reference):
    '''
    레퍼런스 프레임을 기준으로 각 관절의 변환 행렬 또는 회전 행렬을 맞추는 작업
    '''
    N = Segment_matrix.shape[-1]
    dim = Segment_matrix.shape[0]
    if dim == 3:
        Matrix = np.zeros((3,3,N))
    elif dim == 4:
        Matrix = np.zeros((4,4,N))
        
    for i in range(N):
        Matrix[:,:,i] = np.dot(Segment_matrix[:,:,i], 
                               np.linalg.inv(Reference))
    
    return Matrix
    

def joint_angle(Prox_Matrix, Distal_Matrix, joint_name, cardan_squence):
    '''
    관절 각도 계산
    Prox_Matrix = 근위(레퍼런스 분절) 분절의 변환 행렬 또는 회전 행렬
    Distal_Matrix = 원위(세그먼트 분절) 분절의 변환 행렬 또는 회전 행렬
    X, Y, Z 값을 데이터프레임 형태로 아웃
    '''
    Matrix = joint_angle_matrix(Prox_Matrix, Distal_Matrix)
    N = Matrix.shape[-1]
    angle = np.zeros((3,N))
    
    # 카르단 시퀀스
    if cardan_squence == 'xyz':
        for i in range(N):
            [angle[0,i], angle[1,i], angle[2,i]] = ROTATION_to_EULER.rot2eul_XYZ(Matrix[:,:,i])
    elif cardan_squence == 'xzy':
        for i in range(N):
            [angle[0,i], angle[1,i], angle[2,i]] = ROTATION_to_EULER.rot2eul_XZY(Matrix[:,:,i])
    elif cardan_squence == 'yxy':
        for i in range(N):
            [angle[0,i], angle[1,i], angle[2,i]] = ROTATION_to_EULER.rot2eul_YXY(Matrix[:,:,i])
    elif cardan_squence == 'yxz':
        for i in range(N):
            [angle[0,i], angle[1,i], angle[2,i]] = ROTATION_to_EULER.rot2eul_YXZ(Matrix[:,:,i])            
    elif cardan_squence == 'yzx': # X-Y-Z
        for i in range(N):
            [angle[0,i], angle[1,i], angle[2,i]] = ROTATION_to_EULER.rot2eul_YZX(Matrix[:,:,i])
    elif cardan_squence == 'yzy': # Z-Y-Z
        for i in range(N):
            [angle[0,i], angle[1,i], angle[2,i]] = ROTATION_to_EULER.rot2eul_YZY(Matrix[:,:,i])
    elif cardan_squence == 'zxy': # X-Y-Z
        for i in range(N):
            [angle[0,i], angle[1,i], angle[2,i]] = ROTATION_to_EULER.rot2eul_ZXY(Matrix[:,:,i])
    elif cardan_squence == 'zxz': # Z-Y-Z
        for i in range(N):
            [angle[0,i], angle[1,i], angle[2,i]] = ROTATION_to_EULER.rot2eul_ZXZ(Matrix[:,:,i])
    elif cardan_squence == 'zyx': # Z-Y-Z
        for i in range(N):
            [angle[0,i], angle[1,i], angle[2,i]] = ROTATION_to_EULER.rot2eul_ZYX(Matrix[:,:,i])
    elif cardan_squence == 'zyz': # Z-Y-Z
        for i in range(N):
            [angle[0,i], angle[1,i], angle[2,i]] = ROTATION_to_EULER.rot2eul_ZYZ(Matrix[:,:,i])
    
    col = [f"{joint_name}_ANGLE_X", f"{joint_name}_ANGLE_Y",f"{joint_name}_ANGLE_Z"]
    angle = np.rad2deg(angle)
    angle = pd.DataFrame(angle.T, columns = col)
    
    return angle



def process_angle(data):
    '''
    관절 각도 값이 -180 ~ 180 도까지 표현되어 그 범위 이상으로 가는 값을 전처리해주는 작업
    '''
    up = np.where((np.diff(data) > 300))[0].tolist()
    down = np.where((np.diff(data) < -300))[0].tolist()
    all = sorted([*up, *down])
    all_len = len(all)
    if (len(up) == 0) & (len(down) == 0):
        pass
    
    elif (len(up) == 1) & (len(down) == 0):
        data.iloc[all[0]+1:] = data.iloc[all[0]+1:] - 360

    elif (len(up) == 0) & (len(down) == 1):
        data.iloc[all[0]+1:] = data.iloc[all[0]+1:] + 360

    elif (len(up) >= 1) & (len(down) >= 1):
        if up[0] < down[0]:
            if all_len % 2 == 0:
                for i in range(0,all_len,2):
                    data.iloc[all[i]+1:all[i+1]+1] = data.iloc[all[i]+1:all[i+1]+1] - 360
            elif all_len % 2 == 1:
                for i in range(0,all_len - 1, 2):
                    data.iloc[all[i]+1:all[i+1]+1] = data.iloc[all[i]+1:all[i+1]+1] - 360
                data.iloc[all[-1]+1:] = data.iloc[all[-1]+1:] - 360                                               
        elif up[0] > down[0]:
            if all_len % 2 == 0:
                for i in range(0,all_len,2):
                    data.iloc[all[i]+1:all[i+1]+1] = data.iloc[all[i]+1:all[i+1]+1] + 360
            elif all_len % 2 == 1:
                for i in range(0,all_len - 1, 2):
                    data.iloc[all[i]+1:all[i+1]+1] = data.iloc[all[i]+1:all[i+1]+1] + 360
                data.iloc[all[-1]+1:] = data.iloc[all[-1]+1:] + 360
                
    return data



def dataframe_differentiate(data, sr):
    '''
    data = 데이터 프레임 형태의 데이터
    sr = sampling rate 
    아웃풋 : 데이터 프레임 형태의 각도값을 미분을 통한 데이터 프레임 형태의 각속도 값으로 추출
    '''
    N = data.shape[0]
    cols = [col.replace('ANGLE','ANGULAR_VELOCITY') for col in data.columns]
    velocity = (data - data.shift(2)).shift(-1)/ (2/sr)
    velocity.columns = cols
    return velocity
    
def lowpass_filter(data, sr, cut_off, order):
    '''
    버터워스 로우패스 필터
    '''
    nyq = 0.5 * sr
    b, a = signal.butter(order, cut_off/nyq, btype = 'low')
    lp_df = signal.filtfilt(b, a, data)
    return lp_df



def baseball_theia_pipeline(file):
    '''
    야구 떼이아 c3d 파일을 위한 파이프 라인
    컬럼명 => 주동팔 + Stride Leg : LEAD, 글러브 팔 + Drive Leg : REAR 
    kinematic : 각 관절의 각도, 각속도, 관절점에 대한 데이터 프레임 추출
    TMAT : 각 분절의 변환 행렬 추출
    origin : 관절점에 대한 데이터 딕셔너리 형태로 추출
    '''
    import ezc3d
    import os
    
    c3d = ezc3d.c3d(file,extract_forceplat_data=True)
    label = [i.replace('_4X4','') for i in c3d['parameters']['ROTATION']['LABELS']['value']]
    tmat = c3d['data']['rotations']
    name = os.path.basename(file).replace('.c3d','')
    pit_type = file.split('_')[2]
    RMAT = {}
    TMAT = {}
    origin = {}
    origin_df = pd.DataFrame()
    
    for i in range(tmat.shape[2]):
    
        TMAT[label[i]] = tmat[:,:,i,:]
        RMAT[label[i]] = tmat[:3,:3,i,:]
        
        # Scaling 조정
        TMAT[label[i]][:3,3,:] = TMAT[label[i]][:3,3,:] / 1000 # mm 단위 -> m 단위로 변환
        origin[label[i]] = TMAT[label[i]][:3,3,:]
        org = pd.DataFrame(TMAT[label[i]][:3,3,:].T, columns = [label[i] + '_X',label[i] + '_Y',label[i] + '_Z'])
        origin_df = pd.concat([origin_df, org], axis = 1)
        
    l_ankle = joint_angle(TMAT['l_shank'], TMAT['l_foot'], 'LT_ANKLE', 'xyz') # XYZ
    l_knee  = joint_angle(TMAT['l_thigh'], TMAT['l_shank'], 'LT_KNEE', 'xyz') # XYZ
    l_hip   = joint_angle(TMAT['pelvis'], TMAT['l_thigh'], 'LT_HIP', 'xyz') # XYZ

    r_ankle = joint_angle(TMAT['r_shank'], TMAT['r_foot'], 'RT_ANKLE', 'xyz') # XYZ
    r_knee  = joint_angle(TMAT['r_thigh'], TMAT['r_shank'], 'RT_KNEE', 'xyz') # XYZ
    r_hip   = joint_angle(TMAT['pelvis'], TMAT['r_thigh'], 'RT_HIP', 'xyz') # XYZ

    pel_tor = joint_angle(TMAT['pelvis'], TMAT['torso'], 'TORSO_PELVIS', 'xyz') #ZYX

    l_sho   = joint_angle(TMAT['torso'], TMAT['l_uarm'], 'LT_SHOULDER', 'zyz') # ZYZ
    l_elb   = joint_angle(TMAT['l_uarm'], TMAT['l_larm'], 'LT_ELBOW', 'xyz') # XYZ
    l_wrist = joint_angle(TMAT['l_larm'], TMAT['l_hand'], 'LT_WRIST', 'xyz') # XYZ
    r_sho   = joint_angle(TMAT['torso'], TMAT['r_uarm'], 'RT_SHOULDER', 'zyz') # ZYZ
    r_elb   = joint_angle(TMAT['r_uarm'], TMAT['r_larm'], 'RT_ELBOW', 'xyz') # XYZ
    r_wrist = joint_angle(TMAT['r_larm'], TMAT['r_hand'], 'RT_WRIST', 'xyz') # XYZ

    pel     = global_angle(TMAT['pelvis'],'PELVIS', 'zyx') # ZYX
    tor     = global_angle(TMAT['torso'],'TORSO', 'zyx') # ZYX
    l_ank   = global_angle(TMAT['l_foot'],'LT_ANKLE_VIRTUAL_LAB', 'xyz') # XYZ
    r_ank   = global_angle(TMAT['r_foot'],'RT_ANKLE_VIRTUAL_LAB', 'xyz') # XYZ
    
    angle = pd.concat(
            [pel     ,tor    ,l_ank, r_ank, pel_tor,
                l_ankle ,l_knee ,l_hip,
                r_ankle ,r_knee ,r_hip,
                l_sho   ,l_elb  ,l_wrist ,
                r_sho   ,r_elb  ,r_wrist], axis=1
            )
        
    ang_cols = angle.columns

    for col in ang_cols:
        process = process_angle(angle[col])
        filtering = lowpass_filter(process, 108, 13.4, 4)
        angle[col] = filtering
        
    newcols = []
    for idx ,col in enumerate(ang_cols):
        if 'R' in file: # 오른손 투수
            pit_type = 'R'
            if ('ANKLE' in col) | ('HIP' in col) | ('KNEE' in col) | ('HEEL' in col):
                split_col = col.split('_')
                if 'R' in split_col[0]: # 오른손 투수의 오른쪽 다리 => REAR
                    split_col[0] = 'REAR'
                    split_col = '_'.join(split_col)
                    newcols.append(split_col)
                elif 'L' in split_col[0]: # 오른손 투수의 왼쪽 다리 => LEAD
                    split_col[0] = 'LEAD'
                    split_col = '_'.join(split_col)
                    newcols.append(split_col)

            elif ('ELBOW' in col) | ('SHOULDER' in col) | ('WRIST' in col):
                split_col = col.split('_')
                
                if 'R' in col.split('_')[0]: # 오른손 투수의 오른쪽 팔 => LEAD
                    split_col[0] = 'LEAD'
                    split_col = '_'.join(split_col)
                    newcols.append(split_col)
                elif 'L' in col.split('_')[0]: # 오른손 투수의 왼쪽 팔 => REAR
                    split_col[0] = 'REAR'
                    split_col = '_'.join(split_col)
                    newcols.append(split_col)
            else:
                newcols.append(col)
                    
        elif 'L' in file: # 왼손 투수
            pit_type = 'L'
            if ('ANKLE' in col) | ('HIP' in col) | ('KNEE' in col) | ('HEEL' in col):
                split_col = col.split('_')
                
                if 'R' in split_col[0]: # 왼쪽 투수의 오른쪽 다리 => LEAD
                    split_col[0] = 'LEAD'
                    split_col = '_'.join(split_col)
                    newcols.append(split_col)
                elif 'L' in split_col[0]: # 왼쪽 투수의 왼쪽 다리 => REAR
                    split_col[0] = 'REAR'
                    split_col = '_'.join(split_col)
                    newcols.append(split_col)
                    
            elif ('ELBOW' in col) | ('SHOULDER' in col) | ('WRIST' in col):
                split_col = col.split('_')
                
                if 'R' in col.split('_')[0]: # 왼쪽 투수의 오른쪽 팔 => REAR
                    split_col[0] = 'REAR'
                    split_col = '_'.join(split_col)
                    newcols.append(split_col)
                elif 'L' in col.split('_')[0]: # 왼쪽 투수의 왼쪽 팔 => LEAD
                    split_col[0] = 'LEAD'
                    split_col = '_'.join(split_col)
                    newcols.append(split_col)
            else:
                newcols.append(col)
                
    angle.columns = newcols

    origin_col = []
    for col in origin_df.columns:
        if pit_type == 'R':
            if ('thigh' in col) | ('shank' in col) | ('foot' in col) | ('toes' in col):
                split_col = col.split('_')
                if split_col[0] == 'r':
                    split_col[0] = 'REAR'
                    split_col = '_'.join(split_col)
                    origin_col.append(split_col)
                elif split_col[0] == 'l':
                    split_col[0] = 'LEAD'
                    split_col = '_'.join(split_col)
                    origin_col.append(split_col)
            elif ('uarm' in col) | ('larm' in col) | ('clavicle' in col) | ('hand' in col):
                split_col = col.split('_')
                if split_col[0] == 'l':
                    split_col[0] = 'REAR'
                    split_col = '_'.join(split_col)
                    origin_col.append(split_col)
                elif split_col[0] == 'r':
                    split_col[0] = 'LEAD'
                    split_col = '_'.join(split_col)
                    origin_col.append(split_col)
            else:
                origin_col.append(col)
        if pit_type == 'L':
            if ('thigh' in col) | ('shank' in col) | ('foot' in col) | ('toes' in col):
                split_col = col.split('_')
                if split_col[0] == 'l':
                    split_col[0] = 'REAR'
                    split_col = '_'.join(split_col)
                    origin_col.append(split_col)
                elif split_col[0] == 'r':
                    split_col[0] = 'LEAD'
                    split_col = '_'.join(split_col)
                    origin_col.append(split_col)
            elif ('uarm' in col) | ('larm' in col) | ('clavicle' in col) | ('hand' in col):
                split_col = col.split('_')
                if split_col[0] == 'r':
                    split_col[0] = 'REAR'
                    split_col = '_'.join(split_col)
                    origin_col.append(split_col)
                elif split_col[0] == 'l':
                    split_col[0] = 'LEAD'
                    split_col = '_'.join(split_col)
                    origin_col.append(split_col)
            else:
                origin_col.append(col)
                
    origin_df.columns = origin_col

    if pit_type == 'R':
        change_minus_plus = ['TORSO_ANGLE_X','TORSO_ANGLE_Y','LEAD_KNEE_ANGLE_X','LEAD_SHOULDER_ANGLE_X','LEAD_SHOULDER_ANGLE_Y','LEAD_SHOULDER_ANGLE_Z']
        angle[change_minus_plus] = - angle[change_minus_plus] 

        angle['LEAD_SHOULDER_ANGLE_X'] = angle['LEAD_SHOULDER_ANGLE_X'] + 180
        angle['LEAD_SHOULDER_ANGLE_Z'] = angle['LEAD_SHOULDER_ANGLE_Z'] - 180
        
    elif pit_type == 'L':
        change_minus_plus = ['TORSO_ANGLE_X','TORSO_ANGLE_Y','LEAD_KNEE_ANGLE_X','LEAD_SHOULDER_ANGLE_X','LEAD_SHOULDER_ANGLE_Y','LEAD_SHOULDER_ANGLE_Z']
        angle[change_minus_plus] = - angle[change_minus_plus]
        angle[['REAR_SHOULDER_ANGLE_Y','LEAD_SHOULDER_ANGLE_Y']] = - angle[['REAR_SHOULDER_ANGLE_Y','LEAD_SHOULDER_ANGLE_Y']]

        change_col = []
        for col in newcols:
            if ('Y' in col) | ('Z' in col):
                change_col.append(col)
        
        angle[change_col] = - angle[change_col]
        angle['REAR_SHOULDER_ANGLE_X'] = angle['REAR_SHOULDER_ANGLE_X'] + 180
        angle['REAR_SHOULDER_ANGLE_Z'] = angle['REAR_SHOULDER_ANGLE_Z'] - 180
        
    velocity = dataframe_differentiate(angle, 180)
    kinematic = pd.concat([angle, velocity, origin_df], axis=1)
    
    return kinematic, TMAT, origin