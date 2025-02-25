import pandas as pd
import numpy as np

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

if __name__ == "__main__":
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