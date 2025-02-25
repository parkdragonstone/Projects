import pandas as pd
import numpy as np
import os

from sklearn.utils import resample

from collections import defaultdict
from io import StringIO

def data_load(data_dir):
    DATA = {}
    for f in data_dir:
    # 파일을 텍스트 형식으로 읽기
        with open(f, 'r', encoding='cp949') as file:
            lines = file.readlines()

            # 'Time' 행 이후의 데이터만 추출
            start_index = next(i for i, line in enumerate(lines) if 'Trial' in line)
            end_index = next(i for i, line in enumerate(lines) if 'Ave' in line)
            data_subset = lines[start_index:end_index]
            # 리스트를 문자열로 결합
            data_str = ''.join(data_subset)

            # StringIO를 사용해 데이터프레임으로 변환
            df = pd.read_csv(StringIO(data_str), delimiter='\t')
            df = df[~df['Trial'].str.contains('=+', na=False)].astype(float)
            df['force/pen (N/mm)'] = df['Force (N)'] / df['Max Pen (mm)']
            DATA[os.path.splitext(os.path.basename(f))[0]] = np.mean(df,axis=0)
        
    return DATA


def bootstrap_coefficients(X, y, model, n_iterations=1000):
    coefs = []
    
    for _ in range(n_iterations):
        # 부트스트랩 샘플 생성
        X_resampled, y_resampled = resample(X, y, random_state=None)
        
        # 모델 학습
        model.fit(X_resampled, y_resampled)
        
        # 계수 저장
        coefs.append(model.coef_)
    
    return np.array(coefs)

def compute_confidence_intervals(coefs, alpha=0.05):
    lower_percentile = 100 * (alpha / 2)
    upper_percentile = 100 * (1 - alpha / 2)
    
    lower_bounds = np.percentile(coefs, lower_percentile, axis=0)
    upper_bounds = np.percentile(coefs, upper_percentile, axis=0)
    
    return lower_bounds, upper_bounds


### 여러개의 클러스터링 갯수를 List로 입력 받아 각각의 실루엣 계수를 면적으로 시각화한 함수 작성
def visualize_silhouette(cluster_lists, X_features): 
    
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_samples, silhouette_score

    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    
    # 입력값으로 클러스터링 갯수들을 리스트로 받아서, 각 갯수별로 클러스터링을 적용하고 실루엣 개수를 구함
    n_cols = len(cluster_lists)
    
    # plt.subplots()으로 리스트에 기재된 클러스터링 수만큼의 sub figures를 가지는 axs 생성 
    fig, axs = plt.subplots(figsize=(4*n_cols, 4), nrows=1, ncols=n_cols)
    
    # 리스트에 기재된 클러스터링 갯수들을 차례로 iteration 수행하면서 실루엣 개수 시각화
    for ind, n_cluster in enumerate(cluster_lists):
        
        # KMeans 클러스터링 수행하고, 실루엣 스코어와 개별 데이터의 실루엣 값 계산. 
        clusterer = KMeans(n_clusters = n_cluster, random_state=0, init='k-means++', max_iter=300)
        cluster_labels = clusterer.fit_predict(X_features)
        
        sil_avg = silhouette_score(X_features, cluster_labels)
        sil_values = silhouette_samples(X_features, cluster_labels)
        
        y_lower = 10
        axs[ind].set_title('Number of Cluster : '+ str(n_cluster)+'\n' \
                          'Silhouette Score :' + str(round(sil_avg,3)) )
        axs[ind].set_xlabel("The silhouette coefficient values")
        axs[ind].set_ylabel("Cluster label")
        axs[ind].set_xlim([-0.1, 1])
        axs[ind].set_ylim([0, len(X_features) + (n_cluster + 1) * 10])
        axs[ind].set_yticks([])  # Clear the yaxis labels / ticks
        axs[ind].set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1])
        
        # 클러스터링 갯수별로 fill_betweenx( )형태의 막대 그래프 표현. 
        for i in range(n_cluster):
            ith_cluster_sil_values = sil_values[cluster_labels==i]
            ith_cluster_sil_values.sort()
            
            size_cluster_i = ith_cluster_sil_values.shape[0]
            y_upper = y_lower + size_cluster_i
            
            color = cm.nipy_spectral(float(i) / n_cluster)
            axs[ind].fill_betweenx(np.arange(y_lower, y_upper), 0, ith_cluster_sil_values, \
                                facecolor=color, edgecolor=color, alpha=0.7)
            axs[ind].text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
            y_lower = y_upper + 10
            
        axs[ind].axvline(x=sil_avg, color="red", linestyle="--")
    
    
    return fig