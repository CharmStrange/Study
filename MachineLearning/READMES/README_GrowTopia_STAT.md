# 직접 하는 회귀 분석
## 게임 내 아이템 거래 시장에 대한 통계적 분석이 필요하여 직접 데이터를 기록하고 적합한 선형 회귀 모델을 만들어 앞으로의 활동을 전략적으로 해 나가려는 목적(Profit Maximization). 
---
- 작성한 ipynb 파일(1차) :  [ipynb file](https://github.com/CharmStrange/Study/blob/%EC%9D%B8%EA%B3%B5%EC%A7%80%EB%8A%A5/MachineLearning/ipynb/GrowTopia_stat(V2).ipynb)
---
### 1. 분석에 필요한 데이터의 정의, 수집
먼저 아이템 거래에 대해 분석을 하려면 '***재고 보충량***', '***팔린 물품 수***', '***광고 게재 수***', '***방문자 수***' 정보가 필요하다. 여기서 모든 수치는 하루 기준이며 하루가 지나면 새로 기록을 해 주어야 한다. 
```
# 데이터 형태 정의
 
import pandas as pd

Columns = ['Number of Restock', 'Number of sold items', 'Number of ADS', 'Number of Visitors']
Index = ['Day 1', 'Day 2', 'Day 3', 'Day 4', 'Day 5', 'Day 6', 'Day 7', 'Day 8', 'Day 9', 'Day 10']
```
효율적인 데이터 관리를 위해 ***pandas***를 사용한 데이터프레임을 사용했다. 


```
# 데이터 수집, 구체화
 
DailyData=[
    [2200, 2200, 7, 4],
    [3515, 2400, 8, 3],
    [2400, 2400, 5, 4],
    [4991, 4800, 14, 5],
    [1231, 200, 1, 1],
    [911, 0, 0, 0],
    [0, 800, 0, 1],
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    [1800, 800, 6, 2]
]
Data = pd.DataFrame(DailyData, columns=Columns, index=Index)
Data.to_csv('DailyData.csv')
```
<img width="540" alt="image" src="https://github.com/CharmStrange/Snippet/assets/105769152/61320205-3f12-4182-a22d-a0fd0fc5ef8a">

그리고 실제 기록한 10일간의 데이터를 작성해 주고 csv 파일로 저장해준다.

---
### 2. 데이터로부터 얻어낼 수 있는 인사이트
일단 '***광고 게재 수***'(*독립 변수*)에 따른 '***방문자 수***'(*종속 변수*)와, '***재고 보충량***'(*독립 변수*)에 따른 '***팔린 물품 수***'(*종속 변수*)를 집중적으로 분석하기로 했다.
```
# 데이터 관계의 선형성 확인

import matplotlib.pyplot as plt

# Linearity Checking

Data.plot(x='Number of ADS', y='Number of Visitors', style='o')
plt.title("Number of Visitors vs Number of ADS")
plt.xlabel('Number of ADS')
plt.ylabel('Number of Visitors')
plt.show()

Data.plot(x='Number of Restock', y='Number of sold items', style='o')
plt.title("Number of sold items vs Number of Restock")
plt.xlabel('Number of Restock')
plt.ylabel('Number of sold items')
plt.show()
```
<img width="414" alt="image" src="https://github.com/CharmStrange/Snippet/assets/105769152/da53e99c-f294-4f33-868b-93160dc52950">
<img width="434" alt="image" src="https://github.com/CharmStrange/Snippet/assets/105769152/98bb7f59-a9dd-47ed-a57a-a0d80b38a2de">

데이터 분포의 선형성을 확인하기 위해 시각화를 해 보았다. 데이터가 10개임에도 불구하고 선형성이 나타는 것으로 보아 회귀 분석이 적합할 것으로 예상된다.

---
### 3. 분석 시작
```
# 모델 구현

from sklearn.linear_model import LinearRegression

LR_V = LinearRegression()
LR_S = LinearRegression()

LR_V.fit(Data['Number of ADS'].values.reshape(-1, 1), Data['Number of Visitors'].values.reshape(-1, 1))
LR_S.fit(Data['Number of Restock'].values.reshape(-1, 1), Data['Number of sold items'].values.reshape(-1, 1))
```
라이브러리를 사용해 회귀 분석을 시작했다. ***.fit()*** 메소드는 파라미터에 2차원 배열이 들어가야 하므로 인자로 들어갈 데이터의 형태 변환을 알맞게 해 주었다.
```
# 예측 시도

predicted_visitors = LR.predict( [ [13], [14] ] )
predicted_sold_items = LR.predict( [ [4800], [3200], [600] ] )
```
이제 모델에 독립 변수를 테스트삼아 넣어본다. 위 코드에선 '***광고 게재 수***' : **13**, **14** 이고 '***재고 보충량***' : **4800**, **3200**, **600** 이다.
```
# 결과 확인

print('predicted visitors : ', predicted_visitors)
>>> predicted visitors :  [[5.20206999], [5.56185313]]

print('predicted sold items : ', predicted_sold_items)
>>> predicted sold items : [ [4087.96704049], [2677.80056828], [ 386.28005094] ]
```
예측된 결과를 회귀 직선 그래프를 그려 표현하면 아래와 같다.
```
# 회귀 직선 그래프 그리기

plt.scatter(Data['Number of ADS'], Data['Number of Visitors'], color='blue', label='Actual')
plt.plot(Data['Number of ADS'], LR_V.predict(Data['Number of ADS'].values.reshape(-1, 1)), color='red', label='Predicted')
plt.xlabel('Number of ADS')
plt.ylabel('Number of Visitors')
plt.title('Number of ADS vs Number of Visitors')
plt.legend()
plt.show()

plt.scatter(Data['Number of Restock'], Data['Number of sold items'], color='blue', label='Actual')
plt.plot(Data['Number of Restock'], LR_S.predict(Data['Number of Restock'].values.reshape(-1, 1)), color='red', label='Predicted')
plt.xlabel('Number of Restock')
plt.ylabel('Number of sold items')
plt.title('Number of Restock vs Number of sold items')
plt.legend()
plt.show()
```
<img width="411" alt="image" src="https://github.com/CharmStrange/Snippet/assets/105769152/6ce6e232-a3b1-48e4-b457-897e4abbdbb8">
<img width="429" alt="image" src="https://github.com/CharmStrange/Snippet/assets/105769152/18cdd8d6-8124-48bf-96ae-c1a02679bd21">

```
# 오차 구하기

mse_visitors = mean_squared_error(Data['Number of Visitors'].values.reshape(-1, 1), predicted_visitors)
mse_sold_items = mean_squared_error(Data['Number of sold items'].values.reshape(-1, 1), predicted_sold_items)

print('MSE of visitors: ', mse_visitors)
print('MSE of sold items: ', mse_sold_items)
```
```
>>>
MSE of visitors:  3.5810438634865918
MSE of sold items:  1392562.4923213236
```
오차가 좀 크긴 하지만 고작 10개의 데이터 치고는 괜찮은 직선의 모습이다. 데이터가 너무 적은 것이 원인인 듯 하지만 현재보다 더 많은 실제 데이터를 축적해 나가면 더욱 괜찮은 예측치를 알아낼 수 있을 것이다.

추가로 데이터를 스케일링하면 오차가 좀 줄어들지 않을까 싶어서 해 보았는데 큰 차이는 없었다. 역시나 데이터 수의 부족함이 문제되는것 같다.
```
from sklearn.preprocessing import StandardScaler

# Standardization
scaler = StandardScaler()
scaled_Data_std = scaler.fit_transform(Data)

scaled_Data_std_df = pd.DataFrame(scaled_Data_std, columns=Columns, index=Index)

scaled_Data_std_df

LR_V = LinearRegression()
LR_S = LinearRegression()

LR_V.fit(scaled_Data_std_df['Number of ADS'].values.reshape(-1, 1), scaled_Data_std_df['Number of Visitors'].values.reshape(-1, 1))
LR_S.fit(scaled_Data_std_df['Number of Restock'].values.reshape(-1, 1), scaled_Data_std_df['Number of sold items'].values.reshape(-1, 1))

predicted_visitors = LR_V.predict([[13], [14], [8], [7], [7], [0], [0], [0], [9], [10]])  # give Number of ADS in params
predicted_sold_items = LR_S.predict([[4800], [3200], [600], [6600], [1200], [800], [0], [600], [1561], [2800]])  # give Number of Restock in params

# Adjust negative predictions to 0
predicted_visitors = [max(0, value[0]) for value in predicted_visitors]
predicted_sold_items = [max(0, value[0]) for value in predicted_sold_items]

print('predicted visitors:', predicted_visitors)
print('predicted sold items:', predicted_sold_items)

# Calculate MSE
mse_visitors = mean_squared_error(scaled_Data_std_df['Number of Visitors'].values.reshape(-1, 1), predicted_visitors)
mse_sold_items = mean_squared_error(scaled_Data_std_df['Number of sold items'].values.reshape(-1, 1), predicted_sold_items)

print('MSE of visitors: ', mse_visitors)
print('MSE of sold items: ', mse_sold_items)
```
```
predicted visitors: [11.77742099917544, 12.683376460650475, 7.247643691800271, 6.3416882303252375, 6.3416882303252375, 0, 0, 0, 8.153599153275305, 9.05955461475034]
predicted sold items: [4431.605755547653, 2954.4038370317685, 553.9507194434566, 6093.457913878022, 1107.9014388869132, 738.6009592579421, 8.88241302397551e-17, 553.9507194434566, 1441.1951217520595, 2585.1033574027974]

MSE of visitors:  53.84245232013412
MSE of sold items:  7661501.684724535
```

```
from sklearn.preprocessing import MinMaxScaler

# Normalization
scaler = MinMaxScaler()
scaled_Data_mms = scaler.fit_transform(Data)

scaled_Data_mms_df = pd.DataFrame(scaled_Data_mms, columns=Columns, index=Index)

scaled_Data_mms_df

LR_V = LinearRegression()
LR_S = LinearRegression()

LR_V.fit(scaled_Data_mms_df['Number of ADS'].values.reshape(-1, 1), scaled_Data_mms_df['Number of Visitors'].values.reshape(-1, 1))
LR_S.fit(scaled_Data_mms_df['Number of Restock'].values.reshape(-1, 1), scaled_Data_mms_df['Number of sold items'].values.reshape(-1, 1))

predicted_visitors = LR_V.predict([[13], [14], [8], [7], [7], [0], [0], [0], [9], [10]])  # give Number of ADS in params
predicted_sold_items = LR_S.predict([[4800], [3200], [600], [6600], [1200], [800], [0], [600], [1561], [2800]])  # give Number of Restock in params

# Adjust negative predictions to 0
predicted_visitors = [max(0, value[0]) for value in predicted_visitors]
predicted_sold_items = [max(0, value[0]) for value in predicted_sold_items]

print('predicted visitors:', predicted_visitors)
print('predicted sold items:', predicted_sold_items)

# Calculate MSE
mse_visitors = mean_squared_error(scaled_Data_mms_df['Number of Visitors'].values.reshape(-1, 1), predicted_visitors)
mse_sold_items = mean_squared_error(scaled_Data_mms_df['Number of sold items'].values.reshape(-1, 1), predicted_sold_items)

print('MSE of visitors: ', mse_visitors)
print('MSE of sold items: ', mse_sold_items)
```
```
predicted visitors: [13.201084277969446, 14.208477082306558, 8.164120256283885, 7.156727451946773, 7.156727451946773, 0.1049778215869886, 0.1049778215869886, 0.1049778215869886, 9.171513060620997, 10.178905864958109]
predicted sold items: [4398.808345002317, 2932.5289985865347, 549.8250606608892, 6048.3726097200715, 1099.6798155668075, 733.109978962862, 0, 549.8250606608892, 1430.5090931018683, 2565.959161982589]

MSE of visitors:  65.92838783134934
MSE of sold items:  7549335.969611017
```

---
### 결론
이번에 진행한 회귀 분석은 머신 러닝에서 주로 사용하는 사이킷런 라이브러리의 알고리즘만 빌려와 주어진 데이터에서 유의미한 인사이트를 추출하는데에 그쳤다. 하지만 더 나아가 레이블 제공, 오차 줄이기, 데이터 크롤링 봇 제작에 활용 등 기능을 향상시켜 많은 유저들이 사용 가능한 프로그램(소스 코드 공개)이 될 수 있게끔 만들어 보겠다.
