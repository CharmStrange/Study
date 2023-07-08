# 미분
-> 어느 순간(***h->0***)의 변화량을 구할 수 있다. 

<img width="86" alt="image" src="https://github.com/CharmStrange/Study/assets/105769152/13254ddb-9cc5-4e0c-b4a7-7e6fd6800543">

```
def numerical_diff(f, x):
    h = 1e-4
    return (f(x+h) - f(x-h) ) / (2*h)
```
차분을 구해 이것으로 미분하는 수치 미분 함수이다.

```
def function_2(x):
    return x[0]**2 + x[1]**2
```
배열을 받아 편미분을 계산하는 함수이다.
