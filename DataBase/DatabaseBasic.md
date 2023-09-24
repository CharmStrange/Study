# 데이터베이스 모델링 / 데이터 모델링

> 정보 시스템 구축을 위해 ***(분석) - (설계) - (구현) - (시험) - (유지 및 보수)*** 단계를 거친다. ***(분석) - (설계)*** 가 전체 단계 중 가장 중요한 부분이다. 

>  **개념적**, **논리적**, **물리적** 데이터 모델링, **추상화**, **단순화**, **명확화**가 모델링에 있어서 고려해야 할 주요 요소다.

> #### **외부**, **개념**, **내부** Schema Structure
> **외부 스키마**는 데이터베이스를 이용하는 외부, 고객과 같은 사용자 입장에서 고려한 스키마
>
> **개념 스키마**는 데이터베이스 설계 시 외부와 내부에 대한 구조를 고려한 스키마
>
> **내부 스키마**는 데이터베이스를 직접 설계, 관리하는 개발자 혹은 관리자가 데이터베이스의 물리적 구조를 고려한 스키마

> #### 데이터베이스 모델링과 관련되어 필수적으로 알아야 할 용어
> 1. 테이블(table)
> 2. 필드(field=column)
> 3. 레코드(record=row)
> 4. 기본 키(primary key) : 각 행을 구분하는 유일한 특징을 지닌 열
> 5. 외래 키(foreign key) : 테이블 간 관계와 관련됨
> 6. SQL