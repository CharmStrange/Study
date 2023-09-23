> ## 새로운 데이터베이스 생성하기
> **데이터베이스**(->**스키마**) 파일을 생성한 뒤 작업을 시작한다.
> (+MySQL 같은 프로그램은 SQL문을 작성하지 않아도 체크만 해 주면 SQL문이 자동으로 작성됨)
> ```SQL
> CREATE 데이터베이스명;
> USE 데이터베이스명;
> ```
> 1. 테이블 만들고 여러 설정 해주기.
> ```SQL
> CREATE TABLE 테이블명(
>    [필드명] [데이터 형] [설정],
>    [필드명] [데이터 형식] [설정],
>    [필드명] [데이터 형식] [설정],
>    ...
> );
> ```
>
> 2. **CRUD** : CREATE, READ, UPDATE, DELETE
> - **SELECT**
> ```SQL
> SELECT * FROM 테이블명; --전체 선택
> SELECT 필드명 FROM 테이블명; --필드명 선택
> SELECT 필드명 FROM 테이블명 WHERE 조건; --조건에 따른 데이터를 선택
> ```
> - **INSERT**
> ```SQL
> INSERT INTO 테이블명(필드명) VALUES(값);
> ```
> - **UPDATE**
> ```SQL
> UPDATE 테이블명 SET 변경문 WHERE 조건;
> ```
> - **DELETE**
> ```SQL
> DELETE FROM 테이블명 WHERE 조건;
> ```
