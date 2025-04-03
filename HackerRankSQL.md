* 20250403

"
Query all columns for all American cities in the CITY table with populations larger than 100000. The CountryCode for America is USA.
"
>>>
```
SELECT * FROM CITY WHERE POPULATION>100000 AND COUNTRYCODE ='USA'
```

"
Query the NAME field for all American cities in the CITY table with populations larger than 120000. The CountryCode for America is USA.
"
>>>
```
SELECT NAME FROM CITY WHERE POPULATION>120000 AND COUNTRYCODE ='USA'
```