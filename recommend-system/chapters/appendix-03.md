
# Hive SQL 参考

## Partition

### 查看partition信息

```SQL
show partitions <table name>
```

## Join

### Inner Join

```SQL
SELECT  
    * 
FROM 
    <Table 1> u  
INNER JOIN  
    <Table 2> d  
ON 
    d.id = u.department_id;
```
此时返回的是两个表的在department_id上的交集


## Analyze Table

