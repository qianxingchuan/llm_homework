# 总览库存

Method: POST

Request:

```javascript
{
 "searchSku":"苹果",
 "quality":"合格"
}
```

接口行为：

查询指定的货品的库存概览

参数说明：

| 参数      | 是否必选 | 描述                                                                                             |
| --------- | -------- | ------------------------------------------------------------------------------------------------ |
| searchSku | 否       | 可以输入名字模糊搜索，也可以填入code                                                             |
| quality   | 否       | 支持用户用自然语言来表示合格、残次、待检。对应的<br />接口内部的枚举是 GENIUNE, DEFECTIVE, GRADE |

Response:

```javascript
{
 "success":true,
 "data":[
   {
     "skuCode":"Apple",
     "quality":"GENIUNE",
     "qty":"100",
     "occupied": 30
   }
  ]
}
```

# 物理库存

Note: 标注当前库存真实所在的位置

Method: POST

Request:

```javascript
{
 "searchSku":"苹果",
 "quality":"合格",
 "binCode":"L-01-01",
 "batchInfo":{
   "produceDate":"2025-01-01",
   "productionNo":"D0100210"
  },
 "area":"N01"
}

```

接口行为：

查询指定的货品的库存，包含实际的位置信息

参数说明：

| 参数      | 是否必选 | 描述                                                                                             |
| --------- | -------- | ------------------------------------------------------------------------------------------------ |
| searchSku | 否       | 可以输入名字模糊搜索，也可以填入code                                                             |
| quality   | 否       | 支持用户用自然语言来表示合格、残次、待检。对应的<br />接口内部的枚举是 GENIUNE, DEFECTIVE, GRADE |
| binCode   | 否       | 储位Code                                                                                         |
| area      | 否       | 库区的code或者库区的名称（支持模糊搜索）                                                         |
| batchInfo | 否       | 批次属性指定，批次属性当前有生产日期（produceDate）、生产批次（produceDate）                     |

Response:

```javascript
{
 "success":true,
 "data":[
   {
     "skuCode":"Apple",
     "quality":"GENIUNE",
     "binCode":"L-01-01",
     "areaCode":"N01",
     "areaName":"常规01区",
     "batchInfo":{
	   "produceDate":"2025-01-01",
	   "productionNo":"D0100210"
    },
     "qty":"100",
     "occupied": 30
   }
  ]
}
```
