# Dataframe for Lua/Torch7
Dataframe is inspired by data.frame in R. You can use it to load and save large csv 
or tsv file like data.frame in R.  It can be converted to or from 2D Tensor.

## Requirements
- torch

## Installation
```
luarocks install dataframe
```

## Usage

Read/write csv/tsv file
```
require 'dataframe'

df = dataframe():readcsv("test.csv") -- By default, auto detect header in the file 
                                     -- and convert string to factor 
df:writecsv("test.tsv", {sep="\t"})  -- write to tsv file
print(df)                            -- print top 30 rows

df = dataframe():readcsv("test.csv", {stringAsFactor=false}) -- not to convert string to factor
df:asFactor(5) -- convert 5-th col to factor col (from string or number col)
```

Convert to/from 2D Tensor
```
t = df:toTensor()  -- t is a 2D Tensor with all numeric cols in df
                   -- note: number, boolean and factor are considered as "numeric"
df2 = dataframe(t) -- create a dataframe from tensor
```

Get a specific row or col
```
df:col(1)        -- return first column
df:col("Width")  -- return the "Width" column 
df:row(2)    -- return a copy of 2nd row
df:get(1, 2) -- return value of the cell (row 1, col 2)
df:get(1, "Width") -- return value of the cell (row1, col "Width")
```

Get a sub dataframe
```
df:cols({1,2,"Width"}) -- return a dataframe that contains the 3 cols (same underline cols)
df:cols({1,2}, {clone=true}) -- similar to above, except the underline cols are cloned.

df:rows({4,10}) -- return a dataframe that has a copy of row 4 and 10.
df:rows(torch.range(5, 100)) -- return a dataframe that has a copy of row 5 to 100.
df:rows(torch.lt(df:col(1), 5.0)) -- return a dataframe that has a copy of rows whose first col is less than 5.0

df:sub(torch.range(1, 10), {1,3, "Width"}) -- a copy of rows 1-10 for the 3 cols.
```

Get/set column names
```
df:names()  -- get names for all cols
df:names(1) -- get first col name.

df:setnames({"SepLen", "SepWid", "V3", "V4"}) -- set names for first 4 columns
df:setnames({"Len", "Wid"}, {3, "V4"})        -- set names for col 3 and col "V4", respectively
```

## [Demo](demo.md)
