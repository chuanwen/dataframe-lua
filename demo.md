

```lua
dataframe = require("dataframe")
```


```lua
-- Initialze dataframe from a csv file
x = dataframe()
x:readcsv("test.csv")
print(x)
```




    
    5.1	3.5	1.4	0.2	Iris-setosa
    4.9	3	1.4	0.2	Iris-setosa
    6.2	3.4	5.4	2.3	Iris-virginica
    5.9	3	5.1	1.8	Iris-virginica
    6.5	2.8	4.6	1.5	Iris-versicolor
    4.7	3.2	1.3	0.2	Iris-setosb
    





```lua
-- Set col names
x:setnames({"V1", "SepWid", "PetLen", "PetWid", "Species"})
-- Set col names for specific cols
x:setnames({"SepLen"}, {1})

print(x)
```




    SepLen	SepWid	PetLen	PetWid	Species
    5.1	3.5	1.4	0.2	Iris-setosa
    4.9	3	1.4	0.2	Iris-setosa
    6.2	3.4	5.4	2.3	Iris-virginica
    5.9	3	5.1	1.8	Iris-virginica
    6.5	2.8	4.6	1.5	Iris-versicolor
    4.7	3.2	1.3	0.2	Iris-setosb
    





```lua
-- Get a single column, return is either a 1D tensor or a vector of strings.
print(x:col(2))
print(x:col("Species"))
```




     3.5000
     3.0000
     3.4000
     3.0000
     2.8000
     3.2000
    [torch.DoubleTensor of size 6]
    







     1
     1
     4
     4
     3
     2
    [torch.DoubleTensor of size 6]
    





```lua
-- Get one or more columns, return is a dataframe
print(x:cols{1,2})
-- You can use col names
print(x:cols{"SepLen", "Species"})
-- You can also mix col index and col names
-- print(x:cols{1, "Species"})
```




    SepLen	SepWid
    5.1	3.5
    4.9	3
    6.2	3.4
    5.9	3
    6.5	2.8
    4.7	3.2
    
    SepLen	Species
    5.1	Iris-setosa
    4.9	Iris-setosa
    6.2	Iris-virginica
    5.9	Iris-virginica
    6.5	Iris-versicolor
    4.7	Iris-setosb
    





```lua
-- Set clone to true means the colmns in the return dataframe are clone of
-- the original columns; otherwise the columns reference the original columns.
-- Default is clone=false
print(x:cols({1, "Species"}, {clone=true}))
```




    SepLen	Species
    5.1	Iris-setosa
    4.9	Iris-setosa
    6.2	Iris-virginica
    5.9	Iris-virginica
    6.5	Iris-versicolor
    4.7	Iris-setosb
    
    





```lua
-- Get some rows of the dataframe, returned dataframe is a copy of
-- some rows of the original dataframe.
print(x:rows{1,2})
```




    SepLen	SepWid	PetLen	PetWid	Species
    5.1	3.5	1.4	0.2	Iris-setosa
    4.9	3	1.4	0.2	Iris-setosa
    
    





```lua
-- Get a range of rows
print(x:rows(torch.range(1,5)))
```




    SepLen	SepWid	PetLen	PetWid	Species
    5.1	3.5	1.4	0.2	Iris-setosa
    4.9	3	1.4	0.2	Iris-setosa
    6.2	3.4	5.4	2.3	Iris-virginica
    5.9	3	5.1	1.8	Iris-virginica
    6.5	2.8	4.6	1.5	Iris-versicolor
    
    





```lua
-- Use logical operation to select some rows
local rowIndicator = torch.le(x:col("SepLen"), 5.0)
print(x:rows(rowIndicator))
```




    SepLen	SepWid	PetLen	PetWid	Species
    4.9	3	1.4	0.2	Iris-setosa
    4.7	3.2	1.3	0.2	Iris-setosb
    





```lua
-- combine rows and cols selection together
-- return dataframe is a copy of some rectangle area of the original dataframe.
print(x:sub({1,2}, {"SepLen", "Species"}))
print(x:sub(rowIndicator, {"SepLen", "Species"}))
```




    SepLen	Species
    5.1	Iris-setosa
    4.9	Iris-setosa
    
    SepLen	Species
    5.1	Iris-setosa
    4.9	Iris-setosa
    6.2	Iris-virginica
    5.9	Iris-virginica
    6.5	Iris-versicolor
    4.7	Iris-setosb
    





```lua
-- You can convert all numeric columns in the dataframe to a 2D tensor
t = x:toTensor()
print(t)
```




     5.1000  3.5000  1.4000  0.2000  1.0000
     4.9000  3.0000  1.4000  0.2000  1.0000
     6.2000  3.4000  5.4000  2.3000  4.0000
     5.9000  3.0000  5.1000  1.8000  4.0000
     6.5000  2.8000  4.6000  1.5000  3.0000
     4.7000  3.2000  1.3000  0.2000  2.0000
    [torch.DoubleTensor of size 6x5]
    





```lua
--- Initialize a dataframe from a 2D tensor
print(dataframe(t))
```




    
    5.1	3.5	1.4	0.2	1
    4.9	3	1.4	0.2	1
    6.2	3.4	5.4	2.3	4
    5.9	3	5.1	1.8	4
    6.5	2.8	4.6	1.5	3
    4.7	3.2	1.3	0.2	2
    
    





```lua
-- Save dataframe to a csv file.
x:writecsv("test_copy.csv")
os.execute("cat test_copy.csv && rm test_copy.csv")
```




    SepLen,SepWid,PetLen,PetWid,Species
    5.1,3.5,1.4,0.2,Iris-setosa
    4.9,3,1.4,0.2,Iris-setosa
    6.2,3.4,5.4,2.3,Iris-virginica
    5.9,3,5.1,1.8,Iris-virginica
    6.5,2.8,4.6,1.5,Iris-versicolor
    4.7,3.2,1.3,0.2,Iris-setosb





```lua
-- Initialize dataframe from a remote csv file (need to have curl installed)
x2 = dataframe({file="http://www.stat.ucla.edu/projects/datasets/birds.dat"})
print(x2:rows{2,5,10})
```




    SITE	Elevation	Profile Area	Height	Half-height	Latitude	Longitude	No. Species	Total density
    Goshen	45	36.8	21.5	9.3	44.03	122.96	21	7.9
    Redding	nan	21.2	11.5	2.3	40.64	122.28	21	7.25
    Sonoma	147	nan	nan	nan	38.33	122.5	26	3.255
    



