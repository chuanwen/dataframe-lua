-- parse a line from csv file. Code is copied from
-- http://lua-users.org/wiki/LuaCsv
local function ParseCSVLine (line, sep)
	local res = {}
	local pos = 1
	sep = sep or ','
	while true do
		local c = string.sub(line,pos,pos)
		if (c == "") then break end
		if (c == '"') then
			-- quoted value (ignore separator within)
			local txt = ""
			repeat
				local startp,endp = string.find(line,'^%b""',pos)
				txt = txt..string.sub(line,startp+1,endp-1)
				pos = endp + 1
				c = string.sub(line,pos,pos)
				if (c == '"') then txt = txt..'"' end
				-- check first char AFTER quoted string, if it is another
				-- quoted string without separator, then append it
				-- this is the way to "escape" the quote char in a quote. example:
				--   value1,"blub""blip""boing",value3  will result in blub"blip"boing  for the middle
			until (c ~= '"')
			table.insert(res,txt)
			assert(c == sep or c == "")
			pos = pos + 1
		else
			-- no quotes used, just look for the first separator
			local startp,endp = string.find(line,sep,pos)
			if (startp) then
				table.insert(res,string.sub(line,pos,startp-1))
				pos = endp + 1
			else
				-- no separator found -> use rest of string and terminate
				table.insert(res,string.sub(line,pos))
				break
			end
		end
	end
	return res
end

---------------------------------------------------------------------
-- copied from csvisgo:
-- https://github.com/clementfarabet/lua---csv

-- enclose commas and quotes between quotes and escape original quotes
local function escapeCsv(s, separator)
   if string.find(s, '["' .. separator .. ']') then
   --if string.find(s, '[,"]') then
      s = '"' .. string.gsub(s, '"', '""') .. '"'
   end
   return s
end

-- convert an array of strings or numbers into a row in a csv file
local function tocsv(t, separator)
   separator = separator or ','
   local s = ""
   for _,p in pairs(t) do
      s = s .. separator .. escapeCsv(p, separator)
   end
   return string.sub(s, 2) -- remove first comma
end

-- END of copied from csvisgo
-------------------------------------------------------------

local function which(tb, element)
	for k, v in pairs(tb) do
		if v == element then
			return k
		end
	end
	return nil
end

-- given a string x, "guess" its type and return both type and value.
local function guessType(x)
    local value = tonumber(x)
    if value ~= nil then
        return "number", value
    elseif x:lower() == "true" then
        return "boolean", true
    elseif x:lower() == "false" then
        return "boolean", false
    elseif x == "NA" then
        return "number", 0/0
    end
    return "string", x
end

-- convert a string x to the specified type.
local function convert(x, toType)
    if toType == "number" then
		if x == "NA" or x == "." then
			return 0/0 -- R supports NA, torch.Tensor does not, use 0/0
		end
        return tonumber(x)
    elseif toType == "boolean" then
        x = x:lower()
        if x == "true" then
            return 1
        elseif x == "false" then
            return 0
        else
            assert(false, "can't convert to boolean type")
        end
    elseif toType == "string" then
        return x
    else
        assert(false, "unkown type: ".. toType)
    end
end

-- Look up value in levels, if exists, return its id (number), otherwise
-- assign it an id and put it in the dictionary, return the new id.
local function convertToFactor(value, levels)
	local dictionary = assert(levels.dictionary)
	local id = dictionary[value]
	if id ~= nil then
		return id
	end
	-- assign an id to the value
	id = #levels + 1
	dictionary[value] = id
	levels[id] = value
	return id
end

local sprintf = string.format

--------------------------------

torch = require 'torch'
local class = require 'class'
local vector = class("vector")
local dataframe = class("dataframe")

--[[
Note:
1. dataframe uses vector to store string column, uses 1D Tensor for numeric column
2. Both vector and Tensor support clone(), nElements, rows()
--]]

-- deep clone of a table
local function cloneTable(self)
	if self == nil then
		return nil
	end
	assert(type(self) == "table")
	local ans = {}
	for key, value in pairs(self) do
		if type(value) == "table" then
			ans[key] = cloneTable(value)
		else
			ans[key] = value
		end
	end
	return ans
end

-- Now both vector and torch.Tensor support clone()
function vector:clone()
    local ans = vector()
    for i=1,#self do
        ans[i] = self[i]
    end
	return ans
end

-- Now both vector and torch.Tensor support nElement
function vector:nElement()
    return #self
end

-- Add rows() to both vector and torch.Tensor
function vector:rows(rows)
    local ans = vector()
    if type(rows) == "table" then
        for i=1,#rows do
            ans[#ans + 1] = self[rows[i]]
        end
        return ans
    end
	if type(rows) == "userdata" and rows:type() == "torch.DoubleTensor" then
		for i=1,rows:nElement() do
			ans[i] = self[rows[i]]
		end
		return ans
	end
    if type(rows) == "userdata" and rows:type() == "torch.ByteTensor" then
        assert(rows:nElement() == self:nElement())
        for i=1,rows:nElement() do
            if rows[i] == 1 then
                ans[#ans + 1] = self[i]
            end
        end
        return ans
    end
end

-- Add rows() to both vector and torch.Tensor
function torch.Tensor:rows(rows)
    if type(rows) == "table" then
        local ans = torch.zeros(#rows)
        for i=1,#rows do
            ans[i] = self[rows[i]]
        end
        return ans
    end
    if type(rows) == "userdata" and rows:type() == "torch.DoubleTensor" then
        local ans = torch.zeros(rows:nElement())
        for i=1,rows:nElement() do
            ans[i] = self[rows[i]]
        end
        return ans
    end
    if type(rows) == "userdata" and rows:type() == "torch.ByteTensor" then
        return self[rows]
    end
end

function dataframe:__zero()
    self.data = {}
    self.__dim = {}
    self.colNames = {}
    self.colTypes = {}
	self.stringAsFactor = true -- convert string to factor?
	self.levels = {} -- self.levels[j] has levels of jth col (if it's factor)
end

function dataframe:__init(arg)
    self:__zero()
	arg = arg or {}
	if arg.file ~= nil then
        self:readcsv(arg.file, arg)
    elseif type(arg) == "userdata" and arg:type() == "torch.DoubleTensor" then
        self:fromTensor(arg)
    end
end

-- return col j.  "j" can be the name of the col, or a number (jth-column)
function dataframe:col(j, param)
	j = self:__j(j)
	local ans
	if self.dataIsTensor then
		ans = self.data:select(2, j)
	else
		ans = self.data[j]
	end
	if param ~= nil and param.clone == true then
		return ans:clone()
	else
		return ans
	end
end

-- return i-th row of the data
function dataframe:row(i)
    local ans = {}
    for j=1,self.__dim[2] do
        ans[j]=self.data[j][i]
		if self.colTypes[j] == "factor" then
			ans[j] = (self.levels[j])[ans[j]]
		end
    end
    return ans
end

local function isHeader(fields)
	for _, field in pairs(fields) do
		local valType
		valType, _ = guessType(field)
		if valType == "number" or valType == "boolean" then
			return false
		end
	end
	return true
end

function dataframe:readcsv(url, param)
	local fileName = url
	param = param or {}
	if string.match(url, "^http") ~= nil then
		fileName = 'tmp_' .. math.random(10000) .. '.csv'
		assert(os.execute("which curl > /dev/null"))
		assert(os.execute(sprintf("curl -s -o %s %s", fileName, url)))
		self:readcsvFile(fileName, param)
		assert(os.execute('rm ' .. fileName))
		return self
	end
	return self:readcsvFile(fileName, param)
end

local function lineCount(fileName, skipEmptyLines)
    local fileHandle = assert(io.open(fileName, "r"))
    local line = fileHandle:read("*l")
    local ans = 0
	if skipEmptyLines==true then
		while line ~= nil do
			ans = ans + 1
			line = fileHandle:read("*l")
		end
	else
		while line ~= nil and line ~= '' do
	        ans = ans + 1
	        line = fileHandle:read("*l")
		end
    end
    return ans
end

-- read csv file, return an object similar to data.frame in R
function dataframe:readcsvFile(fileName, param)
    local fileHandle = assert(io.open(fileName, "r"))
	self:__zero()

	local skipEmptyLines = true
	if param.skipEmptyLines ~= nil then
		skipEmptyLines = param.skipEmptyLines
	end
	local nrows = lineCount(fileName, skipEmptyLines)
	local sep = param.sep or ","

	if param.stringAsFactor ~= nil then
		self.stringAsFactor = param.stringAsFactor
	end

	self.dataIsTensor = false
	if param.returnTensor == true then
		assert(self.stringAsFactor == true)
		self.dataIsTensor = true
	end

    local first =  ParseCSVLine(fileHandle:read("*l"), sep)

	local header = isHeader(first)
	if  param.header ~= nil then
		header = param.header
	end

    if header then
        self.colNames = first
        first =  ParseCSVLine(fileHandle:read("*l"), sep)
        assert(#self.colNames == #first)
		nrows = nrows - 1
    end

	if self.dataIsTensor then
		self:initTensorData(first, nrows)
	else
		self:initData(first, nrows)
	end

    for j=1,#first do
        local valType, val
        valType, val = guessType(first[j])
        self.colTypes[j] = valType
		if valType == "string" and self.stringAsFactor then
			self.colTypes[j] = "factor"
			self.levels[j] = {dictionary={}}
			val = convertToFactor(val, self.levels[j])
		end
		self:__set(1, j, val)
    end

    local line = fileHandle:read("*l")
    local i = 1
    while line ~= nil do
		if line ~= '' then
	        fields = ParseCSVLine(line, sep)
	        assert(#fields == #first)
	        i = i + 1
	        for j=1,#fields do
				local valType = self.colTypes[j]
				local raw = fields[j]
				if valType ~= "factor" then
					raw = convert(raw, valType)
				else
					raw = convertToFactor(raw, self.levels[j])
				end
				self:__set(i, j, raw)
	        end
		else
			if skipEmptyLines ~= true then
				break
			end
		end
        line = fileHandle:read("*l")
    end
    fileHandle:close()
	for j,valType in ipairs(self.colTypes) do
		if valType == "factor" then
			self:rebaseFactor(j)
		end
	end
	if self.dataIsTensor then
		return self:toTensor()
	end
    self:__updateDerivedFields()
	return self
end

function dataframe:initData(first, nrows)
	for j = 1, #first do
		local valType
		valType = guessType(first[j])
		if valType == "number" or valType == "boolean" then
			self.data[j] = torch.zeros(nrows)
		elseif valType == "string" and self.stringAsFactor then
			self.data[j] = torch.zeros(nrows)
		else
			self.data[j] = vector()
		end
	end
end

function dataframe:initTensorData(first, nrows)
	self.data = torch.zeros(nrows, #first)
end

function dataframe:__set(i, j, val)
	if self.dataIsTensor==true then
		self.data[i][j] = val
	else
		self.data[j][i] = val
	end
end

function dataframe:hasColNames()
	return #self.colNames == #self.colTypes
end

-- save dataframe to csv file
function dataframe:writecsv(fileName, param)
    local fileHandle = assert(io.open(fileName, "w"))
	param = param or {}
	local header = true
	if param.header ~= nil then
		header = param.header
	end
	local sep = param.sep or ","
    if header and self:hasColNames() then
        fileHandle:write(tocsv(self.colNames, sep))
        fileHandle:write("\n")
    end
    for i=1,self.__dim[1] do
        fileHandle:write(tocsv(self:row(i), sep))
        fileHandle:write("\n")
    end
    fileHandle:close()
end

-- unique of a table whose content is a list of elements (with key=1,2,...)
local function unique(tb)
	local ans = {}
	local dict = {}
	for _, v in ipairs(tb) do -- use ipairs, assume key is 1,2,...
		dict[v] = 1
	end
	for k, _ in pairs(dict) do -- use pairs, as key can be anything.
		ans[#ans + 1] = k
	end
	return ans
end

-- set dataframe col names to newnames
function dataframe:setnames(newnames, oldnames)
	local origColNames = cloneTable(self.colNames)
	if oldnames == nil then
		for j=1,math.min(#newnames, #self.colTypes) do
			self.colNames[j] = newnames[j]
		end
	else
    	assert(#oldnames == #newnames)
	    for i=1,#oldnames do
			local j = self:__j(oldnames[i])
	        self.colNames[j] = newnames[i]
	    end
	end
	if #unique(self.colNames) ~= #self.colNames then
		self.colNames = origColNames
		assert(false, "col name should be unique")
	end
	return self
end

function dataframe:names(j)
	if j == nil then
    	return self.colNames
	else
		assert(type(j) == "number")
		return self.colNames[j]
	end
end

-- create dataframe from a 2D tensor
function dataframe:fromTensor(t)
    assert(t:dim() == 2)
    self:__zero()
    self.__dim = {t:size(1), t:size(2)}
    for j=1,self.__dim[2] do
        self.data[j] = t:select(2, j):clone()
        self.colTypes[j] = "number"
    end
end

function dataframe:numericColIndex()
	local ans = {}
	local numericTypes = {"number", "boolean", "factor"}
	for j=1,#self.colTypes do
		if which(numericTypes, self.colTypes[j]) then
			ans[#ans + 1] = j
		end
	end
	return ans
end

-- put numeric cols together into a 2D tensor with proper dimension.
function dataframe:toTensor()
	if self.dataIsTensor==true then
		return self.data, self.levels
	end
	local numericColIndex = self:numericColIndex()
    assert(self.__dim[1] > 0 and #numericColIndex > 0)
    local ans = torch.zeros(self.__dim[1], #numericColIndex)
	local levels={}
    for j=1,#numericColIndex do
        local j0 = numericColIndex[j]
        ans:select(2, j):copy(self.data[j0])
		if self.colTypes[j0] == "factor" then
			levels[j] = cloneTable(self.levels[j0])
		end
    end
    return ans, levels
end

-- extract a subset of dataframe
function dataframe:cols(cols, param)
	local clone = false
	if param ~= nil and param.clone ~= nil then
		clone = param.clone
	end
    local ans = dataframe()
	local j0 = 1
    for _, j in pairs(cols) do
        j = self:__j(j)
        if clone then
            ans.data[j0] = self.data[j]:clone()
			ans.levels[j0] = cloneTable(self.levels[j])
        else
            ans.data[j0] = self.data[j]
			ans.levels[j0] = self.levels[j]
        end
        ans.colTypes[j0] = self.colTypes[j]
		ans.colNames[j0] = self.colNames[j]
		j0 = j0 + 1
    end
    ans:__updateDerivedFields()
    return ans
end

function dataframe:rows(rows)
    local ans = dataframe()
    for j=1,self.__dim[2] do
        ans.data[j] = self.data[j]:rows(rows)
    end
    ans.colNames = self.colNames
    ans.colTypes = self.colTypes
	ans.levels = cloneTable(self.levels)
    ans:__updateDerivedFields()
    return ans
end

function dataframe:sub(rows, cols)
    if rows == nil then
        return self:cols(cols, {clone=true})
    end
    if cols == nil then
        return self:rows(rows)
    end
	-- no need to set clone in cols(...) here as rows(..) copy data
    return self:cols(cols):rows(rows)
end

function dataframe:get(row, col)
	local j = self:__j(col)
	return self:row(row)[j]
end

function dataframe:set(row, col, value)
	local j = self:__j(col)
	if self.colTypes[j] == "factor" and
	   self.levels[j].dictionary[value] == nil then
		value = convertToFactor(value, self.levels[j])
	end
	self:__set(row, j, value)
	return self
end

function dataframe:__updateDerivedFields()
    self.__dim = {self.data[1]:nElement(), #self.data}
end

function dataframe:type()
    return "dataframe"
end

function dataframe:__j(j)
    if type(j) == "string" then
        j = self:colIndex(j)
    end
    assert(type(j) == "number")
    return j
end

function dataframe:nrow()
    return self.__dim[1]
end

function dataframe:ncol()
    return self.__dim[2]
end

function dataframe:dim()
    return self.__dim
end

function dataframe:colIndex(colName)
	local j = which(self.colNames, colName)
	assert(j ~= nil, 'No column is named as ' .. colName)
	return j
end

local function rowFormat(t)
    return table.concat(t, "\t") .. "\n"
end

function dataframe:__tostring()
	if #self.colTypes == 0 or #self.__dim == 0 then
		return "empty dataframe"
	end
    local ans = rowFormat(self:names())
    for i = 1, math.min(30, self.__dim[1]) do
        ans = ans .. rowFormat(self:row(i))
    end
    if self.__dim[1] > 30 then
        ans = ans .. "....\n" .. rowFormat(self:names())
    end
    return ans
end

function dataframe:levels(j)
	if j == nil then
		return self.levels
	end
	assert(self.colTypes[j] == "factor")
	return self.levels[j]
end

function dataframe:colType(j)
	assert(type(j) == "number" and j >= 1 and j <= self.__dim[2])
	return self.colTypes[j]
end

-- ToDO:
-- 1. check value before set self.data[j]
-- 2. convert value to Tensor if value is not a Tensor.
function dataframe:addCols(dat, param)
	param = param or {}
	assert(type(dat) == "table")
	local j = self:ncol() + 1
	for key, value in pairs(dat) do
		if type(key) == "string" then
			self.colName[j] = key
		end
		self.data[j] = value
		self.colTypes[j] = type(value[1])
	end
end

local function seq(from, to, step)
	step = step or 1
	assert(step > 0 and from <= to)
	local ans = {}
	local j = from
	local i = 1
	while j <= to do
		ans[i] = j
		i = i + 1
		j = j + step
	end
	return ans
end

local function shallowCopy(dest_tb, src_tb)
	for k, v in pairs(src_tb) do
		dest_tb[k] = v
	end
end

function dataframe:removeCols(cols)
	param = param or {}
	assert(type(cols) == "table")
	local keepCols = seq(1, self:ncol())
	for _, j in ipairs(cols) do
		j = self:__j(j)
		keepCols[j] = nil
	end
	shallowCopy(self, self:cols(keepCols))
	return self
end

function VectorEq(ta, tb)
	if #ta ~= #tb then
		return false
	end
	for k, v in ipairs(ta) do
		if tb[k] ~= v then
			return false
		end
	end
	return true
end

function dataframe:rebaseFactor(col)
	local j = self:__j(col)
	assert(self.colTypes[j] == "factor")
	local levels = self.levels[j]
	local origLevels = cloneTable(levels)
	table.sort(levels)
	if not VectorEq(levels, origLevels) then
		local mapData = torch.data(torch.zeros(#levels))
		local dictionary = levels.dictionary
		for i, level in ipairs(levels) do
			local oldi = dictionary[level]
			mapData[oldi-1] = i
			dictionary[level] = i
		end
		self:col(j):apply(function(x) return mapData[x-1] end)
	end
	return self
end

-- convert a
function dataframe:asFactor(col)
	local j = self:__j(col)
	local n = self:nrow()
	local origCol = self.data[j]
	assert(#origCol == n)
	self.colTypes[j] = "factor"
	local newCol = torch.zeros(n)
	local levels = {dictionary={}}
	for i, value in ipairs(origCol) do
		newCol[i] = convertToFactor(value, levels)
	end
	self.levels[j] = levels
	self.data[j] = newCol
	return self:rebaseFactor(j)
end

return dataframe
