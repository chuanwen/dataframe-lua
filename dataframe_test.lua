local say = require("say")
local function tensorEq(state, arguments)
    local ans = false
    local ta = arguments[1]
    local tb = arguments[2]
    local eps = arguments[3] or 1e-9 -- two numbers within eps are defined as "equal"
    if type(ta) ~= "userdata" or type(tb) ~= "userdata" then
        return false
    end
    if not (ta:isTensor() and tb:isTensor()) then
        return false
    end
    if not ta:isSameSizeAs(tb) then
        return false
    end
    return torch.all(torch.lt(torch.abs(ta-tb), eps))
end

say:set("assertion.tensorEq.positive", "Expected %s\nGot %s\n")
say:set("assertion.tensorEq.negative", "Expected %s\nGot %s\n")
assert:register("assertion", "tensorEq", tensorEq, "assertion.tensorEq.positive", "assertion.tensorEq.negative")

describe("dataframe test", function()

    local csvContent = [==[5.1,3.5,1.4,0.2,Iris-setosa
    4.9,3.0,1.4,0.2,Iris-setosa
    6.2,3.4,5.4,2.3,Iris-virginica
    5.9,3.0,5.1,1.8,Iris-virginica
    6.5,2.8,4.6,1.5,Iris-versicolor
    4.7,3.2,1.3,0.2,Iris-setosb]==]
    local csvFile
    local df

    local x
    local firstrow
    local lastrow
    local firstcol
    local lastcol

    setup(function()
        csvFile = "_dataframe_test_" .. math.random() .. ".csv"
        local file = io.open(csvFile, "w")
        file:write(csvContent)
        file:close()
        df = require("dataframe")
    end)

    teardown(function()
        os.remove(csvFile)
        csvFile = nil
        df = nil
    end)

    before_each(function()
        x = df():readcsv(csvFile):setnames({"V1", "V2", "V3", "V4", "V5"})
        firstrow = {5.1,3.5,1.4,0.2,"Iris-setosa"}
        lastrow = {4.7,3.2,1.3,0.2,"Iris-setosb"}
        firstcol = torch.Tensor{5.1, 4.9, 6.2, 5.9, 6.5, 4.7}
        lastcol = torch.Tensor{1,1,4,4,3,2} -- stringAsFactor = true
    end)

    it("df:dim()", function()
        assert.same(x:dim(), {6,5})
    end)

    it("df:col()", function()
        assert.tensorEq(x:col(1), firstcol)
        assert.tensorEq(x:col(x:ncol()), lastcol)
        assert.tensorEq(x:col("V1"), firstcol)
    end)

    it("df:row()", function()
        assert.same(x:row(1), firstrow)
        assert.same(x:row(x:nrow()), lastrow)
    end)

    it("df:cols()", function()
        assert(x:cols({1,3}):ncol(), 2)
        assert(x:cols({"V1","V5"}):ncol(), 2)
        assert.tensorEq(x:cols({3, 2, 1}):col(3), firstcol)
        assert.tensorEq(x:cols({"V1","V5"}):col(2), lastcol)
    end)

    it("df:rows()", function()
        assert.same(x:rows{1,2,3}:row(1), x:row(1))
        assert.same(x:rows(torch.range(1,3)):nrow(), 3)
        assert.same(x:rows(torch.range(1,3)):row(3), x:row(3))
        local indicator = torch.lt(x:col(1), 5)
        assert.same(x:rows(indicator):nrow(), torch.sum(indicator))
    end)

    it("df:sub()", function()
        local y = x:sub(torch.range(1,3), {1, 5})
        assert.tensorEq(y:col(1), firstcol[{{1,3}}])
        assert.tensorEq(y:col(2), lastcol[{{1,3}}])

        local indicator = torch.lt(firstcol, 5.0)
        y = x:sub(indicator, {1, "V5"})
        assert.tensorEq(y:col(1), firstcol[indicator])
        assert.tensorEq(y:col(2), lastcol[indicator])

        local n = x:nrow()
        y = x:sub({1, 2, n}, nil)
        assert.same(y:row(1), firstrow)
        assert.same(y:row(3), lastrow)
    end)

    it("df:get()", function()
        local n = x:nrow()
        local p = x:ncol()
        assert.same(x:get(1,1), firstcol[1])
        assert.same(x:get(n,1), firstcol[n])
        assert.same(x:get(n, p), lastrow[p])
    end)

    it("df:set()", function()
        local p = x:ncol()
        assert.same(x:set(1,1, 1.5):get(1,1), 1.5)
        assert.same(x:set(2,p, "new_value"):get(2,p), "new_value")
    end)

    it("df:setnames()", function()
        x:setnames({"SepLen", "SepWid", "V3", "V4", "Species"})
        assert.same(x:names(1), "SepLen")
        assert.same(x:names(), {"SepLen", "SepWid", "V3", "V4", "Species"})
        assert.tensorEq(x:col("SepLen"), x:col(1))
        x:setnames({"PetWidth"}, {"V4"})
        assert.same(x:names(4), "PetWidth")
        x:setnames({"PetLen", "PetWid"}, {3,4})
        assert.same(x:names(3), "PetLen")
        assert.same(x:names(4), "PetWid")
    end)

    it("df:readcsv() -- set stringAsFactor=false", function()
        local x = df():readcsv(csvFile, {stringAsFactor=false})
        assert.same(x:dim(), {6,5})
        assert.same(x:col(5)[1], firstrow[5])
        assert.same(x:col(5)[6], lastrow[5])
    end)

    it("df:readcsv() -- set returnTensor=true", function()
        local t, meta = df():readcsv(csvFile, {returnTensor=true}) -- this is more efficient than line below
        local t2, meta2 = x:toTensor()
        assert.same(type(t), "userdata")
        assert.same(t:size(1), 6)
        assert.same(t:size(2), 5)
        assert.tensorEq(t:select(2, 1), firstcol)
        assert.tensorEq(t:select(2, 5), lastcol)
        assert.tensorEq(t, t2)
        assert.same(meta, meta2)
    end)

    it("df:readcsv() -- read remote space separated file", function()
        local url = "http://www.stats.ox.ac.uk/pub/datasets/csb/ch11b.dat"
        -- local y = df():readcsv(url, {sep=" "}) -- fields separated by space
        -- assert.same(y:dim(), {100, 5})
        -- assert.same(y:row(1), {1, 307, 930, 36.58, 0})
    end)

    it("df:writecsv() -- write local tab separated file", function()
        local tsvFile = "_dataframe_testwritecsv_" .. math.random() .. ".tsv"
        colnames = {"SepLen", "SepWid", "PetLen", "PetWid", "Species"}
        x:setnames(colnames)
        x:writecsv(tsvFile, {sep="\t"})
        local y = df():readcsv(tsvFile, {sep="\t"})
        assert.same(y:names(), colnames)
        assert.same(x:dim(), y:dim())
        assert.tensorEq(x:col(1), y:col(1))
        assert.same(x:row(1), y:row(1))
        local n = x:nrow()
        local p = x:ncol()
        assert.same(x:row(n), y:row(n))
        assert.tensorEq(x:col(p), y:col(p))
        os.remove(tsvFile)
    end)
end)
