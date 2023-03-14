local function AND(p, q)
    if (p==0 and q==0) or (p==0 and q==1) or (p==1 and q==0) then
        return 0
    else if (p==1 and q==1) then
        return 1
    else
        return "Failed"
    end end
end

local function OR(p, q)
    if (p==0 and q==0) then
        return 0
    else if (p==1 and q==1) or (p==0 and q==1) or (p==1 and q==0) then
        return 1
    else
        return "Failed"
    end end
end

local function BUFFER(p, q)
    if (p==0 and q==0) or (p==0 and q==1) or (p==1 and q==0) or (p==1 and q==1) then
        return p, q
    else
        return "Failed"
    end
end

local function NOT(p, q)
    if (p==0 and q==0) then
        return 1, 1
    else if (p==0 and q==1) then
        return 1, 0
    else if (p==1 and q==0) then
        return 0, 1
    else if (p==1 and q==1) then
        return 0, 0
    else
        return "Failed"
    end end end end
end

local function NAND(p, q)
    if (p==0 and q==0) or (p==0 and q==1) or (p==1 and q==0) then
        return 1
    else if (p==1 and q==1) then
        return 0
    else
        return "Failed"
    end end
end

local function NOR(p, q)
    if (p==0 and q==0) then
        return 1
    else if (p==1 and q==1) or (p==0 and q==1) or (p==1 and q==0) then
        return 0
    else
        return "Failed"
    end end
end

local function XOR(p, q)
    if (p==0 and q==0) or (p==1 and q==1) then
        return 0
    else if (p==0 and q==1) or (p==1 and q==1) then
        return 1
    else
        return "Failed"
    end end
end

local function XNOR(p, q)
    if (p==0 and q==0) or (p==1 and q==1) then
        return 1
    else if (p==0 and q==1) or (p==1 and q==0) then
        return 0
    else
        return "Failed"
    end end
end