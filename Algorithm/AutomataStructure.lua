String={}
ExtraString={}
Alphabet={'a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z'}
--
function MakeList(Size)
    if (Size<=26) then
        for i=1, Size, 1 do
            table.insert(String, Alphabet[i])
            --check : print(String[i]) 
        end
        
    
    else 
        for i=1, math.floor(Size/26) do
            for j=1, 26 do
                table.insert(String, Alphabet[j])
                --check : print(String[j]) 
            end
        end
    end
        for i=1, (Size%26) do
            ExtraString[i]=Alphabet[i]
            table.insert(String, ExtraString[i])
        end    
        
    return String
    
end
--
function Pattern(Number) -- Number 1~Size
    if (Number==1) then
        return String
    else
        for i=2, Number, 2 do
            String[i]=String[i-1]
        end
    end
    return String
end
--
function Automata(Size, Number)
    print("BeforeState:")
    for i=1, #MakeList(Size) do
        print(MakeList(Size)[i])
    end
    
    print("\nAfterState:")
    for i=1, #Pattern(Number) do
        print(Pattern(Number)[i])
    end
    return Pattern(Number)  
end
--
function StringQueue(List)
    MinString={"front"}
    for i=1, #List+1 do
        table.insert(MinString, List[i])
        --check : print(MinString[i])
    end
    Front=MinString[1]
    Rear=MinString[#MinString]
    --check : print(Front, Rear)
    --check : print(#MinString)
    return MinString
end
--
--[[#changed, added code#]]
setFunction={}
setFunction["String"]={}
setFunction["newString"]={}
function setFunction.B_State()
    for i=1, 26 do
        table.insert(setFunction["String"], string.char(i+96))
        --check : print(setFunction["String"][i])
    end
    return setFunction["String"]
end
--[[ ASCII 0~127 ]]
function setFunction.Func(KEY) -- KEY : 1 ~ 101
    print("\nInitial String is { a, b, d, ... z }.\nApplying Function... KEY is", KEY)
    for i=1, 26 do
        table.insert(setFunction["newString"], i+KEY)
    end
    return setFunction["newtring"]
end
--
function setFunction.Automata(Number)
    print("BeforeState:")
    for i=1, 26 do
        print(setFunction["String"][i])
    end
    
    setFunction.Func(Number)
    print("AfterState")
    for i=1, 26 do
        print(setFunction["newString"][i])
    end
    return setFunction["newString"]
end
--
