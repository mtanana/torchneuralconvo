ModelTracker = {}

--(from mtanana) this is my own custom model tracker  I'd be happy to open source it if someone wants it

--table is the lua table that will become json and the funct is the function id
function ModelTracker.sendJsonObj(table,funct)

    local endpoint = "http://camber:8080/modeltracker/tracking.jsp"
    -- load required modules
    local http = require("socket.http") --luasocket
    local ltn12 = require("ltn12")
    local mime = require("mime")
    local io = require("io")
    local json = require("json") -- luajson
    local url = require("socket.url")
    
    -- Create a Lua table to represent our entity to save
    --- This is from our doc REST example: http://docs.kinvey.com/rest-appdata.html
    --jamesBond = { ["firstName"] = "James", ["lastName"] =  "Bond", ["email"] = "james.bond@mi6.gov.uk", ["age"] = 34 }
    
    -- Save the table to the backend
    --- convert to json
    local jsstr = url.escape(json.encode(table))
    
    --- build a http request
    local request = endpoint.."?function="..funct.."&jsonobj="..jsstr
    
    local response_body = { }
    --- send the request
    ok, code, headers = http.request{url = request, method = "POST", sink = ltn12.sink.table(response_body)}
    
    --- show that we got a valid response
   -- print(code) -- should be 201 for POST success 
    saveditem = response_body[1]; -- kinvey appdata responses return arrays (which are tables in Lua)
    --print(saveditem)
    
    --- convert from json to lua object
    objAsTable = json.decode(saveditem)
    return objAsTable
end



--CODE FOR BUILDING NEW SUPERMODEL:

--require 'util.ModelTracker'
--id=ModelTracker.createSupermodel("New SM","a testing sm");

--{["name"]="NewCross",["description"]="newcross",["submodelid"]=-99}
function ModelTracker.createCross(newcross)
    return ModelTracker.sendJsonObj(newcross,"1001")
end
--{["name"]="NewSM",["description"]="newsm",["supermodelid"]=-99}
function ModelTracker.createSubmodel(newsubmodel)
    return ModelTracker.sendJsonObj(newsubmodel,"1002")
end
--{["modelname"]="NewSM",["description"]="newsm"}
function ModelTracker.createSupermodel(name,description)
    local newsupermodel = {["modelname"]=name,["modeldescription"]=description}
    return ModelTracker.sendJsonObj(newsupermodel,"1003")
end

--{["category"]="changetalk",["group"]="train",["n"]=200,["crossid"]=-99,["value"]=3.45}
function ModelTracker.sendStatistic(statistic)
    return ModelTracker.sendJsonObj(statistic,"1000")
end
--{["reportname"]="",["parentid"]=crossid,["report"]="The text of the report"}
function ModelTracker.sendReport(report)
    return ModelTracker.sendJsonObj(report,"1010")
end