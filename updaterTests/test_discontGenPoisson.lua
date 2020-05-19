-- Gkyl ------------------------------------------------------------------------
--
--
--------------------------------------------------------------------------------

local Basis = require "Basis"
local DataStruct = require "DataStruct"
local Grid = require "Grid"
local Updater = require "Updater"

-- diffusion coefficients
local DxxFn = function(t, z)
   local x, y = z[1], z[2]
   return 1.0
end
local DyyFn = function(t, z)
   local x, y = z[1], z[2]
   return 1.0
end
local DxyFn = function(t, z)
   local x, y = z[1], z[2]
   return 0.0
end

local numQuad = 7
local solFn = function(t, z)
   local x, y = z[1], z[2]
   local a, b = 2, 5
   local c1, d0 = 0, 0
   local c0 = a/12 - 1/2
   local d1 = b/12 - 1/2
   local t1 = x^2/2 - a*x^4/12 + c0*x + c1
   local t2 = y^2/2 - b*y^4/12 + d0*y + d1
   return t1*t2
end
local srcFn = function(t, z)
   local x, y = z[1], z[2]
   local a, b = 2, 5
   local c1, d0 = 0, 0
   local c0 = a/12 - 1/2
   local d1 = b/12 - 1/2
   local t1 = (1-a*x^2)*(-b*y^4/12 + y^2/2 + d0*y + d1)
   local t2 = (1-b*y^2)*(-a*x^4/12 + x^2/2 + c0*x + c1)
   return -t1-t2
end

local grid = Grid.RectCart {
   lower = {0.0, 0.0},
   upper = {1.0, 1.0},
   cells = {16, 16},
   periodicDirs = {}
}
local basis = Basis.CartModalSerendipity {
   ndim = grid:ndim(),
   polyOrder = 1,
}
local function getField()
   return DataStruct.Field {
      onGrid = grid,
      numComponents = basis:numBasis(),
      ghost = {1, 1},
      metaData = {
         polyOrder = basis:polyOrder(),
         basisType = basis:id(),
      },
   }
end

local src = getField()
local initSource = Updater.ProjectOnBasis {
   onGrid = grid,
   basis = basis,
   numQuad = numQuad,
   evaluate = srcFn,
}
initSource:advance(0.0, {}, {src})

local solExact = getField()
local initSol = Updater.ProjectOnBasis {
   onGrid = grid,
   basis = basis,
   numQuad = numQuad,
   evaluate = solFn,
}
initSol:advance(0.0, {}, {solExact})

local Dxx = getField()
local initDxx = Updater.ProjectOnBasis {
   onGrid = grid,
   basis = basis,
   numQuad = numQuad,
   evaluate = DxxFn,
   projectOnGhosts = true,
}
initDxx:advance(0.0, {}, {Dxx})

local Dyy = getField()
local initDyy = Updater.ProjectOnBasis {
   onGrid = grid,
   basis = basis,
   numQuad = numQuad,
   evaluate = DyyFn,
   projectOnGhosts = true,
}
initDyy:advance(0.0, {}, {Dyy})

local Dxy = getField()
local initDxy = Updater.ProjectOnBasis {
   onGrid = grid,
   basis = basis,
   numQuad = numQuad,
   evaluate = DxyFn,
   projectOnGhosts = true,
}
initDxy:advance(0.0, {}, {Dxy})

local solSim = getField()
local discontPoisson = Updater.DiscontGenPoisson {
   onGrid = grid,
   basis = basis,
   Dxx = Dxx,
   Dyy = Dyy,
   Dxy = Dxy,
   bcLower = { {D=1, N=0, val=0.0}, {D=0, N=1, val=0.0} },
   bcUpper = { {D=1, N=0, val=0.0}, {D=1, N=0, val=0.0} },
   writeMatrix = false,
}
discontPoisson:advance(0.0, {src}, {solSim})

--src:write(string.format('src.bp'), 0.0, 0)
solExact:write(string.format('solExact.bp'), 0.0, 0)
solSim:write(string.format('solSim.bp'), 0.0, 0)
--Dxx:write(string.format('Dxx.bp'), 0.0, 0)
--Dyy:write(string.format('Dyy.bp'), 0.0, 0)
--Dxy:write(string.format('Dxy.bp'), 0.0, 0)
