"""
    CoNCMOR2

Module for Coherent Node Cluster model reduction.

Version 06/2020. This will work with https://github.com/PetrKryslUCSD/MixedFEbyReduction.jl 
"""
module CoNCMOR2

using SparseArrays
using StaticArrays
using LinearAlgebra
import Base.copyto!
using Statistics: mean

include("conc.jl")
include("partitioning.jl")

end # module
