"""
    CoNCMOR

Module for Coherent Node Cluster model reduction.
"""
module CoNCMOR

using SparseArrays
using StaticArrays
using LinearAlgebra
import Base.copyto!
using Statistics: mean

include("conc.jl")
include("partitioning.jl")

end # module
