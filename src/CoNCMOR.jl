"""
    CoNCMOR

Module for Coherent Node Cluster model reduction.
"""
module CoNCMOR

using SparseArrays
using LinearAlgebra
import Base.copyto!
using FinEtools
using Statistics: mean

struct LegendreBasis end
struct SineCosineBasis end
struct DivfreeBasis end

"""
    CoNC

Coherent Node Cluster type.

Collect information for one coherent node cluster.

"""
mutable struct CoNC
    nlist::FIntVec
    xyz::FFltMat
    _havenormalized::Bool # PRIVATE: do not access directly
    _normalizedxyz::FFltMat # PRIVATE: do not access directly
    _basis::FFltMat # PRIVATE: do not access directly
    function CoNC(node_list::FIntVec, xyz::FFltMat)
        return new(node_list, deepcopy(xyz), false, deepcopy(xyz), Array{FFlt}(undef, 0, 0))
    end
end

"""
    basisdim(self::CoNC)

What is the dimension of the base function set in the cluster?

In other words, how many linearly independent basis functions are there?
"""
basisdim(self::CoNC) =  size(self._basis, 2);

"""
    nnodes(self::CoNC)

How many nodes are there?
"""
nnodes(self::CoNC) =  size(self.xyz, 1);

"""
    CoNCData

Data structure for the Coherent Node Cluster model reduction.
"""
mutable struct CoNCData
    nodepartitioning::FIntVec
    clusters::Array{CoNC,1} # array of coherent node clusters
    function CoNCData()
        return new(Array{FInt,1}(undef, 0), Array{CoNC,1}(undef, 0))
    end
end

"""
    CoNCData(fens::FENodeSet, partitioning::AbstractVector{Int})

Constructor  of the Coherent Nodal Cluster model-reduction object.
"""
function CoNCData(fens::FENodeSet, partitioning::AbstractVector{Int})
    partitionnumbers = unique(partitioning)
    numclusters = length(partitionnumbers)
    self = CoNCData()
    self.nodepartitioning = deepcopy(partitioning)
    self.clusters = Array{CoNC,1}(undef, numclusters)
    nodelists = Array{Array{FInt,1},1}(undef, numclusters)
    for j = 1:numclusters
        nodelists[j] = FInt[]
        sizehint!(nodelists[j], count(fens))
    end 
    for k = 1:length(partitioning)
        push!(nodelists[partitioning[k]], k)
    end 
    for j = 1:numclusters
        p = partitionnumbers[j]
        self.clusters[j] =  CoNC(nodelists[p], fens.xyz[nodelists[p], :])
    end
    return self
end

"""
    nclusters(self::CoNCData)

Retrieve the number of node clusters.
"""
nclusters(self::CoNCData) = length(self.clusters)

"""
    nfuncspercluster(self::CoNCData)

Retrieve the number of basis functions per cluster.
"""
function nfuncspercluster(self::CoNCData)
    return basisdim(self.clusters[1])
end

"""
    nbasisfunctions(self::CoNCData)

Retrieve the total number of basis functions in the model reduction object.
"""
function nbasisfunctions(self::CoNCData)
    n = 0
    for ixxxx = 1:length(self.clusters)
        n = n + basisdim(self.clusters[ixxxx])
    end
    return n
end

"""
    assignfestoclusters(self::CoNCData, fes::AbstractFESet)

Assign finite elements to node clusters.

A finite element belongs to a cluster if at least one of its nodes belong to this
cluster.
"""
function assignfestoclusters(self::CoNCData, fes::AbstractFESet)
    feclusters = Vector{Vector{Bool}}(nclusters(self))
    for g = 1:nclusters(self)
        feclusters[g] = falses(count(fes))
        for e = 1:count(fes)
            n = unique(self.nodepartitioning[fes.conn[e, :]])
            if g in n
                feclusters[g][e] = true
            end
        end
    end
    return feclusters
end

"""
    transfmatrix(self::CoNCData, ::Type{LegendreBasis}, bnumbers::AbstractRange, fld::NodalField)

Compute the transformation matrix for the Legendre polynomial basis for the
one-dimensional basis functions given by the range `bnumbers`.
"""
function transfmatrix(self::CoNCData, ::Type{LegendreBasis}, bnumbers::AbstractRange, fld::NodalField)
    return _transfmatrix(self, bnumbers, fld, _legpol)
end

function transfmatrix(self::CoNCData, ::Type{LegendreBasis}, bnumber::Int, fld::NodalField)
    return transfmatrix(self, LegendreBasis, 1:bnumber, fld)
end 

"""
    transfmatrix(self::CoNCData, ::Type{SineCosineBasis}, degrees::AbstractRange, fld::NodalField)

Compute the transformation matrix for the mixed polynomial and cosine basis.
"""
function transfmatrix(self::CoNCData, ::Type{SineCosineBasis}, bnumbers::AbstractRange, fld::NodalField)
    return _transfmatrix(self, bnumbers, fld, _cosbas)
end

function transfmatrix(self::CoNCData, ::Type{SineCosineBasis}, bnumber::Int, fld::NodalField)
    return transfmatrix(self, SineCosineBasis, 1:bnumber, fld)
end 

"""
    transfmatrix(self::CoNCData, ::Type{DivfreeBasis}, degrees::AbstractRange, fld::NodalField)

Compute the transformation matrix for the divergence-free polynomial basis.
"""
function transfmatrix(self::CoNCData, ::Type{DivfreeBasis}, bnumbers::AbstractRange, fld::NodalField)
    return _transfmatrix(self, bnumbers, fld)
end

function transfmatrix(self::CoNCData, ::Type{DivfreeBasis}, bnumber::Int, fld::NodalField)
    return transfmatrix(self, DivfreeBasis, 1:bnumber, fld)
end 

# #############################################################################
# PRIVATE FUNCTIONS 

"""
    _transfmatrix(self::CoNCData, bnumbers::AbstractRange, fld::NodalField, f::F) where {F}

Compute the transformation matrix for the function `f` for the one-dimensional
basis functions given by the range `bnumbers`.
"""
function _transfmatrix(self::CoNCData, bnumbers::AbstractRange, fld::NodalField, f::F) where {F}
	ndof = ndofs(fld);
    ncol = 0; elem_mat_nrows = 0
    for  mm = 1:length( self.clusters )
        _generatebasis!(self.clusters[mm], bnumbers, f);
        ncol = ncol + ndof*basisdim(self.clusters[mm]);
        elem_mat_nrows = max(elem_mat_nrows, nnodes(self.clusters[mm]));
    end
    assembler = SysmatAssemblerSparse()
    startassembly!(assembler, elem_mat_nrows, 1, ncol, fld.nfreedofs, ncol)
    c=1;
    for  g = 1:length(self.clusters)
        for   dof = 1:ndof
            dofnums_r = fld.dofnums[self.clusters[g].nlist, dof]
            dofnums_c = fill(0, 1)
            for   b = 1:basisdim(self.clusters[g])
                fill!(dofnums_c, c)
                # assemble unsymmetric matrix
                assemble!(assembler, reshape(self.clusters[g]._basis[:,b], length(dofnums_r), 1), dofnums_r, dofnums_c);
                c=c+1;
            end
        end 
    end
    return makematrix!(assembler);
end

"""
    _makenormalized!(self::CoNC)

Compute the ANISOTROPIC normalized coordinates.
"""
function _makenormalized!(self::CoNC)
	if !self._havenormalized
		box = boundingbox(self.xyz);
	    sdim = size(self.xyz,2);
	    for   j=1:sdim
	        rang = box[(j-1)*2+1:(j-1)*2+2];
	        self._normalizedxyz[:,j] = broadcast(-, self.xyz[:,j], mean(rang))*2/diff(rang)[1];
	    end
	    self._havenormalized = true
	end
    return self
end

"""
    _makenormalized!(self::CoNC)

Compute the ISOTROPIC normalized coordinates.
"""
function _makenormalizedisotropic!(self::CoNC)
	if !self._havenormalized
		box = boundingbox(self.xyz);
	    sdim = size(self.xyz,2);
	    # We scale the physical coordinates by the largest extent, the same in all directions.
	    j=1
	    rang = box[(j-1)*2+1:(j-1)*2+2];
	    maxdiffrang = diff(rang)[1]
	    for   j in 2:sdim
	    	maxdiffrang = max(maxdiffrang, diff(rang)[1])
	    end
	    for   j in 1:sdim
	        self._normalizedxyz[:,j] = broadcast(-, self.xyz[:,j], mean(rang))*2/maxdiffrang;
	    end
	    self._havenormalized = true
	end
    return self
end

"""
    _legpol(n::Int, x::FFltVec)

Evaluate Legendre polynomial at given stations.
"""
function _legpol(n::Int, x::FFltVec)
    d = n - 1  # convert basis function number to polynomial degree
    @assert (n >= 1) && (n <= 12) "This degree of the polynomial ($d) is not implemented"
    p = zeros(size(x))
    if d == 0
        copyto!(p,  ones(size(x)));
    elseif d == 1
        copyto!(p, x);
    elseif d == 2
        copyto!(p, @. 1/2*(3*x^2-1));
    elseif d == 3
        copyto!(p, @. 1/2*(5*x^3-3*x));
    elseif d == 4
        copyto!(p, @. 1/8*(35*x^4-30*x^2+3));
    elseif d == 5
        copyto!(p, @. 1/8*(63*x^5-70*x^3+15*x));
    elseif d == 6
        copyto!(p, @. 1/16*(231*x^6-315*x^4+105*x^2-5));
    elseif d == 7
        copyto!(p, @. 1/16*(429*x^7-693*x^5+315*x^3-35*x));
    elseif d == 8
        copyto!(p, @. (1.0/128) * ((6435*x^8) - (12012*x^6) + (6930*x^4) - (1260*x^2) + 35));
    elseif d == 9
        copyto!(p, @. (1.0/128) * ((12155*x^9) - (25740*x^7) + (18018*x^5) - (4620*x^3) + (315*x)));
    elseif d == 10
        copyto!(p, @. (1.0/256) * ((46189*x^10) - (109395*x^8) + (90090*x^6) - (30030*x^4) + (3465*x^2) - 63));
    elseif d == 11
        copyto!(p, @. (1.0/256) * ((88179*x^11) - (230945*x^9) + (218790*x^7) - (90090*x^5) + (15015*x^3) - (693*x)));
    elseif d == 12
        copyto!(p, @. (1.0/1024) * ((676039*x^12) - (1939938*x^10) + (2078505*x^8) - (1021020*x^6) + (225225*x^4) - (18018*x^2) + 231));
    end
    return p
end


function _cosbas(n::Int, x::FFltVec)
    @assert (n >= 1) && (n <= 12) "This basis function ($n) is not implemented"
    p = zeros(size(x))
    if n == 1
        copyto!(p,  ones(size(x)));
    elseif n == 2
        copyto!(p, x);
    elseif n == 3
        copyto!(p, @. cos(1.0/2.0*pi*x) - 1);
    elseif n == 4
        copyto!(p, @. sin(1.0/2.0*pi*x));
    elseif n == 5
        copyto!(p, @. cos(2.0/2.0*pi*x) - 1);
    elseif n == 6
        copyto!(p, @. sin(2.0/2.0*pi*x));
    elseif n == 7
        copyto!(p, @. cos(3.0/2.0*pi*x) - 1);
    elseif n == 8
        copyto!(p, @. sin(3.0/2.0*pi*x));
    elseif n == 9
        copyto!(p, @. cos(4.0/2.0*pi*x) - 1);
    elseif n == 10
        copyto!(p, @. sin(4.0/2.0*pi*x));
    elseif n == 11
        copyto!(p, @. cos(5.0/2.0*pi*x) - 1);
    elseif n == 12
        copyto!(p, @. sin(5.0/2.0*pi*x));
    end
    return p
end

function _monopol!(p::FFltMat, n::Int, x::T) where {T}
    d = n - 1  # convert basis function number to polynomial degree
    @assert (n >= 1) && (n <= 2) "This degree of the polynomial ($d) is not implemented"
    @assert length(p) == length(x)
    copyto!(p, x.^d);
    return p
end

"""
    _generatebasis!(self::CoNC, bnumbers::AbstractRange, f::F) where {F}

Generate basis described by the function `f` for the one-dimensional basis
functions given by the range `bnumbers`.
"""

function _generatebasis!(self::CoNC, bnumbers::AbstractRange, f::F) where {F}
    self = _makenormalized!(self);
    sdim = size(self.xyz,2)
    @assert (sdim >= 1) && (sdim <= 3)
    maxbnumber = maximum(bnumbers)
    sumlo = minimum(bnumbers) + sdim - 1
    sumhi = maximum(bnumbers) + sdim - 1
    if sdim == 1
		# First calculate the total of basis functions
	    b=1;
	    for  ix = 1:maxbnumber
	        if sumlo <= ix <= sumhi
	            b=b+1;
	        end
	    end
	    # Now allocate the basis function buffer, and calculate the one-dimensional basis functions
	    self._basis = zeros(size(self.xyz,1),b-1);
	    fx = zeros(size(self.xyz,1), maxbnumber);
	    for   b = 1:maxbnumber
	        fx[:, b] = f(b, self._normalizedxyz[:,1]);
	    end
	    # Sweep through the basis functions, calculate the b-th column
	    b=1;
	    for ix = 1:maxbnumber
	        if sumlo <= ix <= sumhi # Only for the upper-left pyramid
	            self._basis[:,b] = view(fx, :, ix);
	            b=b+1;
	        end
	    end
    elseif sdim == 2
		# First calculate the total of basis functions
	    b=1;
	    for   iy = 1:maxbnumber, ix = 1:maxbnumber
	        if sumlo <= ix+iy <= sumhi
	            b=b+1;
	        end
	    end
	    # Now allocate the basis function buffer, and calculate the one-dimensional basis functions
	    self._basis = zeros(size(self.xyz,1),b-1);
	    fy = zeros(size(self.xyz,1),maxbnumber);
	    fx = zeros(size(self.xyz,1),maxbnumber);
	    for   b = 1:maxbnumber
	    	fy[:, b] = f(b, self._normalizedxyz[:,2])
	        fx[:, b] = f(b, self._normalizedxyz[:,1]);
	    end
	    # Sweep through the basis functions, calculate the b-th column
	    b=1;
	    for  iy = 1:maxbnumber, ix = 1:maxbnumber
	        if sumlo <= ix+iy <= sumhi # Only for the upper-left pyramid
	            self._basis[:,b] = view(fx, :, ix) .* view(fy, :, iy);
	            b=b+1;
	        end
	    end
    elseif sdim == 3
    	# First calculate the total of basis functions
        b=1;
        for   iz = 1:maxbnumber, iy = 1:maxbnumber, ix = 1:maxbnumber
            if sumlo <= ix+iy+iz <= sumhi
                b=b+1;
            end
        end
        # Now allocate the basis function buffer, and calculate the one-dimensional basis functions
        self._basis = zeros(size(self.xyz,1),b-1);
        fz = zeros(size(self.xyz,1),maxbnumber);
        fy = zeros(size(self.xyz,1),maxbnumber);
        fx = zeros(size(self.xyz,1),maxbnumber);
        for   b = 1:maxbnumber
        	fz[:, b] = f(b, self._normalizedxyz[:,3])
        	fy[:, b] = f(b, self._normalizedxyz[:,2])
            fx[:, b] = f(b, self._normalizedxyz[:,1]);
        end
        # Sweep through the basis functions, calculate the b-th column
        b=1;
        for   iz = 1:maxbnumber, iy = 1:maxbnumber, ix = 1:maxbnumber
            if sumlo <= ix+iy+iz <= sumhi # Only for the upper-left pyramid
                self._basis[:,b] = view(fx, :, ix) .* view(fy, :, iy) .* view(fz, :, iz);
                b=b+1;
            end
        end
    end
    return self
end

"""
    _transfmatrix(self::CoNCData, bnumbers::AbstractRange, fld::NodalField, f::F) where {F<:DivfreeBasis}

Compute the transformation matrix for the divergence-free basis.
"""
function _transfmatrix(self::CoNCData, bnumbers::AbstractRange, fld::NodalField) 
	@assert minimum(bnumbers) == 1
	@assert (maximum(bnumbers) > 1) && (maximum(bnumbers) <= 3)
	ndof = ndofs(fld);
    return _transfmatrix(self, Val(ndof), Val(maximum(bnumbers)), fld)
end

function _transfmatrix(self::CoNCData, ::Val{3}, ::Val{2}, fld::NodalField)
	ndof = ndofs(fld);
	@assert ndof == 3
	maxbn = 2 # Linear basis: Val{2}
	# The constraint that the basis function set should give divergence-free
	# displacement  field generates a certain number of constraints on the
	# generalized degrees of freedom.
	# For a linear basis (maximum(bnumbers) == 2) the number of basis
	# functions is 3 * 4 - 1 = 11. 
	# For a quadratic basis (maximum(bnumbers) == 3) the number of basis
	# functions is 3 * 10 - 4 = 26. 
	totalnbf = [0, 11, 26]
	ncol = 0; elem_mat_nrows = 0
    for  mm in 1:length(self.clusters)
    	_makenormalizedisotropic!(self.clusters[mm]);
        ncol = ncol + totalnbf[maxbn];
        elem_mat_nrows = max(elem_mat_nrows, nnodes(self.clusters[mm]));
    end
    assembler = SysmatAssemblerSparse()
    startassembly!(assembler, elem_mat_nrows, 1, ncol+2*length(self.clusters), fld.nfreedofs, ncol)
    c=1;
    for  g in 1:length(self.clusters)
    	dofnums_rx = fld.dofnums[self.clusters[g].nlist, 1]
    	dofnums_ry = fld.dofnums[self.clusters[g].nlist, 2]
    	dofnums_rz = fld.dofnums[self.clusters[g].nlist, 3]
    	dofnums_c = fill(0, 1)
    	x = reshape(self.clusters[g]._normalizedxyz[:, 1], length(dofnums_rx), 1)
    	y = reshape(self.clusters[g]._normalizedxyz[:, 2], length(dofnums_rx), 1)
    	z = reshape(self.clusters[g]._normalizedxyz[:, 3], length(dofnums_rx), 1)
    	# 1: a_0
    	fill!(dofnums_c, c)
    	assemble!(assembler, x.^0, dofnums_rx, dofnums_c);
    	c=c+1;
    	# 2: b_0
    	fill!(dofnums_c, c)
    	assemble!(assembler, y.^0, dofnums_ry, dofnums_c);
    	c=c+1;
    	# 3: c_0
    	fill!(dofnums_c, c)
    	assemble!(assembler, z.^0, dofnums_rz, dofnums_c);
    	c=c+1;
    	# 4: a_1
    	fill!(dofnums_c, c)
    	assemble!(assembler, x.^1, dofnums_rx, dofnums_c);
    	assemble!(assembler, -z.^1, dofnums_rz, dofnums_c);
    	c=c+1;
    	# 5: b_1
    	fill!(dofnums_c, c)
    	assemble!(assembler, x.^1, dofnums_ry, dofnums_c);
    	c=c+1;
    	# 6: c_1
    	fill!(dofnums_c, c)
    	assemble!(assembler, x.^1, dofnums_rz, dofnums_c);
    	c=c+1;
    	# 7: a_2
    	fill!(dofnums_c, c)
    	assemble!(assembler, y.^1, dofnums_rx, dofnums_c);
    	c=c+1;
    	# 8: b_2
    	fill!(dofnums_c, c)
    	assemble!(assembler, y.^1, dofnums_ry, dofnums_c);
    	assemble!(assembler, -z.^1, dofnums_rz, dofnums_c);
    	c=c+1;
    	# 9: c_2
    	fill!(dofnums_c, c)
    	assemble!(assembler, y.^1, dofnums_rz, dofnums_c);
    	c=c+1;
    	# 10: a_3
    	fill!(dofnums_c, c)
    	assemble!(assembler, z.^1, dofnums_rx, dofnums_c);
    	c=c+1;
    	# 11: b_3
    	fill!(dofnums_c, c)
    	assemble!(assembler, z.^1, dofnums_ry, dofnums_c);
    	c=c+1;
    end
    return makematrix!(assembler);
end

function _transfmatrix(self::CoNCData, ::Val{3}, ::Val{3}, fld::NodalField)
	ndof = ndofs(fld);
	@assert ndof == 3
	maxbn = 3 # Quadratic basis: Val{3}
	# The constraint that the basis function set should give divergence-free
	# displacement  field generates a certain number of constraints on the
	# generalized degrees of freedom.
	# For a linear basis (maximum(bnumbers) == 2) the number of basis
	# functions is 3 * 4 - 1 = 11. 
	# For a quadratic basis (maximum(bnumbers) == 3) the number of basis
	# functions is 3 * 10 - 4 = 26. 
	totalnbf = [0, 11, 26]
	ncol = 0; elem_mat_nrows = 0
    for  mm in 1:length(self.clusters)
    	_makenormalizedisotropic!(self.clusters[mm]);
        ncol = ncol + totalnbf[maxbn];
        elem_mat_nrows = max(elem_mat_nrows, nnodes(self.clusters[mm]));
    end
    assembler = SysmatAssemblerSparse()
    startassembly!(assembler, elem_mat_nrows, 1, ncol+8*length(self.clusters), fld.nfreedofs, ncol)
    c=1;
    for  g in 1:length(self.clusters)
    	dofnums_rx = fld.dofnums[self.clusters[g].nlist, 1]
    	dofnums_ry = fld.dofnums[self.clusters[g].nlist, 2]
    	dofnums_rz = fld.dofnums[self.clusters[g].nlist, 3]
    	dofnums_c = fill(0, 1)
    	x = reshape(self.clusters[g]._normalizedxyz[:, 1], length(dofnums_rx), 1)
    	y = reshape(self.clusters[g]._normalizedxyz[:, 2], length(dofnums_rx), 1)
    	z = reshape(self.clusters[g]._normalizedxyz[:, 3], length(dofnums_rx), 1)
    	# 1: a_0
    	fill!(dofnums_c, c)
        assemble!(assembler, x.^0, dofnums_rx, dofnums_c);
        c=c+1;
        # 2: b_0
        fill!(dofnums_c, c)
        assemble!(assembler, y.^0, dofnums_ry, dofnums_c);
        c=c+1;
        # 3: c_0
        fill!(dofnums_c, c)
        assemble!(assembler, z.^0, dofnums_rz, dofnums_c);
        c=c+1;
        # 4: a_1
        fill!(dofnums_c, c)
        assemble!(assembler, x.^1, dofnums_rx, dofnums_c);
        assemble!(assembler, -z.^1, dofnums_rz, dofnums_c);
        c=c+1;
        # 5: b_1
        fill!(dofnums_c, c)
        assemble!(assembler, x.^1, dofnums_ry, dofnums_c);
        c=c+1;
        # 6: c_1
        fill!(dofnums_c, c)
        assemble!(assembler, x.^1, dofnums_rz, dofnums_c);
        c=c+1;
        # 7: a_2
        fill!(dofnums_c, c)
        assemble!(assembler, y.^1, dofnums_rx, dofnums_c);
        c=c+1;
        # 8: b_2
        fill!(dofnums_c, c)
        assemble!(assembler, y.^1, dofnums_ry, dofnums_c);
        assemble!(assembler, -z.^1, dofnums_rz, dofnums_c);
        c=c+1;
        # 9: c_2
        fill!(dofnums_c, c)
        assemble!(assembler, y.^1, dofnums_rz, dofnums_c);
        c=c+1;
        # 10: a_3
        fill!(dofnums_c, c)
        assemble!(assembler, z.^1, dofnums_rx, dofnums_c);
        c=c+1;
        # 11: b_3
        fill!(dofnums_c, c)
        assemble!(assembler, z.^1, dofnums_ry, dofnums_c);
        c=c+1;
        # 12: a_4
        fill!(dofnums_c, c)
        assemble!(assembler, x.^2, dofnums_rx, dofnums_c);
        assemble!(assembler, -2 .* x.^1 .* z.^1, dofnums_rz, dofnums_c);
        c=c+1;
        # 13: a_5
        fill!(dofnums_c, c)
        assemble!(assembler, y.^2, dofnums_rx, dofnums_c);
        c=c+1;
        # 14: a_6
        fill!(dofnums_c, c)
        assemble!(assembler, z.^2, dofnums_rx, dofnums_c);
        c=c+1;
        # 15: a_7
        fill!(dofnums_c, c)
        assemble!(assembler, x.^1 .* y.^1, dofnums_rx, dofnums_c);
        assemble!(assembler, -y.^1 .* z.^1, dofnums_rz, dofnums_c);
        c=c+1;
        # 16: a_8
        fill!(dofnums_c, c)
        assemble!(assembler, x.^1 .* z.^1, dofnums_rx, dofnums_c);
        assemble!(assembler, -0.5 .* z.^2, dofnums_rz, dofnums_c);
        c=c+1;
        # 17: a_9
        fill!(dofnums_c, c)
        assemble!(assembler, y.^1 .* z.^1, dofnums_rx, dofnums_c);
        c=c+1;
        # 18: b_4
        fill!(dofnums_c, c)
        assemble!(assembler, x.^2, dofnums_ry, dofnums_c);
        c=c+1;
        # 19: b_5
        fill!(dofnums_c, c)
        assemble!(assembler, y.^2, dofnums_ry, dofnums_c);
        assemble!(assembler, -2 .* y.^1 .* z.^1, dofnums_rz, dofnums_c);
        c=c+1;
        # 20: b_6
        fill!(dofnums_c, c)
        assemble!(assembler, z.^2, dofnums_ry, dofnums_c);
        c=c+1;
        # 21: b_7
        fill!(dofnums_c, c)
        assemble!(assembler, x.^1 .* y.^1, dofnums_ry, dofnums_c);
        assemble!(assembler, -x.^1 .* z.^1, dofnums_rz, dofnums_c);
        c=c+1;
        # 22: b_8
        fill!(dofnums_c, c)
        assemble!(assembler, x.^1 .* z.^1, dofnums_ry, dofnums_c);
        c=c+1;
        # 23: b_9
        fill!(dofnums_c, c)
        assemble!(assembler, y.^1 .* z.^1, dofnums_ry, dofnums_c);
        assemble!(assembler, -0.5 .* z.^2, dofnums_rz, dofnums_c);
        c=c+1;
        # 24: c_4
        fill!(dofnums_c, c)
        assemble!(assembler, x.^2, dofnums_rz, dofnums_c);
        c=c+1;
        # 25: c_5
        fill!(dofnums_c, c)
        assemble!(assembler, y.^2, dofnums_rz, dofnums_c);
        c=c+1;
        # 26: c_7
        fill!(dofnums_c, c)
        assemble!(assembler, x.^1 .* y.^1, dofnums_rz, dofnums_c);
        c=c+1;
    end
    return makematrix!(assembler);
end

end # module


# function _transfmatrix(self::CoNCData, ::Val{3}, ::Val{2}, fld::NodalField)
# 	ndof = ndofs(fld);
# 	@assert ndof == 3
# 	maxbn = 2 # Linear basis: Val{2}
# 	# The constraint that the basis function set should give divergence-free
# 	# displacement  field generates a certain number of constraints on the
# 	# generalized degrees of freedom.
# 	# For a linear basis (maximum(bnumbers) == 2) the number of basis
# 	# functions is 3 * 4 - 1 = 11. 
# 	# For a quadratic basis (maximum(bnumbers) == 3) the number of basis
# 	# functions is 3 * 10 - 4 = 26. 
# 	totalnbf = [0, 11, 26]
# 	ncol = 0; elem_mat_nrows = 0
#     for  mm in 1:length(self.clusters)
#     	_makenormalizedisotropic!(self.clusters[mm]);
#         ncol = ncol + totalnbf[maxbn];
#         elem_mat_nrows = max(elem_mat_nrows, nnodes(self.clusters[mm]));
#     end
#     assembler = SysmatAssemblerSparse()
#     startassembly!(assembler, elem_mat_nrows, 1, ncol+2*length(self.clusters), fld.nfreedofs, ncol)
#     c=1;
#     for  g in 1:length(self.clusters)
#     	dofnums_rx = fld.dofnums[self.clusters[g].nlist, 1]
#     	dofnums_ry = fld.dofnums[self.clusters[g].nlist, 2]
#     	dofnums_rz = fld.dofnums[self.clusters[g].nlist, 3]
#     	dofnums_c = fill(0, 1)
#     	x = view(self.clusters[g]._normalizedxyz, :, 1)
#     	y = view(self.clusters[g]._normalizedxyz, :, 2)
#     	z = view(self.clusters[g]._normalizedxyz, :, 3)
#     	p = fill(0.0, length(dofnums_rx), 1)
#     	# 1
#         fill!(dofnums_c, c)
#         _monopol!(p, 1, x)
#         assemble!(assembler, p, dofnums_rx, dofnums_c);
#         c=c+1;
#         # 2
#         fill!(dofnums_c, c)
#         _monopol!(p, 1, y)
#         assemble!(assembler, p, dofnums_ry, dofnums_c);
#         c=c+1;
#         # 3
#         fill!(dofnums_c, c)
#         _monopol!(p, 1, z)
#         assemble!(assembler, p, dofnums_rz, dofnums_c);
#         c=c+1;
#         # 4
#         fill!(dofnums_c, c)
#         _monopol!(p, 2, x)
#         assemble!(assembler, p, dofnums_rx, dofnums_c);
#         _monopol!(p, 2, z)
#         assemble!(assembler, -1.0.*p, dofnums_rz, dofnums_c);
#         c=c+1;
#         # 5
#         fill!(dofnums_c, c)
#         _monopol!(p, 2, x)
#         assemble!(assembler, p, dofnums_ry, dofnums_c);
#         c=c+1;
#         # 6
#         fill!(dofnums_c, c)
#         _monopol!(p, 2, x)
#         assemble!(assembler, p, dofnums_rz, dofnums_c);
#         c=c+1;
#         # 7
#         fill!(dofnums_c, c)
#         _monopol!(p, 2, y)
#         assemble!(assembler, p, dofnums_rx, dofnums_c);
#         c=c+1;
#         # 8
#         fill!(dofnums_c, c)
#         _monopol!(p, 2, y)
#         assemble!(assembler, p, dofnums_ry, dofnums_c);
#         _monopol!(p, 2, z)
#         assemble!(assembler, -1.0.*p, dofnums_rz, dofnums_c);
#         c=c+1;
#         # 9
#         fill!(dofnums_c, c)
#         _monopol!(p, 2, y)
#         assemble!(assembler, p, dofnums_rz, dofnums_c);
#         c=c+1;
#         # 10
#         fill!(dofnums_c, c)
#         _monopol!(p, 2, z)
#         assemble!(assembler, p, dofnums_rx, dofnums_c);
#         c=c+1;
#         # 11
#         fill!(dofnums_c, c)
#         _monopol!(p, 2, z)
#         assemble!(assembler, p, dofnums_ry, dofnums_c);
#         c=c+1;
#     end
#     return makematrix!(assembler);
# end
