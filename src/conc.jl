struct LegendreBasis end
struct SineCosineBasis end
# struct DivfreeBasis end

"""
    CoNC

Coherent Node Cluster type.

Collect information for one coherent node cluster.

"""
mutable struct CoNC{N}
    nlist::Vector{Int64}
    coordinates::Vector{SVector{N, Float64}}

    # PRIVATE: do not access directly. Do we have normalized coordinates?
    _havenormalized::Bool 
    # PRIVATE: do not access directly. Normalized coordinates.
    _normalizedcoordinates::Vector{SVector{N, Float64}} 
    # PRIVATE: do not access directly. Each column is for one basis function.
    # The number of rows corresponds to the number of nodes in the cluster.
    _basis::Matrix{Float64} 
    # PRIVATE: do not access directly. What is the number of each basis function
    # in the global transformation matrix?
    _basis_number::Vector{Int64}

    function CoNC(node_list, xyz)
        N = length(xyz[1])
        ixyz = [SVector{N, Float64}(xyz[idx]) for idx in 1:length(xyz)]
        return new{N}(node_list, ixyz, false, deepcopy(ixyz), Array{Float64}(undef, 0, 0))
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
nnodes(self::CoNC) =  length(self.coordinates)

"""
    CoNCData

Data structure for the Coherent Node Cluster model reduction.
"""
mutable struct CoNCData
    nodepartitioning::Vector{Int64}
    clusters::Vector{CoNC} # array of coherent node clusters
    function CoNCData()
        return new(Vector{Int64}(undef, 0), Vector{CoNC}(undef, 0))
    end
end

"""
    CoNCData(fens::FENodeSet, partitioning::AbstractVector{Int})

Constructor  of the Coherent Nodal Cluster model-reduction object.
"""
function CoNCData(xyz, partitioning)
    partitionnumbers = unique(partitioning)
    numclusters = length(partitionnumbers)
    self = CoNCData()
    self.nodepartitioning = deepcopy(partitioning)
    self.clusters = Array{CoNC,1}(undef, numclusters)
    nodelists = Array{Array{Int64,1},1}(undef, numclusters)
    for j in 1:numclusters
        nodelists[j] = Int64[]
        sizehint!(nodelists[j], length(xyz))
    end 
    for k in 1:length(partitioning)
        push!(nodelists[partitioning[k]], k)
    end 
    for j in 1:numclusters
        p = partitionnumbers[j]
        self.clusters[j] =  CoNC(nodelists[p], [xyz[k] for k in nodelists[p]])
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
    basisfunctionnumbers(self::CoNCData)

Access the numbers
"""
basisfunctionnumbers(self::CoNCData, cluster) = self.clusters[cluster]._basis_number

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
    transfmatrix(self::CoNCData, ::Type{LegendreBasis}, bnumbers::AbstractRange)

Compute the transformation matrix for the Legendre polynomial basis for the
one-dimensional basis functions given by the range `bnumbers`.
"""
function transfmatrix(self::CoNCData, ::Type{LegendreBasis}, bnumbers::AbstractRange)
    return _transfmatrix(self, bnumbers, _legpol)
end

function transfmatrix(self::CoNCData, ::Type{LegendreBasis}, bnumber)
    return transfmatrix(self, LegendreBasis, 1:bnumber)
end 

"""
    transfmatrix(self::CoNCData, ::Type{SineCosineBasis}, degrees::AbstractRange)

Compute the transformation matrix for the mixed polynomial and cosine basis.
"""
function transfmatrix(self::CoNCData, ::Type{SineCosineBasis}, bnumbers::AbstractRange)
    return _transfmatrix(self, bnumbers, _cosbas)
end

function transfmatrix(self::CoNCData, ::Type{SineCosineBasis}, bnumber)
    return transfmatrix(self, SineCosineBasis, 1:bnumber)
end 

# """
#     transfmatrix(self::CoNCData, ::Type{DivfreeBasis}, degrees::AbstractRange)

# Compute the transformation matrix for the divergence-free polynomial basis.
# """
# function transfmatrix(self::CoNCData, ::Type{DivfreeBasis}, bnumbers::AbstractRange)
#     return _transfmatrix(self, bnumbers, fld)
# end

# function transfmatrix(self::CoNCData, ::Type{DivfreeBasis}, bnumber)
#     return transfmatrix(self, DivfreeBasis, 1:bnumber, fld)
# end 

# #############################################################################
# PRIVATE FUNCTIONS 

"""
    _transfmatrix(self::CoNCData, bnumbers::AbstractRange, f::F) where {F}

Compute the transformation matrix for the function `f` for the one-dimensional
basis functions given by the range `bnumbers`.
"""
function _transfmatrix(self::CoNCData, bnumbers::AbstractRange, f::F) where {F}
	for  mm in 1:length( self.clusters )
        _generatebasis!(self.clusters[mm], bnumbers, f);
    end
    I = Int64[]
    J = Int64[]
    V = Float64[]
    c = 1;
    for  g in 1:length(self.clusters)
        for  b in 1:basisdim(self.clusters[g])
            for r in 1:length(self.clusters[g].nlist)
                push!(I, self.clusters[g].nlist[r])
                push!(J, c)
                push!(V, self.clusters[g]._basis[r,b])
            end
            self.clusters[g]._basis_number[b] = c
            c=c+1;
        end
    end
    return I, J, V;
end


function _boundingbox(x) 
    function updatebox!(box, x) 
        sdim = length(x)
        for i in 1:sdim
            box[2*i-1] = min(box[2*i-1],x[i]);
            box[2*i]   = max(box[2*i],x[i]);
        end
        return box
    end
    function initbox(x)
        sdim = length(x)
        box = fill(zero(eltype(x)), 2*sdim)
        for i in 1:sdim
            box[2*i-1] = box[2*i] = x[i];
        end
        return box
    end
    box = initbox(x[1])
    for i in 2:length(x)
        updatebox!(box, x[i])
    end
    return box
end

"""
    _makenormalized!(self::CoNC)

Compute the ANISOTROPIC normalized coordinates.
"""
function _makenormalized!(self::CoNC)
	if !self._havenormalized
		box = _boundingbox(self.coordinates);
	    sdim = length(self.coordinates[1])
	    for   j in 1:sdim
	        rang = box[(j-1)*2+1:(j-1)*2+2];
            mr = mean(rang)
            dr = diff(rang)[1]
            for i in 1:length(self.coordinates)
                nc = (self.coordinates[i][j] - mr) * 2/dr
                c = MVector(self._normalizedcoordinates[i])
                c[j] = nc
                self._normalizedcoordinates[i] = c
            end
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
		box = _boundingbox(self.xyz);
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
    _legpol(n, x)

Evaluate Legendre polynomial at given stations.
"""
function _legpol(n, x)
    d = n - 1  # convert basis function number to polynomial degree
    @assert (n >= 1) && (n <= 12) "This degree of the polynomial ($d) is not implemented"
    if d == 0
        return 1.0
    elseif d == 1
        return x
    elseif d == 2
        return 1/2*(3*x^2-1)
    elseif d == 3
        return 1/2*(5*x^3-3*x)
    elseif d == 4
        return 1/8*(35*x^4-30*x^2+3)
    elseif d == 5
        return 1/8*(63*x^5-70*x^3+15*x)
    elseif d == 6
        return 1/16*(231*x^6-315*x^4+105*x^2-5)
    elseif d == 7
        return 1/16*(429*x^7-693*x^5+315*x^3-35*x)
    elseif d == 8
        return (1.0/128) * ((6435*x^8) - (12012*x^6) + (6930*x^4) - (1260*x^2) + 35)
    elseif d == 9
        return (1.0/128) * ((12155*x^9) - (25740*x^7) + (18018*x^5) - (4620*x^3) + (315*x))
    elseif d == 10
        return (1.0/256) * ((46189*x^10) - (109395*x^8) + (90090*x^6) - (30030*x^4) + (3465*x^2) - 63)
    elseif d == 11
        return (1.0/256) * ((88179*x^11) - (230945*x^9) + (218790*x^7) - (90090*x^5) + (15015*x^3) - (693*x))
    elseif d == 12
        return (1.0/1024) * ((676039*x^12) - (1939938*x^10) + (2078505*x^8) - (1021020*x^6) + (225225*x^4) - (18018*x^2) + 231)
    end
    return 0.0
end


function _cosbas(n, x)
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

function _monopol!(p, n, x::T) where {T}
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
    sdim = length(self.coordinates[1])
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
        self._basis_number = zeros(b-1);
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
	    for iy = 1:maxbnumber, ix = 1:maxbnumber
	        if sumlo <= ix+iy <= sumhi
	            b=b+1;
	        end
	    end
	    # Now allocate the basis function buffer, and calculate the one-dimensional basis functions
	    self._basis = zeros(length(self.coordinates),b-1);
        self._basis_number = zeros(b-1);
	    fy = zeros(length(self.coordinates),maxbnumber);
	    fx = zeros(length(self.coordinates),maxbnumber);
	    for b in 1:maxbnumber
            for i in 1:length(self.coordinates)
                fy[i, b] = f(b, self._normalizedcoordinates[i][2])
                fx[i, b] = f(b, self._normalizedcoordinates[i][1]);
            end
        end
	    # Sweep through the basis functions, calculate the b-th column
	    b=1;
	    for  iy = 1:maxbnumber, ix = 1:maxbnumber
	        if sumlo <= ix+iy <= sumhi # Only for the upper-left pyramid
                for i in 1:size(self._basis, 1)
                    self._basis[i,b] = fx[i, ix] * fy[i, iy];
                end
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
        self._basis = zeros(length(self.coordinates),b-1);
        self._basis_number = zeros(b-1);
        fz = zeros(length(self.coordinates),maxbnumber);
        fy = zeros(length(self.coordinates),maxbnumber);
        fx = zeros(length(self.coordinates),maxbnumber);
        for  b in 1:maxbnumber
            for i in 1:length(self.coordinates)
                fz[i, b] = f(b, self._normalizedcoordinates[i][3])
                fy[i, b] = f(b, self._normalizedcoordinates[i][2])
                fx[i, b] = f(b, self._normalizedcoordinates[i][1]);
            end
        end
        # Sweep through the basis functions, calculate the b-th column
        b=1;
        for   iz = 1:maxbnumber, iy = 1:maxbnumber, ix = 1:maxbnumber
            if sumlo <= ix+iy+iz <= sumhi # Only for the upper-left pyramid
                for i in 1:size(self._basis, 1)
                    self._basis[i,b] = fx[i, ix] * fy[i, iy] * fz[i, iz];
                end
                b=b+1;
            end
        end
    end
    return self
end

# """
#     _transfmatrix(self::CoNCData, bnumbers::AbstractRange, f::F) where {F<:DivfreeBasis}

# Compute the transformation matrix for the divergence-free basis.
# """
# function _transfmatrix(self::CoNCData, bnumbers::AbstractRange) 
# 	@assert minimum(bnumbers) == 1
# 	@assert (maximum(bnumbers) > 1) && (maximum(bnumbers) <= 3)
# 	ndof = ndofs(fld);
#     return _transfmatrix(self, Val(ndof), Val(maximum(bnumbers)), fld)
# end

# function _transfmatrix(self::CoNCData, ::Val{3}, ::Val{2})
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
#     	x = reshape(self.clusters[g]._normalizedxyz[:, 1], length(dofnums_rx), 1)
#     	y = reshape(self.clusters[g]._normalizedxyz[:, 2], length(dofnums_rx), 1)
#     	z = reshape(self.clusters[g]._normalizedxyz[:, 3], length(dofnums_rx), 1)
#     	# 1: a_0
#     	fill!(dofnums_c, c)
#     	assemble!(assembler, x.^0, dofnums_rx, dofnums_c);
#     	c=c+1;
#     	# 2: b_0
#     	fill!(dofnums_c, c)
#     	assemble!(assembler, y.^0, dofnums_ry, dofnums_c);
#     	c=c+1;
#     	# 3: c_0
#     	fill!(dofnums_c, c)
#     	assemble!(assembler, z.^0, dofnums_rz, dofnums_c);
#     	c=c+1;
#     	# 4: a_1
#     	fill!(dofnums_c, c)
#     	assemble!(assembler, x.^1, dofnums_rx, dofnums_c);
#     	assemble!(assembler, -z.^1, dofnums_rz, dofnums_c);
#     	c=c+1;
#     	# 5: b_1
#     	fill!(dofnums_c, c)
#     	assemble!(assembler, x.^1, dofnums_ry, dofnums_c);
#     	c=c+1;
#     	# 6: c_1
#     	fill!(dofnums_c, c)
#     	assemble!(assembler, x.^1, dofnums_rz, dofnums_c);
#     	c=c+1;
#     	# 7: a_2
#     	fill!(dofnums_c, c)
#     	assemble!(assembler, y.^1, dofnums_rx, dofnums_c);
#     	c=c+1;
#     	# 8: b_2
#     	fill!(dofnums_c, c)
#     	assemble!(assembler, y.^1, dofnums_ry, dofnums_c);
#     	assemble!(assembler, -z.^1, dofnums_rz, dofnums_c);
#     	c=c+1;
#     	# 9: c_2
#     	fill!(dofnums_c, c)
#     	assemble!(assembler, y.^1, dofnums_rz, dofnums_c);
#     	c=c+1;
#     	# 10: a_3
#     	fill!(dofnums_c, c)
#     	assemble!(assembler, z.^1, dofnums_rx, dofnums_c);
#     	c=c+1;
#     	# 11: b_3
#     	fill!(dofnums_c, c)
#     	assemble!(assembler, z.^1, dofnums_ry, dofnums_c);
#     	c=c+1;
#     end
#     return makematrix!(assembler);
# end

# function _transfmatrix(self::CoNCData, ::Val{3}, ::Val{3})
# 	ndof = ndofs(fld);
# 	@assert ndof == 3
# 	maxbn = 3 # Quadratic basis: Val{3}
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
#     startassembly!(assembler, elem_mat_nrows, 1, ncol+8*length(self.clusters), fld.nfreedofs, ncol)
#     c=1;
#     for  g in 1:length(self.clusters)
#     	dofnums_rx = fld.dofnums[self.clusters[g].nlist, 1]
#     	dofnums_ry = fld.dofnums[self.clusters[g].nlist, 2]
#     	dofnums_rz = fld.dofnums[self.clusters[g].nlist, 3]
#     	dofnums_c = fill(0, 1)
#     	x = reshape(self.clusters[g]._normalizedxyz[:, 1], length(dofnums_rx), 1)
#     	y = reshape(self.clusters[g]._normalizedxyz[:, 2], length(dofnums_rx), 1)
#     	z = reshape(self.clusters[g]._normalizedxyz[:, 3], length(dofnums_rx), 1)
#     	# 1: a_0
#     	fill!(dofnums_c, c)
#         assemble!(assembler, x.^0, dofnums_rx, dofnums_c);
#         c=c+1;
#         # 2: b_0
#         fill!(dofnums_c, c)
#         assemble!(assembler, y.^0, dofnums_ry, dofnums_c);
#         c=c+1;
#         # 3: c_0
#         fill!(dofnums_c, c)
#         assemble!(assembler, z.^0, dofnums_rz, dofnums_c);
#         c=c+1;
#         # 4: a_1
#         fill!(dofnums_c, c)
#         assemble!(assembler, x.^1, dofnums_rx, dofnums_c);
#         assemble!(assembler, -z.^1, dofnums_rz, dofnums_c);
#         c=c+1;
#         # 5: b_1
#         fill!(dofnums_c, c)
#         assemble!(assembler, x.^1, dofnums_ry, dofnums_c);
#         c=c+1;
#         # 6: c_1
#         fill!(dofnums_c, c)
#         assemble!(assembler, x.^1, dofnums_rz, dofnums_c);
#         c=c+1;
#         # 7: a_2
#         fill!(dofnums_c, c)
#         assemble!(assembler, y.^1, dofnums_rx, dofnums_c);
#         c=c+1;
#         # 8: b_2
#         fill!(dofnums_c, c)
#         assemble!(assembler, y.^1, dofnums_ry, dofnums_c);
#         assemble!(assembler, -z.^1, dofnums_rz, dofnums_c);
#         c=c+1;
#         # 9: c_2
#         fill!(dofnums_c, c)
#         assemble!(assembler, y.^1, dofnums_rz, dofnums_c);
#         c=c+1;
#         # 10: a_3
#         fill!(dofnums_c, c)
#         assemble!(assembler, z.^1, dofnums_rx, dofnums_c);
#         c=c+1;
#         # 11: b_3
#         fill!(dofnums_c, c)
#         assemble!(assembler, z.^1, dofnums_ry, dofnums_c);
#         c=c+1;
#         # 12: a_4
#         fill!(dofnums_c, c)
#         assemble!(assembler, x.^2, dofnums_rx, dofnums_c);
#         assemble!(assembler, -2 .* x.^1 .* z.^1, dofnums_rz, dofnums_c);
#         c=c+1;
#         # 13: a_5
#         fill!(dofnums_c, c)
#         assemble!(assembler, y.^2, dofnums_rx, dofnums_c);
#         c=c+1;
#         # 14: a_6
#         fill!(dofnums_c, c)
#         assemble!(assembler, z.^2, dofnums_rx, dofnums_c);
#         c=c+1;
#         # 15: a_7
#         fill!(dofnums_c, c)
#         assemble!(assembler, x.^1 .* y.^1, dofnums_rx, dofnums_c);
#         assemble!(assembler, -y.^1 .* z.^1, dofnums_rz, dofnums_c);
#         c=c+1;
#         # 16: a_8
#         fill!(dofnums_c, c)
#         assemble!(assembler, x.^1 .* z.^1, dofnums_rx, dofnums_c);
#         assemble!(assembler, -0.5 .* z.^2, dofnums_rz, dofnums_c);
#         c=c+1;
#         # 17: a_9
#         fill!(dofnums_c, c)
#         assemble!(assembler, y.^1 .* z.^1, dofnums_rx, dofnums_c);
#         c=c+1;
#         # 18: b_4
#         fill!(dofnums_c, c)
#         assemble!(assembler, x.^2, dofnums_ry, dofnums_c);
#         c=c+1;
#         # 19: b_5
#         fill!(dofnums_c, c)
#         assemble!(assembler, y.^2, dofnums_ry, dofnums_c);
#         assemble!(assembler, -2 .* y.^1 .* z.^1, dofnums_rz, dofnums_c);
#         c=c+1;
#         # 20: b_6
#         fill!(dofnums_c, c)
#         assemble!(assembler, z.^2, dofnums_ry, dofnums_c);
#         c=c+1;
#         # 21: b_7
#         fill!(dofnums_c, c)
#         assemble!(assembler, x.^1 .* y.^1, dofnums_ry, dofnums_c);
#         assemble!(assembler, -x.^1 .* z.^1, dofnums_rz, dofnums_c);
#         c=c+1;
#         # 22: b_8
#         fill!(dofnums_c, c)
#         assemble!(assembler, x.^1 .* z.^1, dofnums_ry, dofnums_c);
#         c=c+1;
#         # 23: b_9
#         fill!(dofnums_c, c)
#         assemble!(assembler, y.^1 .* z.^1, dofnums_ry, dofnums_c);
#         assemble!(assembler, -0.5 .* z.^2, dofnums_rz, dofnums_c);
#         c=c+1;
#         # 24: c_4
#         fill!(dofnums_c, c)
#         assemble!(assembler, x.^2, dofnums_rz, dofnums_c);
#         c=c+1;
#         # 25: c_5
#         fill!(dofnums_c, c)
#         assemble!(assembler, y.^2, dofnums_rz, dofnums_c);
#         c=c+1;
#         # 26: c_7
#         fill!(dofnums_c, c)
#         assemble!(assembler, x.^1 .* y.^1, dofnums_rz, dofnums_c);
#         c=c+1;
#     end
#     return makematrix!(assembler);
# end
