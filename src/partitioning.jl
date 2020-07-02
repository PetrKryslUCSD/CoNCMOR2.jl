
function pointpartitioning2d(points, nincluded, npartitions)
    coordinatetype = eltype(points[1])
    function inertialcutpartitioning!(partitions, parts, X)
        nspdim = 2
        @assert nspdim == length(X[1])
        StaticMoments = fill(zero(coordinatetype), nspdim, length(parts));
        npart = fill(0, length(parts))
        for spdim in 1:nspdim
            @inbounds for j in 1:length(X)
                if nincluded[j] # Is the point to be included in the partitioning?
                    jp = partitions[j]
                    StaticMoments[spdim, jp] += X[j][spdim]
                    npart[jp] += 1 # count the points in the current partition
                end
            end
        end
        CG = fill(zero(coordinatetype), nspdim, length(parts));
        for p = parts
            npart[p] = Int64(npart[p] / nspdim)
            CG[:, p] = StaticMoments[:, p] / npart[p] # center of gravity of each partition
        end
        MatrixMomentOfInertia = fill(zero(coordinatetype), nspdim, nspdim, length(parts))
        @inbounds for j in 1:length(X)
            if nincluded[j] # Is the point to be included in the partitioning?
                jp = partitions[j]
                xj, yj = X[j][1] - CG[1, jp], X[j][2] - CG[2, jp]
                MatrixMomentOfInertia[1, 1, jp] += yj^2
                MatrixMomentOfInertia[2, 2, jp] += xj^2
                MatrixMomentOfInertia[1, 2, jp] -= xj * yj
            end
        end
        for p in parts
            MatrixMomentOfInertia[2, 1, p] = MatrixMomentOfInertia[1, 2, p]
        end
        longdir = fill(zero(coordinatetype), nspdim, length(parts))
        for p in parts
            F = eigen(MatrixMomentOfInertia[:, :, p])
            six = sortperm(F.values)
            longdir[:, p] = F.vectors[:, six[1]]
        end
        toggle = fill(one(coordinatetype), length(parts));
        @inbounds for j in 1:length(X)
            if nincluded[j] # Is the point to be included in the partitioning?
                jp = partitions[j]
                vx, vy = longdir[:, jp]
                xj, yj = X[j][1] - CG[1, jp], X[j][2] - CG[2, jp]
                d = xj * vx + yj * vy
                c = 0
                if d < 0.0
                    c = 1
                elseif d > 0.0
                    c = 0
                else # disambiguate d[ixxxx] == 0.0
                    c = (toggle[jp] > 0) ? 1 : 0
                    toggle[jp] = -toggle[jp]
                end
                partitions[j] = 2 * jp - c
            end
        end
    end

    nlevels = Int(round(ceil(log(npartitions)/log(2))))
    partitions = fill(1, length(points))  # start with points assigned to partition 1
    for level = 0:1:(nlevels - 1)
        inertialcutpartitioning!(partitions, collect(1:2^level), points)
    end
    return partitions
end

function pointpartitioning2d(points, npartitions = 2)
    return pointpartitioning2d(points, [true for idx in points], npartitions)
end


function pointpartitioning3d(points, nincluded, npartitions)
    coordinatetype = eltype(points[1])
    function inertialcutpartitioning!(partitions, parts, X)
        nspdim = 3
        @assert nspdim == length(X[1])
        StaticMoments = fill(zero(coordinatetype), nspdim, length(parts));
        npart = fill(0, length(parts))
        for spdim in 1:nspdim
            @inbounds for j in 1:length(X)
                if nincluded[j] # Is the point to be included in the partitioning?
                    jp = partitions[j]
                    StaticMoments[spdim, jp] += X[j][spdim]
                    npart[jp] += 1 # count the points in the current partition
                end
            end
        end
        CG = fill(zero(coordinatetype), nspdim, length(parts));
        for p = parts
            npart[p] = Int64(npart[p] / nspdim)
            CG[:, p] = StaticMoments[:, p] / npart[p] # center of gravity of each partition
        end
        MatrixMomentOfInertia = fill(zero(coordinatetype), nspdim, nspdim, length(parts))
        @inbounds for j in 1:length(X)
            if nincluded[j] # Is the point to be included in the partitioning?
                jp = partitions[j]
                xj, yj, zj = X[j][1] - CG[1, jp], X[j][2] - CG[2, jp], X[j][3] - CG[3, jp]
                MatrixMomentOfInertia[1, 1, jp] += yj^2 + zj^2
                MatrixMomentOfInertia[2, 2, jp] += xj^2 + zj^2
                MatrixMomentOfInertia[3, 3, jp] += yj^2 + xj^2
                MatrixMomentOfInertia[1, 2, jp] -= xj * yj
                MatrixMomentOfInertia[1, 3, jp] -= xj * zj
                MatrixMomentOfInertia[2, 3, jp] -= yj * zj
            end
        end
        for p in parts
            MatrixMomentOfInertia[2, 1, p] = MatrixMomentOfInertia[1, 2, p]
            MatrixMomentOfInertia[3, 1, p] = MatrixMomentOfInertia[3, 1, p]
            MatrixMomentOfInertia[3, 2, p] = MatrixMomentOfInertia[3, 2, p]
        end
        longdir = fill(zero(coordinatetype), nspdim, length(parts))
        for p in parts
            F = eigen(MatrixMomentOfInertia[:, :, p])
            six = sortperm(F.values)
            longdir[:, p] = F.vectors[:, six[1]]
        end
        toggle = fill(one(coordinatetype), length(parts));
        @inbounds for j in 1:length(X)
            if nincluded[j] # Is the point to be included in the partitioning?
                jp = partitions[j]
                vx, vy, vz = longdir[:, jp]
                xj, yj, zj = X[j][1] - CG[1, jp], X[j][2] - CG[2, jp], X[j][3] - CG[3, jp]
                d = xj * vx + yj * vy + zj * vz
                c = 0
                if d < 0.0
                    c = 1
                elseif d > 0.0
                    c = 0
                else # disambiguate d[ixxxx] == 0.0
                    c = (toggle[jp] > 0) ? 1 : 0
                    toggle[jp] = -toggle[jp]
                end
                partitions[j] = 2 * jp - c
            end
        end
    end

    nlevels = Int(round(ceil(log(npartitions)/log(2))))
    partitions = fill(1, length(points))  # start with points assigned to partition 1
    for level = 0:1:(nlevels - 1)
        inertialcutpartitioning!(partitions, collect(1:2^level), points)
    end
    return partitions
end

function pointpartitioning3d(points, npartitions = 2)
    return pointpartitioning3d(points, [true for idx in points], npartitions)
end
