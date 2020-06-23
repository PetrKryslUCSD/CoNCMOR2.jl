using Test

module mpart1
using StaticArrays
using Test
using CoNCMOR: pointpartitioning2d
function test()
    points = [SVector{2}([1.0, 0.0]), SVector{2}([1.0, 1.0]), SVector{2}([-1.0, -1.0]), SVector{2}([-1.0, 1.0])]
    p = pointpartitioning2d(points, 2)
@test isapprox(p, [1, 1, 2, 2])
end
end
using .mpart1
mpart1.test()

module mpart2
using StaticArrays
using Test
using PlotlyJS
using CoNCMOR: pointpartitioning2d
colors = ["rgb(164, 194, 244)", "rgb(194, 194, 144)", "rgb(194, 144, 244)", "rgb(164, 244, 144)", "rgb(164, 194, 244)", "rgb(255, 217, 102)", "rgb(234, 153, 153)", "rgb(142, 124, 195)"]
function test()
    points = [SVector{2}([rand(), rand()])  for idx in 1:1000]
    ppartitioning = pointpartitioning2d(points, 8)
    partitionnumbers = unique(ppartitioning)
    # data = PlotlyBase.AbstractTrace[]
    # for gp in partitionnumbers
    #     trace1 = scatter(; 
    #         x=[points[i][1] for i in 1:length(points) if ppartitioning[i] == gp], 
    #         y=[points[i][2] for i in 1:length(points) if ppartitioning[i] == gp] ,
    #       mode="markers",
    #       marker=attr(color=colors[gp], size=12,
    #           line=attr(color="white", width=0.5))
    #       )
    #     push!(data, trace1)
    # end
    # layout = Layout(;title="Quarter 1 Growth",
    #     xaxis=attr(title="x", zeroline=false),
    #     yaxis=attr(title="y", zeroline=false))

    # pl = plot(data, layout)
    # display(pl)
end
end
using .mpart2
mpart2.test()


module mpart3
using StaticArrays
using Test
using PlotlyJS
using CoNCMOR: pointpartitioning2d, CoNCData, nclusters, nfuncspercluster
using CoNCMOR: nbasisfunctions, transfmatrix, LegendreBasis
colors = ["rgb(164, 194, 244)", "rgb(194, 194, 144)", "rgb(194, 144, 244)", "rgb(164, 244, 144)", "rgb(164, 194, 244)", "rgb(255, 217, 102)", "rgb(234, 153, 153)", "rgb(142, 124, 195)"]
function test()
    points = [SVector{2}([rand(), rand()])  for idx in 1:1000]
    ppartitioning = pointpartitioning2d(points, 8)
    partitionnumbers = unique(ppartitioning)
    # data = PlotlyBase.AbstractTrace[]
    # for gp in partitionnumbers
    #     trace1 = scatter(; 
    #         x=[points[i][1] for i in 1:length(points) if ppartitioning[i] == gp], 
    #         y=[points[i][2] for i in 1:length(points) if ppartitioning[i] == gp] ,
    #       mode="markers",
    #       marker=attr(color=colors[gp], size=12,
    #           line=attr(color="white", width=0.5))
    #       )
    #     push!(data, trace1)
    # end
    # layout = Layout(;title="Quarter 1 Growth",
    #     xaxis=attr(title="x", zeroline=false),
    #     yaxis=attr(title="y", zeroline=false))

    # pl = plot(data, layout)
    # display(pl)
    mor = CoNCData(points, ppartitioning)
    @test nclusters(mor) == length(partitionnumbers)
    I, J, V  = transfmatrix(mor, LegendreBasis, 1)
    @test nfuncspercluster(mor) == 1
    @show nbasisfunctions(mor)
end
end
using .mpart3
mpart3.test()

