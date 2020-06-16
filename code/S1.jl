# Supplementary script comparing feature rankings between ReDX and Mann-
# Whitney U Test
using DataFrames, Random, HypothesisTests, Plots

include("./RepeatedDecisionStumping.jl")


# FUNCTION DEFINITIONS
# Bejamini-Hochberg correction for multiple testing
# Function takes a sorted list of p-values and the local FDR (Q)
function benjamini_hochberg(ps, Q, return_p_ranks_instead = false)

    m = length(ps) # Total number of tests
    ranks = collect(1:m)

    ps_sortperm = sortperm(ps)
    sorted_ps = ps[ps_sortperm]

    BH = (ranks./m).*Q

    a = findlast(sorted_ps.<BH)

    maximum_accepted_p = sorted_ps[a]
    println("Max sorted p ", maximum_accepted_p)

    if a == nothing
        return BH_accepted = []
    elseif return_p_ranks_instead == true
        println("BH accepted ", sum(ps.<=maximum_accepted_p), " / ", length(ps))
        println("BH accepted values", ps.<=maximum_accepted_p)
        return bh_accepted = ps.<=maximum_accepted_p
    else
        return accepted_ps = sorted_ps[1:a]
    end
end

function get_MW_genes(df, transitions, Q, repeats, tt_split, sort_by_U = true)

    labels = df[:Classification]
    MW_results = []

    for ii = 1:size(transitions,1)
        label1 = transitions[ii,1]
        label2 = transitions[ii,2]

        println(label1)

        data1 = df[labels.==label1,:]
        data2 = df[labels.==label2,:]

        n = size(df,2)-1

        zs = []

        for gene_id = 1:n
            g1 = float.(data1[:,gene_id])
            g2 = float.(data2[:,gene_id])


            push!(zs, MannWhitneyUTest(g1, g2))
        end

        genes = string.(names(df))[1:end-1]

        # Peform Benjamini-Hochberg correction for multiple testing
        u_values = [x.U for x in zs]

        if sort_by_U == true
            #Sort genes by U
            sorting_values = u_values
            sorting_indices = sortperm(u_values)

            p_values = map(pvalue, zs[sorting_indices])
        else
            # Sort genes by p-value
            p_values = map(pvalue, zs)

            sorting_values = p_values
            sorting_indices = sortperm(p_values)

        end

        bh_accepted = benjamini_hochberg(p_values, Q, true)
        MW_genes = genes[bh_accepted][sortperm(sorting_values[bh_accepted])]

        u_plot = plot(u_values[sorting_indices], ylabel = "U")
        u_bh_plot = plot(u_values[bh_accepted][sortperm(sorting_values[bh_accepted])], ylabel = "U")

        p_plot = plot(p_values[sorting_indices], ylabel = "p-value")
        println("OUTSIDE")
        println(sum(bh_accepted))
        println("Length of accepted array", length(p_values[bh_accepted]))
        println("p_values[bh_accepted]", p_values[bh_accepted])
        p_bh_plot = plot(p_values[bh_accepted][sortperm(sorting_values[bh_accepted])], ylabel = "p-value")
        println("Plotting p-values:", length(p_values[sortperm(p_values)[bh_accepted]]))
        println(p_values[bh_accepted][sortperm(sorting_values[bh_accepted])])

        plot(u_plot, u_bh_plot, p_plot, p_bh_plot, layout = (4,1))

        push!(MW_results, MW_genes)
    end

    return MW_results
end

function trim_lists(a,b)
    min_length = min(length(a), length(b))
    a = a[1:min_length]
    b = b[1:min_length]

    a = map(string, a)
    b = map(string, b)

    return a,b
end

function rank_comparison_plot(l1, l2, fs = 5)
    n1 = length(l1)
    n2 = length(l2)

    x1 = 1
    x2 = 1.5

    text_offset = 0.02

    scatter(fill(x1, n1), collect(1:n1), markersize = 5)
    annotate!(fill(x1 - text_offset, n1), collect(1:n1), text.(l1, :right, fs))

    scatter!(fill(x2, n2), collect(1:n2), markersize = 5)
    annotate!(fill(x2 + text_offset, n2), collect(1:n2), text.(l2, :left, fs))

    for (l1_id, l1_gene) in enumerate(l1)
        l2_id = findfirst(isequal(l1_gene), l2)
        if l2_id != nothing
            plot!([x1, x2], [l1_id, l2_id], linecolor = :black)
        end
    end

    yflip!(xaxis=false, yaxis=false, grid=false, legend=false)
    xlims!((x1 - 1 ,x2 + 1))
end

# Set to false to rank surviving Mann-Whittney features by p-value
rank_by_U = false


# MURINE RESULTS ////////////////////////////////////////////////////////////
# DATA ENTRY
mur_data = readtable("../data/murine/murine_expression.csv");

# Read in the associated annotation data
mur_labels = readtable("../data/murine/murine_annotations.csv");

# Add the class labels as a column "Classifcation" to the expression data
mur_data[:Classification] = mur_labels[:Class];

# DATA ANALYSIS
Q = 0.01
repeats = 50
tt_split = 0.8

mur_transitions = ["ESC" "EPI"; "EPI" "NPC"]

Random.seed!(1234);

mur_stumps = RepeatedDecisionStumping.grouped_iterative_stumper(mur_data,
 mur_transitions, repeats, tt_split)

(MW_esc_epi_genes, MW_epi_npc_genes) = get_MW_genes(
 mur_data, mur_transitions, Q, repeats, tt_split, rank_by_U)

# Now compare the lists from ReDX and from MannWhitneyU
ReDX_esc_epi_genes = mur_stumps[:Name][mur_stumps[:Transition] .== "ESC - EPI"]
ReDX_epi_npc_genes = mur_stumps[:Name][mur_stumps[:Transition] .== "EPI - NPC"]

MW_esc_epi_genes, ReDX_esc_epi_genes = trim_lists(MW_esc_epi_genes, ReDX_esc_epi_genes)
MW_epi_npc_genes, ReDX_epi_npc_genes = trim_lists(MW_epi_npc_genes, ReDX_epi_npc_genes)


# Plot the ranked features from ReDX and the Mann-Whitney U test
# Blue is Mann-Whitney U, red is RedX
rank_comparison_plot(MW_esc_epi_genes, ReDX_esc_epi_genes)
rank_comparison_plot(MW_epi_npc_genes, ReDX_epi_npc_genes)


# XENOPUS RESULTS //////////////////////////////////////////////////////////////

# DATA ENTRY
# Takes ~11 minutes to read in full Xenopus data
xen_df = readtable("../data/xenopus/xenopus_expression.csv")


 # Read in the associated annotations, adding the Celltype label to the xen_df
 xen_labels = readtable("../data/xenopus/xenopus_annotations.csv");

 xen_df[:Classification] = xen_labels[:Cluster_Label]

xen_transitions = [["S08-blastula","S08-blastula","S08-blastula","S08-blastula",
 "S08-blastula","S08-blastula"] ["S10-neuroectoderm", "S10-non-neural ectoderm",
 "S10-marginal zone", "S10-Spemann organizer (mesoderm)",
 "S10-Spemann organizer (endoderm)", "S10-endoderm"]]

# DATA ANALYSIS
repeats = 50
tt_split = 0.8

Random.seed!(1234);
xen_stumps = RepeatedDecisionStumping.grouped_iterative_stumper(xen_df,
 xen_transitions, repeats, tt_split)

xen_MW_results = get_MW_genes(
 xen_df, xen_transitions, Q, repeats, tt_split, rank_by_U)

# Compare the ranked features from ReDX and the Mann-Whitney U test
# Blue is Mann-Whitney U, red is RedX

rank_comparison_plot(trim_lists(xen_MW_results[1], xen_stumps[:Name][1:50])...)
rank_comparison_plot(trim_lists(xen_MW_results[2], xen_stumps[:Name][51:100])...)
rank_comparison_plot(trim_lists(xen_MW_results[3], xen_stumps[:Name][101:150])...)
rank_comparison_plot(trim_lists(xen_MW_results[4], xen_stumps[:Name][151:200])...)
rank_comparison_plot(trim_lists(xen_MW_results[5], xen_stumps[:Name][201:250])...)
rank_comparison_plot(trim_lists(xen_MW_results[6], xen_stumps[:Name][251:300])...)
