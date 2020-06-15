# Julia package for Repeated Decision Stumping
module RepeatedDecisionStumping

    using DataFrames, DecisionTree, Distributions, Plots, Seaborn, StatsBase

    export grouped_iterative_stumper, iterative_stumper, latex_table_from_dataframe, rs_stability

    function fβ_score(predictions, test_labels, label1, label2, β = 1)
        TP = sum(predictions .== test_labels .== label1)
        FP = sum(predictions .== label1 .!== test_labels)
        FN = sum(predictions .== label2 .!== test_labels)
        TN = sum(predictions .== test_labels .== label2)

        prec = TP / (TP + FP)
        recall = TP / (TP + FN)

        fβ = (1+β^2) * ((prec*recall)/((prec * β^2) + recall))

        return fβ
    end

    function iterative_stumper(data_in::DataFrame, quota::Int, tt_split::Float64, label1, label2, weighted::Bool=false)

        # Make a copy of the input dataframe. This copy will be permuted as
        # features are selected and removed during the iterative stumping
        data_edit = deepcopy(data_in)

        # Construct a new dataframe df, that will eventually be returned with
        # only the information pertaining to the chosen features
        df = DataFrame(Classification  = data_edit[:Classification])

        # Initialise an array to store the generated stump models
        stump_store = []

        # Initiialise a second dataframe stump_data, in which information about
        # each successive stump (and the stump model itself) will be store and
        # ulitimately returned
        stump_data = DataFrame(Rank = Int[], Id = Int[], Name = Any[],
         Threshold = Any[], Accuracy = Float64[], F1 = Float64[], Model = Any[])

        labels = map(string, data_edit[:Classification])

        train_filter, test_filter = get_tt_filters(
         labels, tt_split, weighted)

        train_labels = labels[train_filter]
        test_labels = labels[test_filter]


        for rank = 1:quota
            # Build the stump, first separating the features and the labels

            features = convert(Array{Float64}, data_edit[filter(x -> x != :Classification, names(data_edit))])
            train_features = features[train_filter,:]
            test_features = features[test_filter,:]

            gene_names = filter(x -> x != :Classification, names(data_edit))

            stump = build_tree(train_labels, train_features, 0, 1)


            # Obtain key information about each stump, to store in stump_data
            predictions = apply_tree(stump,test_features)
            conf = confusion_matrix(test_labels, predictions)
            accuracy = conf.accuracy

            # Compute the Fβ score
            Fβ = fβ_score(predictions, test_labels, label1, label2)

            local_featid = Int(stump.featid)

            name = string(gene_names[Int(local_featid)])
            threshold = stump.featval

            # Need to make the feature id selected by the model consistent with
            # the global feature list, not the one being whittled down in each
            # iteration of RS.
            global_featid = findfirst(names(data_in).==gene_names[local_featid])

            stump = DecisionTree.Node(global_featid,
             stump.featval,
             stump.left,
             stump.right)

            # Push this information about the stump to DF stump_data
            push!(stump_data, [rank global_featid name threshold accuracy Fβ stump])

            # Add the information contained about this feature in data_edit to df
            df[gene_names[local_featid]] = data_edit[gene_names[local_featid]]

            # Add the stump model to the store of stumps
            push!(stump_store, stump)

            # Remove this feature from data_edit, so that it is not considered again
            # in future stumps
            data_edit = data_edit[filter(x -> x != gene_names[local_featid], names(data_edit))]
         end

         # Return the dataframe containing the stumps created and their info
         return stump_data
    end

    function grouped_iterative_stumper(data, transitions, repeats, tt_split, weighted=false)

        out = []
        name_list = []

        for ii = 1:size(transitions, 1)

            label1 = transitions[ii,1]
            label2 = transitions[ii,2]

            t_only = data[((data[:Classification].==label1) .| (data[:Classification].==label2)), :]

            stump_data = iterative_stumper(t_only, repeats, tt_split, label1, label2, weighted)

            for ele in stump_data[:Name]
                push!(name_list, ele)
            end

            transition_statement = "$label1 - $label2"
            transition_dataframe = DataFrame(Transition = fill(transition_statement, repeats))

            # Force labels to be strings, to match model output
            label1 = string(label1)
            label2 = string(label2)

            progress_statements = []
            for stump_model in stump_data[:Model]
                left_vote = stump_model.left.majority

                if left_vote == label1
                    statement = "\$X > T \$" # X > T
                    push!(progress_statements, statement)
                elseif left_vote == label2
                    statement = "\$X \\leq T\$" # X <= T
                    push!(progress_statements, statement)
                else
                    @assert false "Stump vote does not match with either label in the given transition"
                end
            end

            stump_data = hcat(transition_dataframe, stump_data, DataFrame(Progress = progress_statements))
            push!(out, stump_data)
        end

        assembled = vcat(out...)
        return assembled
    end

    function get_tt_filters(full_labels::Array, tt_split::Float64, weighted::Bool=false)
        if weighted
            sw = Array{Any}(full_labels)
            num_labels = length(sw)

            for (key, value) in countmap(sw)
                sw = replace(x -> isequal(key, x) ? value : x, sw)
            end
            sw = weights([sw...])

            sample_indices = sample(collect(1:num_labels), sw, round(Int, tt_split*num_labels),
             replace=false, ordered=false)

            train_filter = [x in sample_indices for x in 1:num_labels]
            test_filter = map(!, train_filter)

            return train_filter, test_filter
        else
            # Define a Boolean variable to split the data into training and testing sets
            train_filter = rand(Bernoulli(tt_split), length(full_labels)) .== 1;
            test_filter = map(!, train_filter)

            return train_filter, test_filter
        end
    end

    # Function to generate LaTeX formatted table text from a Julia DataFrame
    function latex_table_from_dataframe(input_file::DataFrame, destination_file::String = "latex_table.txt")
        # Make copy of input DataFrame to access
        daf = deepcopy(input_file)

        # Capitalise column names
        vars = names(daf)
        for (i,v) in enumerate(vars)
            vars[i] = Symbol(titlecase(string(v)))
        end
        names!(daf,vars)

        # Set up running string to ultimately print to output .txt file
        runner = ""

        # Add table preamble and constuct initial columns
        runner*= "\\begin{tabular}{|"
        runner*= "c|"^size(daf,2)*"}\n"
        runner*= "\\hline \n"

        # Construct and add the header row for the table
        for header in broadcast(string, names(daf))
            runner*= header * " & "
        end
        runner = runner[1:end-2] * "\\\\ \n"#\\hline \n"

        # Set initial string for the t transition (for corrrect table lines)
        current = "xxx"
        same = fill("xxx", size(daf,2))

        # For each row of the DataFrame, add the entries element wise to the runner
        for row in eachrow(daf)
            row_text = ""
            # If the next row refers to the same epoch as the row before it,
            # then only need a truncated line heading from column 2 onwards
            still_matching = true
            same_index = 0
            while still_matching
                same_index += 1
                if same[same_index] != row[same_index]
                    still_matching = false
                end
            end

            if row != daf[1,:]
                c_lim = size(daf,2)
                resume_line = same_index
                runner*= "\\cline{$resume_line-$c_lim} \n"

                row_text *= " & " ^(same_index -1)
                for entry in convert(Array, row[:])[:][same_index:end]

                    # If number, then round
                    if typeof(entry) <: AbstractFloat
                        entry = round(entry, digits = 2)
                    end

                    # Convert to string
                    entry = string(entry)

                    # If the string contains underscores, then escape them correctly
                    entry = replace(entry, "_" => "\\_")
                    row_text *= entry * " & "
                end

                runner*= row_text[1:end-2] * "\\\\ \n"

            # Else the new line represents the first in a new epoch and should be
            # proceded by a full horizontal line
            else
                runner*= "\\hline \n"
                for entry in convert(Array, row[:])[:]
                    # If number, then round
                    if typeof(entry) <: AbstractFloat
                        entry = round(entry, digits = 2)
                    end

                    # Convert to string
                    entry = string(entry)
                    # If the string contains underscores, then escape them correctly
                    entry = replace(entry, "_" => "\\_")
                    row_text *= entry * " & "
                end

                runner*= row_text[1:end-2] * "\\\\ \n"
            end

            # Keep track of the current epoch, to compare the next line's to
            # current = row[1]
            same = row

        end

        # Add bottom line to table, along with command to \end{tabular}
        runner*= "\\hline\n\\end{tabular}\n"

        # Write the running string to the destination file
        open(destination_file, "w") do f
          write(f, runner)
        end

    end

    function rs_stability(in_data::DataFrame, transitions, partitions::Int,
         quota::Int, report::Int, tt_split::Float64, weighted::Bool=false)
        # partitions  - how many random partitions of the data
        # quota    - at each partition, consider the top 'quota' of features
        # tt_split - the training-test division

        over_all_transitions = []

        gene_list = map(string, names(in_data)[names(in_data) .!= :Classification])

        for trans_id = 1:size(transitions, 1)

            label1 = transitions[trans_id,1]
            label2 = transitions[trans_id,2]

            trans_list = fill(string(label1)*" - "*string(label2), length(gene_list))

            trans_data = in_data[((in_data[:Classification].==label1) .|
             (in_data[:Classification].==label2)), :]

            daf = DataFrame(Transition = trans_list,
             Rank = fill(0, length(gene_list)),
             Gene = gene_list,
             Stability = fill(0.0, length(gene_list)),
             MeanPerformance = fill(0.0, length(gene_list)),
             ValsPerformance = fill(Any[], length(gene_list)),
             # TODO Maybe we also care about the thresholds chosen?
             # Currently, this column is created but never filled
             ThreshVals = fill(Any[], length(gene_list)))


            # For each of the partitions...
            for partition = 1:partitions

                # TODO: Decide how and when to use tt_filter to test for stump
                # performance when calling rs_stability
                train_filter, test_filter = RepeatedDecisionStumping.get_tt_filters(
                 trans_data[:Classification], tt_split, weighted)

                train_data = trans_data[train_filter,:]
                test_data = trans_data[test_filter,:]

                # Iteratively create stumps on this data, identifying the best n features for
                # discerning the labels [in this case t=0 or t=24]
                stump_data = RepeatedDecisionStumping.iterative_stumper(train_data, quota, tt_split, label1, label2, weighted);
                # stump_data = out[1]

                for rank in stump_data[:Rank]
                    feat_name = stump_data[:Name][rank]
                    daf_rowid = findfirst(daf[:Gene].== feat_name)

                    daf[:Stability][daf_rowid] += (1+quota)-rank

                    # Evaluate model performance on withheld test data
                    model = stump_data[:Model][rank]

                    # in light of testing performance with tt split internally
                    # in iterative stumper, should just be able to pull the
                    # accuracy from the table directly
                    # test_labels = map(string, test_data[:Classification])
                    # test_features = convert(Array{Float64},
                    #  test_data[filter(x -> x != :Classification, names(test_data))])
                    #
                    # predictions = apply_tree(model,test_features)
                    # predictions = map(string, predictions)
                    # conf = confusion_matrix(test_labels, predictions)
                    # accuracy = conf.accuracy

                    accuracy = stump_data[:Accuracy][rank]

                    daf[:ValsPerformance][daf_rowid] =
                     [daf[:ValsPerformance][daf_rowid]...,accuracy]

                    model_threshold = stump_data[:Threshold][rank]
                    daf[:ThreshVals][daf_rowid] =
                     [daf[:ThreshVals][daf_rowid]...,model_threshold]
                end
            end

            for (ii,eacharray) in enumerate(daf[:ValsPerformance])
                if length(eacharray) > 0
                    daf[:MeanPerformance][ii] = mean(eacharray)
                end
            end

            daf[:Stability] = daf[:Stability] ./ (partitions*quota)
            sort!(daf, (:Stability), rev=true)

            # Only show the top (quota) most stable features
            daf = daf[1:report,:]

            push!(over_all_transitions, daf)
        end

        stability_df = vcat(over_all_transitions...)

        rank_list = repeat(collect(1:report); outer=[size(transitions,1)])
        stability_df[:Rank] = rank_list

        return stability_df
    end

    function stability_plot(data, stab_daf, transitions, repeats, quota, report)

      plot_store = []

      for trans_id = 1:size(transitions,1)
        label1 = transitions[trans_id, 1]
        label2 = transitions[trans_id, 2]

        start_row =1+(report*(trans_id-1))
        end_row = report*trans_id

        curr_plot = Plots.plot(stab_daf[:Stability][start_row:end_row],
         linestyle = :solid,
         linecolor = :red,

         # xlabel = "Ranked features",
         ylabel = "Stability",

         title = "Feature stability at "*string(label1)*"/"*string(label2)*" boundary",
         ylim = (0,1),
         xticks = (collect(1:report), stab_daf[:Gene][start_row:end_row]), rotation = 90,
         legend = false, labels = ["Peturbation"])#, xlim = (1,quota))

         # What about complete noise. or perfectly ordered data?
         # Perfect order...
        perfect = collect(quota:-1:1) .* repeats
        perfect = perfect ./ (repeats*quota)
        plot!(perfect[1:report], labels = ["Peturbation", "No peturbation"],
         linestyle = :dash, linecolor = :black)

        # Naive order
        naive = fill(1/size(data,2), quota)
        plot!(naive[1:report], labels = ["Peturbation","No peturbation", "Naïve"],
         linestyle = :solid, linecolor = :blue, margin=5Plots.mm)


        push!(plot_store, curr_plot)
      end

      plot(plot_store..., layout = (length(plot_store),1), grid = false, legend = :topright)
    end

    function plot_thresholds(expr_data, annotations, rs_results, raw_repeats,
         view_ranks, transitions, subplot_size, histbool = true, kdebool = true, rugbool = true)

        colour_counter = 1
        colour_dict = Dict()

        # Set up Seaborn "muted" colour palette
        col_palette = [(0.2823529411764706, 0.47058823529411764, 0.8156862745098039),
         (0.9333333333333333, 0.5215686274509804, 0.2901960784313726),
         (0.41568627450980394, 0.8, 0.39215686274509803),
         (0.8392156862745098, 0.37254901960784315, 0.37254901960784315),
         (0.5843137254901961, 0.4235294117647059, 0.7058823529411765),
         (0.5490196078431373, 0.3803921568627451, 0.23529411764705882),
         (0.8627450980392157, 0.49411764705882355, 0.7529411764705882),
         (0.4745098039215686, 0.4745098039215686, 0.4745098039215686),
         (0.8352941176470589, 0.7333333333333333, 0.403921568627451),
         (0.5098039215686274, 0.7764705882352941, 0.8862745098039215)]

        f, axes = plt.subplots(size(transitions,1), length(view_ranks), figsize=subplot_size)

        # For each boundary or interest...
        for ii_transition = 1:size(transitions, 1)
            # Get the start and end labels
            l_start = transitions[ii_transition,1]
            l_end = transitions[ii_transition,2]

            # Add any new entires to the colour dictionary
            if ! haskey(colour_dict, l_start)
                colour_dict[l_start] = col_palette[colour_counter]
                colour_counter = colour_counter%length(col_palette) + 1
            end

            if ! haskey(colour_dict, l_end)
                colour_dict[l_end] = col_palette[colour_counter]
                colour_counter = colour_counter%length(col_palette) + 1
            end

            # Construct the data filters using the labels
            f_start = annotations .== l_start
            f_end = annotations .== l_end

            # For each rank that we have chosen to view,,,
            for (plot_column, rank) in enumerate(view_ranks)

                # Identify the row index in name_thresh this rank/time boundary combo refers to
                pick = rank + raw_repeats*(ii_transition-1)

                # Use this index to pull out the appropriate name and threshold
                gene_name = rs_results[:Name][pick]
                threshold = rs_results[:Threshold][pick]

                # Use the gene name to pull out the raw data for plotting
                gene_data = expr_data[Symbol(gene_name)]

                curr_axis = axes[ii_transition, plot_column]

                plt.figure()
                Seaborn.distplot(gene_data[f_start], ax = curr_axis,
                 color = colour_dict[l_start],
                 hist=histbool, kde = kdebool, rug=rugbool)
                Seaborn.distplot(gene_data[f_end], ax = curr_axis,
                 color = colour_dict[l_end], hist=histbool,
                 kde = kdebool, rug=rugbool)

                # Set the appropriate xlabel
                curr_axis.set_xlabel(gene_name,fontsize = 18)

                # Add the correct threshold as a vertical line
                curr_axis.axvline(threshold, 0,1.0, c = "k").set_linestyle("--");

                # Fix x axis from 0
                curr_axis.set_xlim(0,)

                # Scale y-axis for legibility
                ymin, ymax = curr_axis.get_ylim()
                if ymax > 0.5
                    axes[ii_transition, plot_column].set_ylim(0,0.3)
                end

                if ii_transition == 1
                    axes[ii_transition, plot_column].set_title("Rank "* string(rank), fontsize = 24)
                end

            end
        end
    end
end
