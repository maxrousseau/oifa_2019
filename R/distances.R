#' Computation and plotting of the landmark distances
#' 
#' - Shape alignment (shapes) (intra group subjects -> inter group mean shapes)
#' - Compute the landmark distances
#' - Group landmarks
#' - Semantic distance plot
#'
#' Author: Maxime Rousseau
#' Date: 24/06/2020

library(here)
library(tibble)
library(dplyr)
library(purrr)
library(stringr)
library(shapes)
library(rstatix)
 
distances <- function()
{
    # load data
    ldmk_db <- read_tbl()

	# LANDMARK DISTANCE  ----------------------------------------
    # generalized procuste analysis 1 (intra group: subjects)
    df_c <- proc(group_matrix(ldmk_db, "c"))
    df_1 <- proc(group_matrix(ldmk_db, "1"))
    df_3 <- proc(group_matrix(ldmk_db, "3"))
    df_4 <- proc(group_matrix(ldmk_db, "4"))

    # baseline mean shape
    baseline <- mshape(group_matrix(ldmk_db, "c"))
	ms_1 <- mshape(group_matrix(ldmk_db, "1"))
	ms_3 <- mshape(group_matrix(ldmk_db, "3"))
	ms_4 <- mshape(group_matrix(ldmk_db, "4"))

    # generalized procuste analysis 2 (inter group: subjects)
    all_ms <- proc(simplify2array(list(baseline, ms_1, ms_3, ms_4)))
    baseline <- all_ms[,,1]
    ms_1 <- all_ms[,,2]
    ms_3 <- all_ms[,,3]
    ms_4 <- all_ms[,,4]

    # compute distances
    dist_1 <- dist(baseline, ms_1)
    dist_3 <- dist(baseline, ms_3)
    dist_4 <- dist(baseline, ms_4)

	# FACIAL RATIO ----------------------------------------
	# compute ratios on the non-rotated faces
	face_ratio <- ratio_table(ldmk_db)

    # mean shape landmark distances
    dist_tbl <- tibble(labels = as_vector(mklabel_tbl(ldmk_db)$label),
					   type_1 = dist_1,
					   type_3 = dist_3,
					   type_4 = dist_4)

    # mean shape landmark distance, label means
    mdist_tbl <- dist_tbl %>%
        group_by(labels) %>%
        summarise_all(mean) %>%
        pivot_longer(cols = c("type_1", "type_3", "type_4"),
                     values_to="mean_dist", names_to="group") %>%
    add_row(labels = "all", group = "type_1", mean_dist =
    mean(as_vector(dist_tbl$type_1))) %>%
    add_row(labels = "all", group = "type_3", mean_dist =
    mean(as_vector(dist_tbl$type_3))) %>%
    add_row(labels = "all", group = "type_4", mean_dist =
    mean(as_vector(dist_tbl$type_4)))



	# PLOTS ----------------------------------------
    # plot ladmark labels (soft tissue area)
    label_plot(ldmk_db)

    # distance plot
    distance_plot(mdist_tbl)

	# TESTS  ----------------------------------------
    a1 <- face_ratio %>% anova_test(ratio1 ~ group)
    a2 <- face_ratio %>% anova_test(ratio2 ~ group)
    a3 <- face_ratio %>% anova_test(ratio3 ~ group)
    a4 <- face_ratio %>% anova_test(lfh ~ group)

    a_res <- bind_rows(a1, a2, a3, a4)
    write_csv(a_res, "anova.csv")

    b1 <- face_ratio %>% pairwise_t_test(ratio1 ~ group, p.adjust.method = "bonferroni")
    b2 <- face_ratio %>% pairwise_t_test(ratio2 ~ group, p.adjust.method = "bonferroni")
    b3 <- face_ratio %>% pairwise_t_test(ratio3 ~ group, p.adjust.method = "bonferroni")
    b4 <- face_ratio %>% pairwise_t_test(lfh ~ group, p.adjust.method = "bonferroni")

    b_res <- bind_rows(b1, b2, b3, b4)
    write_csv(b_res, "bonf.csv")

}

label_mean <- function(x, l)
{
    #' x: distance with the table of all 68 landmarks
    #' l: label
    x %>% filter(labels == l)
}

lbl_vec <- function()
{
    # vector of coordinates according to anatomy
    temple <- c(1, 2, 16, 17)
    brows <- c(18:22, 23:27)
    eyes <- c(37:42, 43:48)
    nose <- c(28:36)
    mouth <- c(49:68)
    jaw <- c(3:15)

    c(temple, brows, eyes, nose, mouth, jaw)
}

ratio_table <- function(ldmk_db)
{
	# distances
	face_height <- euc_dist(ldmk_db, 28, 9)
	lower_face <- euc_dist(ldmk_db, 34, 9)
	temporal <- euc_dist(ldmk_db, 1, 17)
	ocular <- euc_dist(ldmk_db, 37, 46)
	mandibular <- euc_dist(ldmk_db, 5, 13)

	# ratios
	ratio1 <- ratio(ocular, temporal)
	ratio2 <- ratio(mandibular, temporal)
	ratio3 <- ratio(temporal, face_height)
	lfh <- ratio(lower_face, face_height)

	group <- ldmk_db %>%
				select(Group) %>%
				as_vector()

	tibble(group,
		   ratio1,
		   ratio2,
		   ratio3,
		   lfh)
}

euc_dist <- function(df, a, b)
{
	# compute the euclidean distance between two coordinates
	# df: dataframe containing landmarks
	# a: landmark one
	ax <- vectorize(df, str_c(a, "x"))
	ay <- vectorize(df, str_c(a, "y"))
	bx <- vectorize(df, str_c(b, "x"))
	by <- vectorize(df, str_c(b, "y"))

	sqrt((ax - bx)^2 + (ay - by)^2)

}

vectorize <- function(df, ldmk)
{
	# compute the landmark
	# df: dataframe
	# ldmk: landmark to be vectorized

	df %>%
		select(ldmk) %>%
		as_vector()
}

ratio <- function(a, b)
{
	# compute the ratio between the distances inputed
	(a / b) * 100
}

read_tbl <- function()
{
    readRDS(here("data/oifaces.rds"))
}

distance_plot <- function(x)
{
	# plotting the semantic distances
	# x: semantic distance tibble
    ggplot(data = x) +
        geom_point(mapping = aes(x = mean_dist, y = labels, color = group),
                   alpha = 0.8, size=3)
    ggsave("semantic_distance_plot.tiff", width = 7, height = 4)

}

mklabel_tbl <- function(x)
{
    rand_obs <- filter(x, Group == "c") %>%
        filter(Group == "c") %>%
        slice(1)

    temple <- vec_coords(rand_obs, var_names(c(1, 2, 16, 17)))
    brows <- vec_coords(rand_obs, var_names(c(18:22, 23:27)))
    eyes <- vec_coords(rand_obs, var_names(c(37:42, 43:48)))
    nose <- vec_coords(rand_obs, var_names(c(28:36)))
    mouth <- vec_coords(rand_obs, var_names(c(49:68)))
    jaw <- vec_coords(rand_obs, var_names(c(3:15)))

    tibble(label = character(),
                         coords_x = double(),
                         coords_y = double()) %>%
        add_row(label = "temple",
                coords_x = temple[[1]],
                coords_y = temple[[2]]) %>%
        add_row(label = "brows",
                coords_x = brows[[1]],
                coords_y = brows[[2]]) %>%
        add_row(label = "eyes",
                coords_x = eyes[[1]],
                coords_y = eyes[[2]]) %>%
        add_row(label = "nose",
                coords_x = nose[[1]],
                coords_y = nose[[2]]) %>%
        add_row(label = "mouth",
                coords_x = mouth[[1]],
                coords_y = mouth[[2]]) %>%
        add_row(label = "jaw",
                coords_x = jaw[[1]],
                coords_y = jaw[[2]])
}


label_plot <- function(x)
{

	shapes_tbl <- mklabel_tbl(x)

    ggplot(data = shapes_tbl) +
        geom_point(mapping = aes(x = coords_x, y = coords_y, color = label)) +
        scale_x_reverse() +
        scale_y_reverse() +
        theme(axis.line = element_blank(),
              axis.text = element_blank(),
              axis.title = element_blank(),
              axis.ticks = element_blank(),
              panel.background = element_blank()) + 
        labs(color = "Anatomic Labels")

    ggsave("semantic_labels.tiff", width = 7, height = 7)
}

make_vec <- function(df, a, b)
{
    #' create vector of landmarks from dataset x
    #' a: chr x or y coordinate
    #' b: row of the data to vectorize
    df %>%
        slice(b) %>%
        select(contains(a)) %>%
        as_vector() %>%
        unname
}

ldmk_matrix <- function(df)
{
    #' input x y landmark vectors
    #' return 68x2 matrix
    #' df: data frame (tibble)
    matrix_ls <- list()
    
    for (i in 1:length(df$Group))
    {
        matrix_ls[[i]] <- matrix(c(make_vec(df, "x", i),
                                   make_vec(df, "y", i)),
                                   nrow=68, ncol=2, byrow=FALSE)
    }

    return(matrix_ls)
}

group_matrix <- function(df, g)
{
    #' create matrix for a given group
    #' df: data frame of all groups
    #' g: group to isolate into matrix
    simplify2array(ldmk_matrix(filter(df, Group == g)))
}

label_tbl <- function(x)
{
    # map the creation of the table from the above function

    # vector of coordinates according to anatomy
    temple <- c(1, 2, 16, 17)
    brows <- c(18:22, 23:27)
    eyes <- c(37:42, 43:48)
    nose <- c(28:36)
    mouth <- c(49:68)
    jaw <- c(3:15)

    label_vec <- c(temple, brows, eyes, nose, mouth, jaw)

    # map creation of table
    rand_obs <- filter(x, Group == "c") %>%
        filter(Group == "c") %>%
        slice(1)

    label_vec %>%
        map(vec_coords(rand_obs, var_names(.)))

}

vec_coords <- function(x, y)
{
                                        # x df
                                        # y coordinates labels
    x_coord <- as_vector(
        select(x, y[[1]]))
    y_coord <- as_vector(
        select(x, y[[2]]))

    list(x_coord, y_coord)

}

var_names <- function(x)
{
    ## takes vector of coordinates and returns formatted variable names
    ## which are compatible with the current table format

    x_labels <- str_c(as.character(x), "x")
    y_labels <- str_c(as.character(x), "y")
    list(x_labels, y_labels)

}

dist <- function(x, y)
{
    # compute euclidean distance between landmark x (int)
    # x: baseline shape
    # y: mean shape of group

    # euclidean distance
    #for (i in 1:length(l[1,1,])){
    #    if (exists("tmp_vec"))
    #    {
    #        tmp_vec <- cbind(tmp_vec, sqrt((x[,1] - l[,1,i])^2 + (x[,2] - l[,2,i])^2))
    #    }
    #    else {
    #        tmp_vec <- sqrt((x[,1] - l[,1,i])^2 + (x[,2] - l[,2,i])^2)
    #    }
    #}
    #tmp_vec
	sqrt((x[,1] - y[,1])^2 + (x[,2] - y[,2])^2)

}

semantic_dist <- function(x, lb)
{
    #' compute mean distance of semantic landmark group
    #' x: landmark distance vector (size 68)
    #' lb: vector of landmarks of interest
    apply(x[lb,], 2, mean)
}

summary_dist <-  function(x, g)
{
    #' summarize the distances into a tibble
    #' x: the matrix with 1x68xn distances
    #' g: the group of the particular matrix
    temple <- c(1, 2, 16, 17)
    brows <- c(18:22, 23:27)
    eyes <- c(37:42, 43:48)
    nose <- c(28:36)
    mouth <- c(49:68)
    jaw <- c(3:15)
    all <- c(1:68)

    tibble(
        Group = g,
        Temple = semantic_dist(x, temple),
        Brows = semantic_dist(x, brows),
        Eyes = semantic_dist(x, eyes),
        Nose = semantic_dist(x, nose),
        Mouth = semantic_dist(x, mouth),
        Jaw = semantic_dist(x, jaw),
        All = semantic_dist(x, all))
        
}

proc <- function(x)
{
    #' perform generalized procruste analysis
    #' x: list of shapes (grouped)
    #' output: 
    
    procGPA(x)$rotated
    
}

mshape <- function(x)
{
    ## compute mean shape for group x (done after proc analysis
    procGPA(x)$mshape
}


