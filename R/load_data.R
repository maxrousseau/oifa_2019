#' read data from csv
#' format into single tibble
#' save to rds
#'
#' Author: Maxime Rousseau
#' Date: 24/06/2020

library(readr)
library(here)
library(tibble)
library(dplyr)

load_data <- function()
{
    rds_save(
        reformat())
}

setpath <- function()
{
    list.files(here("data"), full.names = TRUE)
}

to_tibble <- function(paths)
{
    map(paths, read_csv)
}

reformat <- function()
{
    raw_df <- to_tibble(setpath())
    meta_raw <- raw_df[[1]]
    ldmk_raw <- raw_df[[2]]

    meta_info <- meta_raw %>%
        rename(Id = "File No.") %>%
        select(Group, Id, Age, Gender)

    x_names_vec <- names(select(ldmk_raw, contains("X-")))

    ldmk_tbl <- make_row(colname = x_names_vec[1], raw_df = ldmk_raw)

    for (colname in x_names_vec[-1]){
        ldmk_tbl <- bind_rows(
            ldmk_tbl,
            make_row(colname = colname, raw_df = ldmk_raw))

    }

    right_join(meta_info, ldmk_tbl)
}

make_row <- function(colname, raw_df){

    ycolname <- get_ycolname(colname)
    Group <- get_group(colname)
    Id <- get_id(colname)

    ## create labels
    x_label <- str_c(as.character(c(1:68)), "x")
    y_label <- str_c(as.character(c(1:68)), "y")
    ldmk_label <- c(x_label, y_label)

    x_tbl <- tibble(x_label, raw_df[, colname]) %>%
        pivot_wider(names_from = x_label, values_from = colname)
    y_tbl <- tibble(y_label, raw_df[, ycolname]) %>%
        pivot_wider(names_from = y_label, values_from = ycolname)

    info_tbl <- tibble(Group, Id)
    bind_cols(info_tbl, x_tbl, y_tbl)
}

get_group <- function(varname){
    grp_str <- str_split(varname, "-")[[1]][2]
    switch(
        grp_str,
        "ctl" = "c",
        "oi_t1" = "1",
        "oi_t3" = "3",
        "oi_t4" = "4")
}

get_id <- function(varname){
    as.numeric(str_split(varname, "-")[[1]][3])
}

get_ycolname <- function(varname){
    str_replace(varname, "X", "Y")
}

rds_save <- function(table)
{
   ## saves the tibble to rda for other files
   saveRDS(table, file=here("data/oifaces.rds")) 
}
