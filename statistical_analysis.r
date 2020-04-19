library('shapes','foreach')
library('dplyr','ggplot2')
library('reshape2','plyr')
library('dplyr')

raw_data <- read.csv("./facial_analysis_data.csv")
raw_data$Group <- as.factor(raw_data$Group)
raw_data$Gender <- as.factor(raw_data$Gender)

set.seed(100)
dplyr::sample_n(raw_data, 10)

# Check that Group and Gender are categorical data
levels(raw_data$Group)
grouped_data <- group_by(raw_data, Group)
summarise(grouped_data, mean=mean(Age), sd=sd(Age))
levels(raw_data$Gender)
count(grouped_data, Gender)

# Plot the group ages
ggplot2::ggplot(raw_data, ggplot2::aes(x=Group, y=Age, color=Group)) + ggplot2::geom_boxplot()

# Compute the analysis of variance on the age of the patients (ANOVA)
res.aov <- aov(Age ~ Group, data = raw_data)
# Summary of the analysis
summary(res.aov)

# Compute the analysis of variance on the gender of the patients (Chi-Square)
chisq.test(raw_data$Group, raw_data$Gender, correct=FALSE)

# lets now load our ladmark data
csv_data_file = './processed-dat//ldmk_coords.csv'
ldmk_data <- read.csv(csv_data_file, header = TRUE)
str(ldmk_data)
head(ldmk_data)

# this function takes as input a patient's group and file number and will fetch the X and Y coordinates of each
# landmark and bind them to our database
bind_to_patient <- function(inpt_db, ldmk_cols){
    matchno <- 0
    tmp_df_list <- list()
    
    for (i in rownames(inpt_db)){
        i <- as.integer(i)
        grp <- inpt_db[i,]$Group
        fln <- inpt_db[i,]$File.No.
        
        # find the matching group and create the appropriate label
        if (grp == 1){
            if (fln > 99){
                pt <- paste('oi_t1.',as.character(inpt_db[i,]$File.No.),sep='')
            }
            if (fln < 100 && fln > 9){
                pt <- paste('oi_t1.0',as.character(inpt_db[i,]$File.No.),sep='')
            }
            if (fln <10){
                pt <- paste('oi_t1.00',as.character(inpt_db[i,]$File.No.),sep='')
            }
        }

        if (grp == 3){
            if (fln > 99){
                pt <- paste('oi_t3.',as.character(inpt_db[i,]$File.No.),sep='')
            }
            if (fln < 100 && fln > 9){
                pt <- paste('oi_t3.0',as.character(inpt_db[i,]$File.No.),sep='')
            }
            if (fln <10){
                pt <- paste('oi_t3.00',as.character(inpt_db[i,]$File.No.),sep='')
            }
        }

        if (grp == 4){
            if (fln > 99){
                pt <- paste('oi_t4.',as.character(inpt_db[i,]$File.No.),sep='')
            }
            if (fln < 100 && fln > 9){
                pt <- paste('oi_t4.0',as.character(inpt_db[i,]$File.No.),sep='')
            }
            if (fln <10){
                pt <- paste('oi_t4.00',as.character(inpt_db[i,]$File.No.),sep='')
            }
        }

        if (as.character(grp) == 'c'){
            pt <- paste('ctl.',as.character(inpt_db[i,]$File.No.),sep='')
        
        }

        # begin by finding matching columns
        prelim_index_matches <- grep(pt, ldmk_cols)
    
        # grep with the ldmk data rownames and then verify to have only the results who are nchars + 2 to
        # avoid partial matches **** ie. 20 in 200
        index_matches <- c()
        for (id in prelim_index_matches){
            if (nchar(ldmk_cols[id]) == (nchar(pt) + 2)){
                index_matches <- c(index_matches, id)
            }
        }
        
        # skip forward if no matches ar found
        if (length(index_matches) == 0){
            print('[ERROR] No matches found')
            print(pt)
            next
        }
        
        # verify matches are accurate and that there are only 2 (x,y) [ie length=1]
        if (length(index_matches) != 2){
            print('[ERROR] Duplicate data in grep function') 
            next
        }

        matchno <- matchno + 1
        # create a 136 row array and bind it to the correct 
        a <- ldmk_data[index_matches[1]]
        b <- ldmk_data[index_matches[2]]
        a <- t(a)
        b <- t(b)
        c <- cbind(a,b)

        # create a new row for our database and append it to our list of dataframes
        new_row <- cbind(inpt_db[i,], c)
        x <- length(tmp_df_list)+1
        tmp_df_list[[x]] <- new_row

    }
    
    # concatenate our dataframes into our final output database
    outpt_db <- do.call("rbind", tmp_df_list)

    return(outpt_db)
}

ldmk_col_names <- colnames(ldmk_data)
full_db <- bind_to_patient(raw_data, ldmk_col_names)





head(full_db)

# create a function to create matrices
make_matrix <- function(group_chr){
    subs_db <- subset(full_db, Group == group_chr)
    x_dat <- subs_db[,6:73]
    x_dat <- t(x_dat)
    y_dat <- subs_db[,74:141]
    y_dat <- t(y_dat)
    list_xy_dat <- list()

    for (i in (1:length(x_dat[1,]))){
        z <- length(list_xy_dat) + 1
        d <- cbind(x_dat[,i], y_dat[,i])
        d <- as.matrix(d)
        list_xy_dat[[z]] <- d
    }

    big_mat <- as.array(list_xy_dat)
    big_mat <- simplify2array(big_mat)
    return(big_mat)
}

oi1_mat <- make_matrix('1')
oi3_mat <- make_matrix('3')
oi4_mat <- make_matrix('4')
ctl_mat <- make_matrix('c')


oi1_gpa <- procGPA(oi1_mat, eigen2d=TRUE)
oi3_gpa <- procGPA(oi3_mat, eigen2d=TRUE)
oi4_gpa <- procGPA(oi4_mat, eigen2d=TRUE)
ctl_gpa <- procGPA(ctl_mat, eigen2d=TRUE)
list_grp_mat <- list(oi1_mat, oi3_mat, oi4_mat)

str(ctl_gpa$rotated)

oi1_rmat <- matrix(oi1_gpa$rotated, 136, 88)
oi1_rmat <- t(oi1_rmat)
oi3_rmat <- matrix(oi3_gpa$rotated, 136, 28)
oi3_rmat <- t(oi3_rmat)
oi4_rmat <- matrix(oi4_gpa$rotated, 136, 57)
oi4_rmat <- t(oi4_rmat)
ctl_rmat <- matrix(ctl_gpa$rotated, 136, 133)
ctl_rmat <- t(ctl_rmat)
full_rmat <- rbind(oi1_rmat, oi3_rmat, oi4_rmat, ctl_rmat)
ml_db <- cbind(raw_data, full_rmat)
write.csv(ml_db, oi_ml_db.csv)

# computes landmark distance from a predetermined baseline
compute_med <- function(input_mat){
    # we set our baseline shape to the mean shape of our control group
    base_shape <- ctl_gpa$mshape
    base_shape <- as.matrix(base_shape)
    len <- (1:(length(input_mat[1,,])/2))
    ed_vec <- c()
    
    # iterate through the matrices in the input
    for (i in len){
        mat <- as.matrix(input_mat[,,i])
        
        # let's gather our variables
        X1 <- mat[,1]
        X2 <- base_shape[,1]
        Y1 <- mat[,2]
        Y2 <- base_shape[,2]

        # simple euclidean arithmetic
        ed <-  sqrt((X2 - X1)**2 + (Y2-Y1)**2)
        ed_vec <- c(ed_vec, ed)
    }
    return(ed_vec)
}

# list of mean landmark distance per group
oi1_ed <- compute_med(oi1_gpa$rotated)
oi3_ed <- compute_med(oi3_gpa$rotated)
oi4_ed <- compute_med(oi4_gpa$rotated)

oi1_ed <- melt(data.frame(oi1_ed))
oi3_ed <- melt(data.frame(oi3_ed))
oi4_ed <- melt(data.frame(oi4_ed))
ed_df <- rbind(oi1_ed, oi3_ed, oi4_ed)

str(ed_df)
head(ed_df)

# lets visualize the euclidean distance per landmark group means

oi1_med <- compute_med(oi1_gpa$rotated)
oi3_med <- compute_med(oi3_gpa$rotated)
oi4_med <- compute_med(oi4_gpa$rotated)

oi1_med <- matrix(oi1_med, nrow=68)
oi3_med <- matrix(oi3_med, nrow=68)
oi4_med <- matrix(oi4_med, nrow=68)

oi1_med <- rowMeans(oi1_med)
oi3_med <- rowMeans(oi3_med)
oi4_med <- rowMeans(oi4_med)

med_df <- data.frame(oi1_med, oi3_med, oi4_med)
med_coln <- c('Type 1', 'Type 3', 'Type 4')
colnames(med_df) <- med_coln

png(filename="histo.png", res=720, width=16, height=9, units="in", pointsize=6)

barplot(
  as.matrix(t(med_df)),
  main='Mean Euclidean Distance of Each Landmark',
  names.arg=c(1:68),
  ylab='Distance',
  xlab='Landmark',
  beside=TRUE,
  col=colours,
  legend=colnames(med_df)
)
dev.off()

zscore <- (ed_df$value - mean(ed_df$value))/sd(ed_df$value)
ed_df <- cbind(ed_df, zscore)
head(ed_df)

compute_ratio <- function(inpt_mat, pt1, pt2, pt3, pt4){
    
    
    # first we gather our coordinates
    X1 <- inpt_mat[pt1,1,]
    X2 <- inpt_mat[pt2,1,]
    X3 <- inpt_mat[pt3,1,]
    X4 <- inpt_mat[pt4,1,]
    
    Y1 <- inpt_mat[pt1,2,]
    Y2 <- inpt_mat[pt2,2,]
    Y3 <- inpt_mat[pt3,2,]
    Y4 <- inpt_mat[pt4,2,]
    
    # we compute the ratio
    ed12 <- sqrt((X2 - X1)**2 + (Y2-Y1)**2)
    ed34 <- sqrt((X3 - X4)**2 + (Y3-Y4)**2)
    ratio_vec <- ed12 / ed34
    
    return(ratio_vec)
}



# ratio 1: (biocular 37:46 / bitemporal 1:17)
oi1_ratio1 <- compute_ratio(oi1_mat, 37, 46, 1, 17)
oi3_ratio1 <- compute_ratio(oi3_mat, 37, 46, 1, 17)
oi4_ratio1 <- compute_ratio(oi4_mat, 37, 46, 1, 17)
ctl_ratio1 <- compute_ratio(ctl_mat, 37, 46, 1, 17)

oi1_ratio1 <- melt(data.frame(oi1_ratio1))
oi3_ratio1 <- melt(data.frame(oi3_ratio1))
oi4_ratio1 <- melt(data.frame(oi4_ratio1))
ctl_ratio1 <- melt(data.frame(ctl_ratio1))
ratio1_df <- rbind(ctl_ratio1, oi1_ratio1, oi3_ratio1, oi4_ratio1)
head(ratio1_df)

# ratio 2: (bimandibular 5:13 / bitemporal 1:17)
oi1_ratio2 <- compute_ratio(oi1_mat, 5, 13, 1, 17)
oi3_ratio2 <- compute_ratio(oi3_mat, 5, 13, 1, 17)
oi4_ratio2 <- compute_ratio(oi4_mat, 5, 13, 1, 17)
ctl_ratio2 <- compute_ratio(ctl_mat, 5, 13, 1, 17)

oi1_ratio2 <- melt(data.frame(oi1_ratio2))
oi3_ratio2 <- melt(data.frame(oi3_ratio2))
oi4_ratio2 <- melt(data.frame(oi4_ratio2))
ctl_ratio2 <- melt(data.frame(ctl_ratio2))
ratio2_df <- rbind(ctl_ratio2, oi1_ratio2, oi3_ratio2, oi4_ratio2)
head(ratio2_df)

# ratio 3: (bitemporal 1:17 / face height 28:9)
oi1_ratio3 <- compute_ratio(oi1_mat, 1, 17, 28, 9)
oi3_ratio3 <- compute_ratio(oi3_mat, 1, 17, 28, 9)
oi4_ratio3 <- compute_ratio(oi4_mat, 1, 17, 28, 9)
ctl_ratio3 <- compute_ratio(ctl_mat, 1, 17, 28, 9)

oi1_ratio3 <- melt(data.frame(oi1_ratio3))
oi3_ratio3 <- melt(data.frame(oi3_ratio3))
oi4_ratio3 <- melt(data.frame(oi4_ratio3))
ctl_ratio3 <- melt(data.frame(ctl_ratio3))
ratio3_df <- rbind(ctl_ratio3, oi1_ratio3, oi3_ratio3, oi4_ratio3)
head(ratio3_df)

# lfh: (lower face 34-9 / face height 28:9)
oi1_lfh <- compute_ratio(oi1_mat, 34, 9, 28, 9)
oi3_lfh <- compute_ratio(oi3_mat, 34, 9, 28, 9)
oi4_lfh <- compute_ratio(oi4_mat, 34, 9, 28, 9)
ctl_lfh <- compute_ratio(ctl_mat, 34, 9, 28, 9)

oi1_lfh <- melt(data.frame(oi1_lfh))
oi3_lfh <- melt(data.frame(oi3_lfh))
oi4_lfh <- melt(data.frame(oi4_lfh))
ctl_lfh <- melt(data.frame(ctl_lfh))
lfh_df <- rbind(ctl_lfh, oi1_lfh, oi3_lfh, oi4_lfh)
head(lfh_df)

# z-score ratio 1
zscore <- (ratio1_df$value - mean(ratio1_df$value))/sd(ratio1_df$value)
ratio1_df <- cbind(ratio1_df, zscore)
head(ratio1_df)

# z-score ratio 2
zscore <- (ratio2_df$value - mean(ratio2_df$value))/sd(ratio2_df$value)
ratio2_df <- cbind(ratio2_df, zscore)
head(ratio2_df)

# z-score ratio 3
zscore <- (ratio3_df$value - mean(ratio3_df$value))/sd(ratio3_df$value)
ratio3_df <- cbind(ratio3_df, zscore)
head(ratio3_df)

# z-score lfh
zscore <- (lfh_df$value - mean(lfh_df$value))/sd(lfh_df$value)
lfh_df <- cbind(lfh_df, zscore)
head(lfh_df)

res_ctl_oi1 <- testmeanshapes(ctl_mat, oi1_mat, resamples=400, replace=TRUE)
res_ctl_oi3 <- testmeanshapes(ctl_mat, oi3_mat, resamples=400, replace=TRUE)
res_ctl_oi4 <- testmeanshapes(ctl_mat, oi4_mat, resamples=400, replace=TRUE)
res_oi1_oi3 <- testmeanshapes(oi1_mat, oi3_mat, resamples=400, replace=TRUE)
res_oi1_oi4 <- testmeanshapes(oi1_mat, oi4_mat, resamples=400, replace=TRUE)
res_oi3_oi4 <- testmeanshapes(oi3_mat, oi4_mat, resamples=400, replace=TRUE)
print('[DONE]')

print(paste('Control - Type 1: ', as.character(res_ctl_oi1$G.pvalue))) 
print(paste('Control - Type 3: ', as.character(res_ctl_oi3$G.pvalue))) 
print(paste('Control - Type 4: ', as.character(res_ctl_oi4$G.pvalue))) 
print(paste('Type 1 - Type 3: ', as.character(res_oi1_oi3$G.pvalue))) 
print(paste('Type 1 - Type 4: ', as.character(res_oi1_oi4$G.pvalue))) 
print(paste('Type 3 - Type 4: ', as.character(res_oi3_oi4$G.pvalue))) 

aov.med.res <- aov(value ~ variable, data = ed_df)
print(paste('Type 1: ', as.character(mean(oi1_ed$value)), '+/-', as.character(sd(oi1_ed$value))))
print(paste('Type 3: ', as.character(mean(oi3_ed$value)), '+/-', as.character(sd(oi3_ed$value))))
print(paste('Type 4: ', as.character(mean(oi4_ed$value)), '+/-', as.character(sd(oi4_ed$value))))
summary(aov.med.res)
pairwise.t.test(ed_df$value, ed_df$variable, p.adj = "bonf")

aov.medz.res <- aov(zscore ~ variable, data = ed_df)
print(paste('Type 1: ', as.character(mean(ed_df[which(ed_df$variable=='oi1_ed'),]$zscore)), '+/-', as.character(sd(ed_df[which(ed_df$variable=='oi1_ed'),]$zscore))))
print(paste('Type 3: ', as.character(mean(ed_df[which(ed_df$variable=='oi3_ed'),]$zscore)), '+/-', as.character(sd(ed_df[which(ed_df$variable=='oi3_ed'),]$zscore))))
print(paste('Type 4: ', as.character(mean(ed_df[which(ed_df$variable=='oi4_ed'),]$zscore)), '+/-', as.character(sd(ed_df[which(ed_df$variable=='oi4_ed'),]$zscore))))
summary(aov.medz.res)
pairwise.t.test(ed_df$zscore, ed_df$variable, p.adj = "bonf")

aov.ratio1.res <- aov(value ~ variable, data = ratio1_df) 
print(paste('Control: ', as.character(mean(ctl_ratio1$value)), '+/-', as.character(sd(ctl_ratio1$value))))
print(paste('Type 1: ', as.character(mean(oi1_ratio1$value)), '+/-', as.character(sd(oi1_ratio1$value))))
print(paste('Type 3: ', as.character(mean(oi3_ratio1$value)), '+/-', as.character(sd(oi3_ratio1$value))))
print(paste('Type 4: ', as.character(mean(oi4_ratio1$value)), '+/-', as.character(sd(oi4_ratio1$value))))
summary(aov.ratio1.res)
pairwise.t.test(ratio1_df$value, ratio1_df$variable, p.adj = "bonf")

aov.ratio1z.res <- aov(zscore ~ variable, data = ratio1_df)
print(paste('Control: ', as.character(mean(ratio1_df[which(ratio1_df$variable=='ctl_ratio1'),]$zscore)), '+/-', as.character(sd(ratio1_df[which(ratio1_df$variable=='ctl_ratio1'),]$zscore))))
print(paste('Type 1: ', as.character(mean(ratio1_df[which(ratio1_df$variable=='oi1_ratio1'),]$zscore)), '+/-', as.character(sd(ratio1_df[which(ratio1_df$variable=='oi1_ratio1'),]$zscore))))
print(paste('Type 3: ', as.character(mean(ratio1_df[which(ratio1_df$variable=='oi3_ratio1'),]$zscore)), '+/-', as.character(sd(ratio1_df[which(ratio1_df$variable=='oi3_ratio1'),]$zscore))))
print(paste('Type 4: ', as.character(mean(ratio1_df[which(ratio1_df$variable=='oi4_ratio1'),]$zscore)), '+/-', as.character(sd(ratio1_df[which(ratio1_df$variable=='oi4_ratio1'),]$zscore))))
summary(aov.ratio1z.res)
pairwise.t.test(ratio1_df$zscore, ratio1_df$variable, p.adj = "bonf")

aov.ratio2.res <- aov(value ~ variable, data = ratio2_df) 
print(paste('Control: ', as.character(mean(ctl_ratio2$value)), '+/-', as.character(sd(ctl_ratio2$value))))
print(paste('Type 1: ', as.character(mean(oi1_ratio2$value)), '+/-', as.character(sd(oi1_ratio2$value))))
print(paste('Type 3: ', as.character(mean(oi3_ratio2$value)), '+/-', as.character(sd(oi3_ratio2$value))))
print(paste('Type 4: ', as.character(mean(oi4_ratio2$value)), '+/-', as.character(sd(oi4_ratio2$value))))
summary(aov.ratio2.res)
pairwise.t.test(ratio2_df$value, ratio2_df$variable, p.adj = "bonf")

aov.ratio2z.res <- aov(zscore ~ variable, data = ratio2_df)
print(paste('Control: ', as.character(mean(ratio2_df[which(ratio2_df$variable=='ctl_ratio2'),]$zscore)), '+/-', as.character(sd(ratio2_df[which(ratio2_df$variable=='ctl_ratio2'),]$zscore))))
print(paste('Type 1: ', as.character(mean(ratio2_df[which(ratio2_df$variable=='oi1_ratio2'),]$zscore)), '+/-', as.character(sd(ratio2_df[which(ratio2_df$variable=='oi1_ratio2'),]$zscore))))
print(paste('Type 3: ', as.character(mean(ratio2_df[which(ratio2_df$variable=='oi3_ratio2'),]$zscore)), '+/-', as.character(sd(ratio2_df[which(ratio2_df$variable=='oi3_ratio2'),]$zscore))))
print(paste('Type 4: ', as.character(mean(ratio2_df[which(ratio2_df$variable=='oi4_ratio2'),]$zscore)), '+/-', as.character(sd(ratio2_df[which(ratio2_df$variable=='oi4_ratio2'),]$zscore))))
summary(aov.ratio2z.res)
pairwise.t.test(ratio2_df$zscore, ratio2_df$variable, p.adj = "bonf")

aov.ratio3.res <- aov(value ~ variable, data = ratio3_df)
print(paste('Control: ', as.character(mean(ctl_ratio3$value)), '+/-', as.character(sd(ctl_ratio3$value))))
print(paste('Type 1: ', as.character(mean(oi1_ratio3$value)), '+/-', as.character(sd(oi1_ratio3$value))))
print(paste('Type 3: ', as.character(mean(oi3_ratio3$value)), '+/-', as.character(sd(oi3_ratio3$value))))
print(paste('Type 4: ', as.character(mean(oi4_ratio3$value)), '+/-', as.character(sd(oi4_ratio3$value))))
summary(aov.ratio3.res)
pairwise.t.test(ratio3_df$value, ratio3_df$variable, p.adj = "bonf")

aov.ratio3z.res <- aov(zscore ~ variable, data = ratio3_df)
print(paste('Control: ', as.character(mean(ratio3_df[which(ratio3_df$variable=='ctl_ratio3'),]$zscore)), '+/-', as.character(sd(ratio3_df[which(ratio3_df$variable=='ctl_ratio3'),]$zscore))))
print(paste('Type 1: ', as.character(mean(ratio3_df[which(ratio3_df$variable=='oi1_ratio3'),]$zscore)), '+/-', as.character(sd(ratio3_df[which(ratio3_df$variable=='oi1_ratio3'),]$zscore))))
print(paste('Type 3: ', as.character(mean(ratio3_df[which(ratio3_df$variable=='oi3_ratio3'),]$zscore)), '+/-', as.character(sd(ratio3_df[which(ratio3_df$variable=='oi3_ratio3'),]$zscore))))
print(paste('Type 4: ', as.character(mean(ratio3_df[which(ratio3_df$variable=='oi4_ratio3'),]$zscore)), '+/-', as.character(sd(ratio3_df[which(ratio3_df$variable=='oi4_ratio3'),]$zscore))))
summary(aov.ratio3z.res)
pairwise.t.test(ratio3_df$zscore, ratio3_df$variable, p.adj = "bonf")

aov.lfh.res <- aov(value ~ variable, data = lfh_df)
print(paste('Control: ', as.character(mean(ctl_lfh$value)), '+/-', as.character(sd(ctl_lfh$value))))
print(paste('Type 1: ', as.character(mean(oi1_lfh$value)), '+/-', as.character(sd(oi1_lfh$value))))
print(paste('Type 3: ', as.character(mean(oi3_lfh$value)), '+/-', as.character(sd(oi3_lfh$value))))
print(paste('Type 4: ', as.character(mean(oi4_lfh$value)), '+/-', as.character(sd(oi4_lfh$value))))
summary(aov.lfh.res)

aov.lfhz.res <- aov(zscore ~ variable, data = lfh_df)
print(paste('Control: ', as.character(mean(lfh_df[which(lfh_df$variable=='ctl_lfh'),]$zscore)), '+/-', as.character(sd(lfh_df[which(lfh_df$variable=='ctl_lfh'),]$zscore))))
print(paste('Type 1: ', as.character(mean(lfh_df[which(lfh_df$variable=='oi1_lfh'),]$zscore)), '+/-', as.character(sd(lfh_df[which(lfh_df$variable=='oi1_lfh'),]$zscore))))
print(paste('Type 3: ', as.character(mean(lfh_df[which(lfh_df$variable=='oi3_lfh'),]$zscore)), '+/-', as.character(sd(lfh_df[which(lfh_df$variable=='oi3_lfh'),]$zscore))))
print(paste('Type 4: ', as.character(mean(lfh_df[which(lfh_df$variable=='oi4_lfh'),]$zscore)), '+/-', as.character(sd(lfh_df[which(lfh_df$variable=='oi4_lfh'),]$zscore))))
summary(aov.lfhz.res)


