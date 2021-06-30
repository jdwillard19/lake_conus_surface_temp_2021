library(sbtools)

##################################################################################
# (Jared - Sept 2020) - pull all data from sciencebase needed for reproducing paper 
# (note) - in the future could use queries if available instead of hard coding IDs
#################################################################################
cat("Enter ScienceBase username (leave blank if you don't have): ");
un <- readLines("stdin",n=1);
cat("Enter ScienceBase password: ");
pas <- readLines("stdin",n=1);
cat( "\n" )

if (un == '') {

} else {
	authenticate_sb(un,pas)
}



dest_dir = '../../data/raw/data_release/'

item_file_download('60341c3ed34eb12031172aa6',overwrite_file=TRUE,dest_dir=dest_dir,verbose=TRUE)



