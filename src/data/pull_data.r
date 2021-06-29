library(sbtools)

##################################################################################
# (Jared - Sept 2020) - pull all data needed for MTL paper from sciencebase
# (note) - in the future should use queries if available instead of hard coding IDs
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


item_file_download('60341c3ed34eb12031172aa6',overwrite_file=TRUE,dest_dir=dest_dir)


# # model inputs 
# item_file_download('60341c3ed34eb12031172aa6?filePath=__disk__19%2Fda%2Fe8%2F19dae88214856c6126765dc74e8f8fd1aa936bc9',overwrite_file=TRUE,dest_dir=dest_dir)

# #lake metadata (updated 06/29/21)
# item_file_download('60341c3ed34eb12031172aa6?f=__disk__40%2F94%2Fd6%2F4094d62c7aac6c03e12b713398f4bbc2807fe0ad',overwrite_file=TRUE,dest_dir=dest_dir)


# #temperature obs (updated 06/25/21)
# item_file_download('60341c3ed34eb12031172aa6?f=__disk__64%2Fba%2Fa3%2F64baa3d97c8c89eed62b3c162ddd04ea770a4cc9',overwrite_file=TRUE,dest_dir=dest_dir)



