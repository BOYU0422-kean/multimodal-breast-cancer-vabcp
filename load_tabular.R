prepare_df <- function(raw) {
  df <- raw
  if (!"diagnosis" %in% tolower(names(df))) {
    dn <- names(df)[grepl("^diag", tolower(names(df)))]
    if (length(dn) == 1) names(df)[names(df) == dn] <- "diagnosis"
  }
  names(df) <- make.names(names(df), unique = TRUE)
  
  df$diagnosis <- toupper(df$diagnosis)
  df$diagnosis <- factor(df$diagnosis, levels = c("B","M"))
  
  id_cols <- names(df)[grepl("id|patient", tolower(names(df)))]
  num_cols <- setdiff(names(df), c(id_cols, "diagnosis"))
  df[num_cols] <- lapply(df[num_cols], function(x) as.numeric(as.character(x)))
  
  return(list(df = df, num_cols = num_cols))
}
