load_breakhis <- function(base_dir, img_size = 150, n_per_class = 100) {
  classes <- c("benign", "malignant")
  all_images <- list()
  all_labels <- c()
  
  for (cls in classes) {
    img_files <- list.files(
      file.path(base_dir, cls),
      pattern = "\\.(png|jpg|jpeg)$",
      recursive = TRUE,
      full.names = TRUE
    )
    
    selected <- sample(img_files, min(n_per_class, length(img_files)))
    
    for (f in selected) {
      img <- EBImage::readImage(f)
      img <- EBImage::resize(img, img_size, img_size)
      arr <- as.array(img)
      
      if (length(dim(arr)) == 2) {
        arr <- abind::abind(arr, arr, arr, along = 3)
      }
      
      all_images[[length(all_images) + 1]] <- arr
      all_labels <- c(all_labels, ifelse(cls == "malignant", 1, 0))
    }
  }
  
  X <- abind::abind(all_images, along = 1)
  if (max(X) > 1) X <- X / 255
  
  return(list(X = X, y = all_labels))
}
