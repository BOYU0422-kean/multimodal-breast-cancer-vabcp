# ==========================================================
# viz_tabular.R — 结构化数据可视化
# Heatmap, PCA, Feature Importance, Pairplot
# ==========================================================

library(ggplot2)
library(ggcorrplot)
library(GGally)
library(ranger)
library(dplyr)
library(scales)

visualize_tabular <- function(df, num_cols, output_dir = "figures/tabular") {
  
  if (!dir.exists(output_dir)) dir.create(output_dir, recursive = TRUE)
  
  # ===== 1. Heatmap =====
  y_numeric <- as.numeric(df$diagnosis == "M")
  corr_mat <- cor(cbind(label = y_numeric, df[, num_cols]), use = "pairwise.complete.obs")
  
  p_corr <- ggcorrplot(
    corr_mat, type = "lower", lab = TRUE,
    lab_size = 2.5, outline.col = "white",
    title = "Feature Correlation Heatmap"
  )
  ggsave(file.path(output_dir, "01_heatmap.png"), p_corr, width = 10, height = 8, dpi = 300)
  
  # ===== 2. PCA =====
  X <- scale(df[, num_cols])
  pca <- prcomp(X, center = FALSE, scale. = FALSE)
  scores <- as.data.frame(pca$x[, 1:2])
  scores$diagnosis <- df$diagnosis
  
  explained <- (pca$sdev^2) / sum(pca$sdev^2)
  
  p_pca <- ggplot(scores, aes(PC1, PC2, color = diagnosis)) +
    geom_point(alpha = 0.8) +
    stat_ellipse(level = 0.95) +
    scale_color_manual(values = c("B"="#1f77b4","M"="#ff7f0e")) +
    labs(
      title="PCA Scatterplot",
      x = paste0("PC1 (", percent(explained[1]), ")"),
      y = paste0("PC2 (", percent(explained[2]), ")")
    ) +
    theme_minimal()
  
  ggsave(file.path(output_dir, "02_pca.png"), p_pca, width=7, height=6, dpi=300)
  
  # ===== 3. Feature Importance (RF) =====
  rf <- ranger(
    diagnosis ~ ., 
    data = df[, c("diagnosis", num_cols)],
    importance = "impurity",
    num.trees = 500,
    seed = 0
  )
  
  imp <- sort(rf$variable.importance, decreasing = TRUE)
  imp_df <- data.frame(feature = names(imp), importance = imp)
  
  p_imp <- ggplot(imp_df[1:15,], aes(x=reorder(feature, importance), y=importance)) +
    geom_col(fill="#ff7f0e") +
    coord_flip() +
    labs(title="Top 15 Feature Importance", x="Feature", y="Importance") +
    theme_minimal()
  
  ggsave(file.path(output_dir, "03_feature_importance.png"), p_imp, width=7, height=6, dpi=300)
  
  # ===== 4. Pairplot =====
  top4 <- imp_df$feature[1:4]
  pair_df <- df[, c("diagnosis", top4)]
  pair_df$diagnosis <- factor(ifelse(pair_df$diagnosis=="M","Malignant","Benign"))
  
  p_pair <- GGally::ggpairs(
    pair_df,
    aes(color = diagnosis, alpha=0.7),
    upper=list(continuous="points"),
    lower=list(continuous="smooth_loess"),
    diag=list(continuous="densityDiag")
  ) +
    scale_color_manual(values=c("Benign"="#1f77b4","Malignant"="#ff7f0e"))
  
  ggsave(file.path(output_dir, "04_pairplot.png"), p_pair, width=10, height=8, dpi=200)
  
}
