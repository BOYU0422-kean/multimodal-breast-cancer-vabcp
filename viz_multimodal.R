# ==========================================================
# viz_multimodal.R — 多模态模型可视化
# Tabular vs Image vs Multimodal AUC
# ==========================================================

library(ggplot2)

visualize_multimodal_auc <- function(tab_auc, img_auc, mm_auc, output_dir="figures/multimodal") {
  
  if (!dir.exists(output_dir)) dir.create(output_dir, recursive = TRUE)
  
  df <- data.frame(
    Model = c("Tabular Only", "Image Only", "Multimodal"),
    AUC = c(tab_auc, img_auc, mm_auc)
  )
  
  p_auc <- ggplot(df, aes(Model, AUC, fill=Model)) +
    geom_col() +
    geom_text(aes(label=round(AUC,3)), vjust=-0.3, size=5) +
    scale_fill_manual(values=c("#1f77b4","#2ca02c","#ff7f0e")) +
    ylim(0,1) +
    labs(title="Model Performance Comparison (AUC)") +
    theme_minimal()
  
  ggsave(file.path(output_dir, "07_multimodal_auc.png"), p_auc, width=7, height=6, dpi=300)
}
