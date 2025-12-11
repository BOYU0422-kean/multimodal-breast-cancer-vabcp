# ==========================================================
# viz_models.R — 模型可视化
# Confusion Matrix, ROC Curves
# ==========================================================

library(ggplot2)
library(caret)
library(pROC)
library(scales)
library(dplyr)

visualize_models <- function(pred, truth, roc_list, output_dir="figures/tabular") {
  
  if (!dir.exists(output_dir)) dir.create(output_dir, recursive = TRUE)
  
  # ===== 1. Confusion Matrix =====
  cm <- caret::confusionMatrix(pred, truth, positive="M")
  cm_df <- as.data.frame(cm$table)
  colnames(cm_df) <- c("Predicted","Actual","Freq")
  
  cm_df <- cm_df %>%
    group_by(Actual) %>%
    mutate(RowPct = Freq / sum(Freq)) %>%
    ungroup() %>%
    mutate(Label = paste0(Freq, "\n", percent(RowPct)))
  
  p_cm <- ggplot(cm_df, aes(Predicted, Actual, fill=Freq)) +
    geom_tile() +
    geom_text(aes(label=Label), color="white", size=5) +
    scale_fill_gradient(low="white", high="#1f77b4") +
    labs(title="Confusion Matrix", x="Predicted", y="Actual") +
    theme_minimal()
  
  ggsave(file.path(output_dir, "05_confusion_matrix.png"), p_cm, width=7, height=6, dpi=300)
  
  # ===== 2. Multi-model ROC =====
  p_roc <- pROC::ggroc(roc_list, size=1.0) +
    geom_abline(slope=1, intercept=0, linetype=2, color="gray60") +
    labs(title="Multi-Model ROC Curves") +
    theme_minimal()
  
  ggsave(file.path(output_dir, "06_multi_roc.png"), p_roc, width=7, height=6, dpi=300)
}
