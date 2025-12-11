# ==========================================================
# viz_crossmodal.R — 跨模态可解释性可视化
# 图 10–12
# ==========================================================

visualize_crossmodal <- function(output_dir="figures/cross_modal") {
  
  if (!dir.exists(output_dir)) dir.create(output_dir, recursive = TRUE)
  
  png(file.path(output_dir, "08_crossmodal_alignment.png"), width=900, height=700)
  
  par(mfrow=c(2,1), mar=c(4,4,3,2))
  
  # 上部分：特征 ↔ 图像概念
  plot(1, type="n", xlim=c(0,10), ylim=c(0,10), axes=FALSE, xlab="", ylab="", main="Cross-Modal Concept Alignment")
  text(5,9,"Feature ↔ Image Patterns", cex=1.5, font=2)
  text(5,7,"• Radius ↔ Tumor Boundary", cex=1.2)
  text(5,6,"• Concavity ↔ Nuclear Atypia", cex=1.2)
  text(5,5,"• Texture ↔ Tissue Structure", cex=1.2)
  text(5,4,"• Area ↔ Lesion Size", cex=1.2)
  
  # 下部分：一致性 vs 不一致性
  plot(1, type="n", xlim=c(0,10), ylim=c(0,10), axes=FALSE, xlab="", ylab="", main="Consistent vs Inconsistent Cases")
  text(5,7,"Consistent Case → SHAP aligns with Grad-CAM", cex=1.2)
  text(5,6,"Inconsistent Case → Highlights diagnostic uncertainty", cex=1.2)
  
  dev.off()
}
