# ======================================================
# UCI 乳腺癌 · 可视化主线（遵循老师建议，自动导出6图 · 颜色统一版）
# 01_Heatmap.png
# 02_Confusion.png
# 03_ROC.png
# 04_PCA.png
# 05_FeatureImportance.png
# 06_Pairplot.png
# ======================================================

# ---- Packages ----
pkgs <- c("e1071","caret","ggplot2","pROC","ggcorrplot",
          "dplyr","tidyr","forcats","scales","GGally","ranger")
new_pkgs <- setdiff(pkgs, rownames(installed.packages()))
if (length(new_pkgs)) install.packages(new_pkgs, dependencies = TRUE)
invisible(lapply(pkgs, library, character.only = TRUE))

set.seed(0)
dir.create("plots", showWarnings = FALSE)

# ===== 全局颜色设定（教授要求：良性=蓝，恶性=橙） =====
color_map <- c(
  "Benign"    = "#1f77b4",
  "B"         = "#1f77b4",
  "Malignant" = "#ff7f0e",
  "M"         = "#ff7f0e"
)

# ---- Load data ----
csv_path <- "C:/Users/RBY/Desktop/bc_data(1).csv"  # ← 改成你的路径
raw <- read.csv(csv_path, header = TRUE, stringsAsFactors = FALSE, check.names = FALSE)


# ---- 诊断列识别 & 清洗 ----
df <- raw
# 自动识别 diagnosis 列
if (!"diagnosis" %in% tolower(names(df))) {
  dn <- names(df)[grepl("^diag", tolower(names(df)))]
  if (length(dn) == 1) names(df)[names(df) == dn] <- "diagnosis"
}
names(df) <- make.names(names(df), unique = TRUE)
stopifnot("diagnosis" %in% names(df))

# 统一标签
df$diagnosis <- toupper(df$diagnosis)
df$diagnosis <- factor(df$diagnosis, levels = c("B","M"))

# 自动识别数值特征
id_cols  <- names(df)[grepl("^id$|patient|^X\\.?.*id$", tolower(names(df)))]
num_cols <- setdiff(names(df), c(id_cols, "diagnosis"))
df[num_cols] <- lapply(df[num_cols], function(x) as.numeric(as.character(x)))

# =========================
# 1) Heatmap
# =========================
y_numeric <- as.numeric(df$diagnosis == "M")
corr_mat <- cor(cbind(y_numeric = y_numeric, df[, num_cols]),
                use = "pairwise.complete.obs")
p_heat <- ggcorrplot(corr_mat, type = "lower", lab = TRUE, lab_size = 2.5,
                     outline.col = "white", title = "Feature Correlation (incl. label)")
ggsave("plots/01_Heatmap.png", p_heat, width = 10, height = 8, dpi = 300)

# =========================
# 2) Naive Bayes + Confusion Matrix
# =========================
pref_feats <- c("radius_worst","perimeter_worst","area_worst","concave_points_worst")
have_pref  <- pref_feats %in% num_cols
if (!all(have_pref)) {
  cors <- cor(df[, num_cols], y_numeric, use = "pairwise.complete.obs")
  nb_feats <- names(sort(abs(as.vector(cors)), decreasing = TRUE))[1:4]
} else {
  nb_feats <- pref_feats
}

set.seed(0)
idx <- caret::createDataPartition(df$diagnosis, p = 0.7, list = FALSE)
train <- df[idx, ]
test  <- df[-idx, ]

x_tr_raw <- train[, nb_feats, drop=FALSE]
x_te_raw <- test[,  nb_feats, drop=FALSE]
y_tr <- train$diagnosis
y_te <- test$diagnosis

# 预处理
pp  <- preProcess(x_tr_raw, method = c("medianImpute","center","scale"))
x_tr <- predict(pp, x_tr_raw)
x_te <- predict(pp, x_te_raw)

# 训练 NB
nb_fit <- e1071::naiveBayes(x = x_tr, y = y_tr)
prob_M <- predict(nb_fit, x_te, type = "raw")[, "M"]
pred_cls <- factor(ifelse(prob_M >= 0.5, "M", "B"), levels = c("B","M"))

cm <- caret::confusionMatrix(pred_cls, y_te, positive = "M")
cm_df <- as.data.frame(cm$table)
colnames(cm_df) <- c("Predicted","Actual","Freq")
cm_vis <- cm_df %>%
  group_by(Actual) %>% mutate(RowPct = Freq / sum(Freq)) %>% ungroup() %>%
  mutate(Label = paste0(Freq, "\n", percent(RowPct, accuracy = 0.1)))

# Confusion Matrix 颜色不使用蓝/橙，用蓝色渐变保持视觉干净
p_cm <- ggplot(cm_vis, aes(Predicted, Actual, fill = Freq)) +
  geom_tile() +
  geom_text(aes(label = Label), color = "white", size = 5, lineheight = 0.9) +
  scale_fill_gradient(low = "white", high = "#1f77b4") +
  coord_equal() +
  labs(title = "Confusion Matrix (Counts + Row %)", x = "Predicted", y = "Actual") +
  theme_minimal(base_size = 13) +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))
ggsave("plots/02_Confusion.png", p_cm, width = 7.2, height = 6.0, dpi = 300)

# =========================
# 3) ROC (AUC)
# =========================
roc_nb <- pROC::roc(response = y_te, predictor = prob_M, levels = c("B","M"), direction = "<")
auc_nb <- as.numeric(pROC::auc(roc_nb))
p_roc <- pROC::ggroc(roc_nb, colour = "#ff7f0e", size = 1.1) +
  geom_abline(slope = 1, intercept = 0, linetype = 2, color = "gray50") +
  ggtitle(sprintf("ROC — Naive Bayes (AUC = %.3f)", auc_nb)) +
  theme_minimal(base_size = 13)
ggsave("plots/03_ROC.png", p_roc, width = 6.5, height = 5.5, dpi = 300)

# =========================
# 4) PCA Scatterplot
# =========================
X_all <- scale(df[, num_cols])
pca   <- prcomp(X_all, center = FALSE, scale. = FALSE)
scores <- as.data.frame(pca$x[, 1:2]); colnames(scores) <- c("PC1","PC2")
scores$diagnosis <- df$diagnosis
explained <- (pca$sdev^2) / sum(pca$sdev^2)
pc1_var <- scales::percent(explained[1], accuracy = 0.1)
pc2_var <- scales::percent(explained[2], accuracy = 0.1)

p_pca <- ggplot(scores, aes(PC1, PC2, color = diagnosis)) +
  geom_point(alpha = 0.85, size = 2) +
  stat_ellipse(level = 0.95, linetype = 2, linewidth = 0.7) +
  scale_color_manual(values = color_map) +
  labs(title = "PCA Scatterplot (Standardized Features)",
       x = paste0("PC1 (", pc1_var, ")"),
       y = paste0("PC2 (", pc2_var, ")"),
       color = "Diagnosis") +
  theme_minimal(base_size = 13)
ggsave("plots/04_PCA.png", p_pca, width = 7.2, height = 6.0, dpi = 300)

# =========================
# 5) Feature Importance
# =========================
rf_train <- data.frame(diagnosis = y_tr, x_tr)
rf_fit <- ranger::ranger(
  diagnosis ~ ., data = rf_train,
  importance = "impurity", probability = FALSE,
  num.trees = 500, mtry = max(1, floor(sqrt(ncol(x_tr)))), min.node.size = 5,
  seed = 0
)
imp <- sort(rf_fit$variable.importance, decreasing = TRUE)
imp_df <- data.frame(feature = names(imp), importance = as.numeric(imp))

top_k <- min(15, nrow(imp_df))
p_imp <- ggplot(imp_df[1:top_k, ], aes(x = reorder(feature, importance), y = importance)) +
  geom_col(fill = "#ff7f0e") +   # ← 统一橙色
  coord_flip() +
  labs(title = "Feature Importance (Random Forest)", x = "Feature", y = "Importance") +
  theme_minimal(base_size = 13)
ggsave("plots/05_FeatureImportance.png", p_imp, width = 6.5, height = 5.5, dpi = 300)

# =========================
# 6) Pairplot（统一蓝/橙）
# =========================
pair_feats <- imp_df$feature[1:4]
pair_df <- df[, c("diagnosis", pair_feats)]
pair_df$diagnosis <- factor(ifelse(pair_df$diagnosis == "M","Malignant","Benign"),
                            levels = c("Benign","Malignant"))

p_pairs <- GGally::ggpairs(
  pair_df,
  columns = 2:ncol(pair_df),
  aes(colour = diagnosis, fill = diagnosis, alpha = 0.7),
  upper = list(continuous = "points"),
  lower = list(continuous = "smooth_loess"),
  diag  = list(continuous = "densityDiag")
) +
  scale_color_manual(values = color_map) +
  scale_fill_manual(values = color_map) +
  theme_bw()

ggsave("plots/06_Pairplot.png", p_pairs, width = 10.5, height = 9.0, dpi = 300)

cat("🎉 六张颜色统一的图已生成，请查看 ./plots 目录！\n")








# ======================================================
# Q3-level Visual Analytics Pipeline for Breast Cancer
# - Datasets: bc_data(1).csv, brca_clean.csv
# - Models: Naive Bayes, Logistic Regression, Random Forest,
#           SVM (RBF), XGBoost
# - CV: 5-fold, ROC as primary metric
# - Outputs:
#   * 01_Heatmap.png
#   * 02_Confusion_<model>.png
#   * 03_ROC_MultiModel.png
#   * 04_PCA.png
#   * 05_FeatureImportance_RF.png
#   * 06_Pairplot.png
#   * 07_SHAP_XGB.png
#   * metrics_cv.csv, metrics_test.csv
# ======================================================

# ---- Packages ----
pkgs <- c(
  "e1071","caret","ggplot2","pROC","ggcorrplot",
  "dplyr","tidyr","forcats","scales","GGally",
  "ranger","randomForest","xgboost","SHAPforxgboost"
)

new_pkgs <- setdiff(pkgs, rownames(installed.packages()))
if (length(new_pkgs)) install.packages(new_pkgs, dependencies = TRUE)
invisible(lapply(pkgs, library, character.only = TRUE))

set.seed(0)
root_dir <- "plots_q3"
dir.create(root_dir, showWarnings = FALSE)

# ===== 全局颜色设定（良性=蓝，恶性=橙） =====
color_map <- c(
  "Benign"    = "#1f77b4",
  "B"         = "#1f77b4",
  "Malignant" = "#ff7f0e",
  "M"         = "#ff7f0e"
)

# ------------------------------------------------------
# 通用预处理函数：给任意数据集统一格式
# ------------------------------------------------------
prepare_df <- function(raw) {
  df <- raw
  
  # 识别 / 重命名 diagnosis 列
  if (!"diagnosis" %in% tolower(names(df))) {
    dn <- names(df)[grepl("^diag", tolower(names(df)))]
    if (length(dn) == 1) names(df)[names(df) == dn] <- "diagnosis"
  }
  names(df) <- make.names(names(df), unique = TRUE)
  stopifnot("diagnosis" %in% names(df))
  
  # diagnosis 统一为因子 B/M
  df$diagnosis <- toupper(df$diagnosis)
  df$diagnosis <- factor(df$diagnosis, levels = c("B","M"))
  
  # 自动识别 ID 列 & 数值特征列（按列名，不按顺序）
  id_cols  <- names(df)[grepl("^id$|patient|^X\\.?.*id$", tolower(names(df)))]
  num_cols <- setdiff(names(df), c(id_cols, "diagnosis"))
  
  # 转成 numeric
  df[num_cols] <- lapply(df[num_cols], function(x) as.numeric(as.character(x)))
  
  list(df = df, id_cols = id_cols, num_cols = num_cols)
}

# ------------------------------------------------------
# 单数据集完整 Q3 Pipeline
# ------------------------------------------------------
run_pipeline_q3 <- function(csv_path, tag) {
  message("=== Running Q3 pipeline for: ", tag, " ===")
  out_dir <- file.path(root_dir, tag)
  dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)
  
  # ---- 读数据 & 预处理 ----
  raw <- read.csv(csv_path, header = TRUE, stringsAsFactors = FALSE, check.names = FALSE)
  prep <- prepare_df(raw)
  df       <- prep$df
  num_cols <- prep$num_cols
  
  # =========================
  # 0) 训练 / 测试划分（用于可视化）
  # =========================
  set.seed(0)
  idx <- caret::createDataPartition(df$diagnosis, p = 0.7, list = FALSE)
  train_df <- df[idx, ]
  test_df  <- df[-idx, ]
  
  # =========================
  # 1) Heatmap（相关结构）
  # =========================
  y_numeric <- as.numeric(df$diagnosis == "M")
  corr_mat <- cor(cbind(y_numeric = y_numeric, df[, num_cols]),
                  use = "pairwise.complete.obs")
  p_heat <- ggcorrplot(
    corr_mat, type = "lower", lab = TRUE, lab_size = 2.5,
    outline.col = "white",
    title = paste0("Feature Correlation (incl. label) — ", tag)
  )
  ggsave(file.path(out_dir, "01_Heatmap.png"),
         p_heat, width = 10, height = 8, dpi = 300)
  
  # =========================
  # 2) 多模型训练（5-fold CV）
  # =========================
  train_ctrl <- trainControl(
    method = "cv",
    number = 5,
    classProbs = TRUE,
    summaryFunction = twoClassSummary,
    savePredictions = "final"
  )
  
  # caret 里的模型名
  model_specs <- c(
    NB  = "nb",          # Naive Bayes
    LR  = "glm",         # Logistic Regression
    RF  = "rf",          # Random Forest
    SVM = "svmRadial",   # SVM with RBF kernel
    XGB = "xgbTree"      # XGBoost
  )
  
  models <- list()
  metrics_cv <- data.frame(
    model = character(),
    ROC_mean = numeric(),
    ROC_sd   = numeric(),
    Sens_mean = numeric(),
    Spec_mean = numeric(),
    stringsAsFactors = FALSE
  )
  
  # caret 公式：全部数值特征 + diagnosis
  formula_all <- as.formula("diagnosis ~ .")
  
  for (m_name in names(model_specs)) {
    method <- model_specs[[m_name]]
    message("  -> Training model: ", m_name, " (", method, ")")
    
    set.seed(0)
    fit <- caret::train(
      formula_all,
      data = train_df,
      method = method,
      metric = "ROC",
      trControl = train_ctrl,
      preProcess = c("center","scale")
    )
    
    # 记录模型
    models[[m_name]] <- fit
    
    # 从 CV 结果中提取表现（用 ROC 最大的那一行）
    best_idx <- which.max(fit$results$ROC)
    res_best <- fit$results[best_idx, ]
    
    metrics_cv <- rbind(
      metrics_cv,
      data.frame(
        model = m_name,
        ROC_mean  = res_best$ROC,
        ROC_sd    = if ("ROCSD"  %in% names(res_best)) res_best$ROCSD  else NA,
        Sens_mean = if ("Sens"   %in% names(res_best)) res_best$Sens   else NA,
        Spec_mean = if ("Spec"   %in% names(res_best)) res_best$Spec   else NA,
        stringsAsFactors = FALSE
      )
    )
  }
  
  # 保存 CV 表
  write.csv(metrics_cv, file.path(out_dir, "metrics_cv.csv"), row.names = FALSE)
  
  # =========================
  # 3) 测试集表现 + 多模型 ROC + Confusion
  # =========================
  metrics_test <- data.frame(
    model = character(),
    Accuracy = numeric(),
    Sensitivity = numeric(),
    Specificity = numeric(),
    stringsAsFactors = FALSE
  )
  
  roc_list <- list()
  
  for (m_name in names(models)) {
    fit <- models[[m_name]]
    
    # 概率预测（M 类）
    prob <- predict(fit, newdata = test_df, type = "prob")[, "M"]
    pred_cls <- predict(fit, newdata = test_df)
    
    cm <- caret::confusionMatrix(pred_cls, test_df$diagnosis, positive = "M")
    
    metrics_test <- rbind(
      metrics_test,
      data.frame(
        model = m_name,
        Accuracy   = cm$overall["Accuracy"],
        Sensitivity = cm$byClass["Sensitivity"],
        Specificity = cm$byClass["Specificity"],
        stringsAsFactors = FALSE
      )
    )
    
    # 画单模型 confusion 图
    cm_df <- as.data.frame(cm$table)
    colnames(cm_df) <- c("Predicted","Actual","Freq")
    cm_vis <- cm_df %>%
      group_by(Actual) %>%
      mutate(RowPct = Freq / sum(Freq)) %>%
      ungroup() %>%
      mutate(Label = paste0(Freq, "\n", percent(RowPct, accuracy = 0.1)))
    
    p_cm <- ggplot(cm_vis, aes(Predicted, Actual, fill = Freq)) +
      geom_tile() +
      geom_text(aes(label = Label), color = "white", size = 4, lineheight = 0.9) +
      scale_fill_gradient(low = "white", high = "#1f77b4") +
      coord_equal() +
      labs(
        title = paste0("Confusion Matrix — ", m_name, " — ", tag),
        x = "Predicted", y = "Actual"
      ) +
      theme_minimal(base_size = 13) +
      theme(axis.text.x = element_text(angle = 45, hjust = 1))
    
    ggsave(file.path(out_dir, paste0("02_Confusion_", m_name, ".png")),
           p_cm, width = 7.2, height = 6.0, dpi = 300)
    
    # ROC（测试集）
    roc_obj <- pROC::roc(
      response = test_df$diagnosis,
      predictor = prob,
      levels = c("B","M"),
      direction = "<"
    )
    roc_list[[m_name]] <- roc_obj
  }
  
  # 保存测试集表现表
  write.csv(metrics_test, file.path(out_dir, "metrics_test.csv"), row.names = FALSE)
  
  # 多模型 ROC 合图
  p_roc_multi <- pROC::ggroc(roc_list, size = 1.0) +
    geom_abline(slope = 1, intercept = 0, linetype = 2, color = "gray50") +
    scale_color_discrete(name = "Model") +
    ggtitle(paste0("Multi-model ROC — ", tag)) +
    theme_minimal(base_size = 13)
  
  ggsave(file.path(out_dir, "03_ROC_MultiModel.png"),
         p_roc_multi, width = 7, height = 6, dpi = 300)
  
  # =========================
  # 4) PCA Scatterplot（类可分性）
  # =========================
  X_all <- scale(df[, num_cols])
  pca   <- prcomp(X_all, center = FALSE, scale. = FALSE)
  scores <- as.data.frame(pca$x[, 1:2]); colnames(scores) <- c("PC1","PC2")
  scores$diagnosis <- df$diagnosis
  explained <- (pca$sdev^2) / sum(pca$sdev^2)
  pc1_var <- scales::percent(explained[1], accuracy = 0.1)
  pc2_var <- scales::percent(explained[2], accuracy = 0.1)
  
  p_pca <- ggplot(scores, aes(PC1, PC2, color = diagnosis)) +
    geom_point(alpha = 0.85, size = 2) +
    stat_ellipse(level = 0.95, linetype = 2, linewidth = 0.7) +
    scale_color_manual(values = color_map) +
    labs(
      title = paste0("PCA Scatterplot (Standardized Features) — ", tag),
      x = paste0("PC1 (", pc1_var, ")"),
      y = paste0("PC2 (", pc2_var, ")"),
      color = "Diagnosis"
    ) +
    theme_minimal(base_size = 13)
  
  ggsave(file.path(out_dir, "04_PCA.png"),
         p_pca, width = 7.2, height = 6.0, dpi = 300)
  
  # =========================
  # 5) RF Feature Importance（解释全局特征）
  # =========================
  # 用训练集上的 x_tr 特征进行 RF
  # 这里复用上面 SVM/XGB 的预处理：直接对 train_df 做 scale
  x_tr_all <- train_df[, num_cols, drop = FALSE]
  pp_rf    <- preProcess(x_tr_all, method = c("medianImpute","center","scale"))
  x_tr    <- predict(pp_rf, x_tr_all)
  rf_train <- data.frame(diagnosis = train_df$diagnosis, x_tr)
  
  rf_fit <- ranger::ranger(
    diagnosis ~ ., data = rf_train,
    importance = "impurity", probability = FALSE,
    num.trees = 500,
    mtry = max(1, floor(sqrt(ncol(x_tr)))),
    min.node.size = 5,
    seed = 0
  )
  imp <- sort(rf_fit$variable.importance, decreasing = TRUE)
  imp_df <- data.frame(feature = names(imp), importance = as.numeric(imp))
  
  top_k <- min(15, nrow(imp_df))
  p_imp <- ggplot(imp_df[1:top_k, ],
                  aes(x = reorder(feature, importance), y = importance)) +
    geom_col(fill = "#ff7f0e") +
    coord_flip() +
    labs(
      title = paste0("Feature Importance (Random Forest) — ", tag),
      x = "Feature", y = "Importance"
    ) +
    theme_minimal(base_size = 13)
  
  ggsave(file.path(out_dir, "05_FeatureImportance_RF.png"),
         p_imp, width = 6.5, height = 5.5, dpi = 300)
  
  # =========================
  # 6) Pairplot（前4重要特征）
  # =========================
  pair_feats <- imp_df$feature[1:4]
  pair_df <- df[, c("diagnosis", pair_feats)]
  pair_df$diagnosis <- factor(
    ifelse(pair_df$diagnosis == "M","Malignant","Benign"),
    levels = c("Benign","Malignant")
  )
  
  p_pairs <- GGally::ggpairs(
    pair_df,
    columns = 2:ncol(pair_df),
    aes(colour = diagnosis, fill = diagnosis, alpha = 0.7),
    upper = list(continuous = "points"),
    lower = list(continuous = "smooth_loess"),
    diag  = list(continuous = "densityDiag")
  ) +
    scale_color_manual(values = color_map) +
    scale_fill_manual(values = color_map) +
    theme_bw()
  
  ggsave(file.path(out_dir, "06_Pairplot.png"),
         p_pairs, width = 10.5, height = 9.0, dpi = 300)
  
  # =========================
  # =========================
  # 7) SHAP for XGBoost（模型无关解释）
  # =========================
  if ("XGB" %in% names(models)) {
    xgb_fit <- models[["XGB"]]
    xgb_model <- xgb_fit$finalModel   # caret 的 xgbTree 最终模型
    
    # 构建设计矩阵（去掉 diagnosis，-1 表示不加截距）
    X_train <- model.matrix(diagnosis ~ . - 1, data = train_df)
    
    # 计算 SHAP 值
    shap_vals <- SHAPforxgboost::shap.values(
      xgb_model,
      X_train
    )
    
    # ⭐ 正确写法：必须显式指定 X_train = X_train
    shap_long <- SHAPforxgboost::shap.prep(
      shap_contrib = shap_vals$shap_score,
      X_train = X_train
    )
    
    # Summary plot
    p_shap <- SHAPforxgboost::shap.plot.summary(shap_long)
    ggsave(file.path(out_dir, "07_SHAP_XGB.png"),
           p_shap, width = 7.5, height = 6.5, dpi = 300)
  }
  
  message("🎉 Q3 级别全流程完成：", out_dir)
}

# ------------------------------------------------------
# 实际跑两个数据集
# ------------------------------------------------------
run_pipeline_q3("C:/Users/RBY/Desktop/bc_data(1).csv", tag = "bc")
run_pipeline_q3("C:/Users/RBY/Desktop/brca_clean.csv",  tag = "brca")

# ------------------------------------------------------
#  Q1 Upgrade 1: Cross-dataset Generalization (XGB)
# ------------------------------------------------------

library(caret)
library(pROC)

# 复用你上面的 prepare_df()
# 假设它已经定义好

train_test_xgb_cross <- function(train_csv, test_csv, tag_train, tag_test, out_dir = "plots_q1_cross") {
  dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)
  
  # 读训练集
  raw_tr <- read.csv(train_csv, header = TRUE, stringsAsFactors = FALSE, check.names = FALSE)
  prep_tr <- prepare_df(raw_tr)
  df_tr   <- prep_tr$df
  num_tr  <- prep_tr$num_cols
  
  # 读测试集
  raw_te <- read.csv(test_csv, header = TRUE, stringsAsFactors = FALSE, check.names = FALSE)
  prep_te <- prepare_df(raw_te)
  df_te   <- prep_te$df
  num_te  <- prep_te$num_cols
  
  # 取交集特征（确保同样的列）
  common_nums <- intersect(num_tr, num_te)
  message("Common numeric features: ", length(common_nums))
  if (length(common_nums) < 5) stop("Too few common features between datasets.")
  
  # 统一只用共同特征 + diagnosis
  df_tr2 <- df_tr[, c("diagnosis", common_nums)]
  df_te2 <- df_te[, c("diagnosis", common_nums)]
  
  # caret 训练控制
  ctrl <- trainControl(
    method = "cv",
    number = 5,
    classProbs = TRUE,
    summaryFunction = twoClassSummary
  )
  
  set.seed(0)
  fit_xgb <- caret::train(
    diagnosis ~ .,
    data = df_tr2,
    method = "xgbTree",
    metric = "ROC",
    trControl = ctrl,
    preProcess = c("center","scale")
  )
  
  # 在训练集和测试集上评估
  prob_tr <- predict(fit_xgb, newdata = df_tr2, type = "prob")[, "M"]
  prob_te <- predict(fit_xgb, newdata = df_te2, type = "prob")[, "M"]
  
  pred_tr <- predict(fit_xgb, newdata = df_tr2)
  pred_te <- predict(fit_xgb, newdata = df_te2)
  
  cm_tr <- confusionMatrix(pred_tr, df_tr2$diagnosis, positive = "M")
  cm_te <- confusionMatrix(pred_te, df_te2$diagnosis, positive = "M")
  
  # AUC & ROC
  roc_tr <- roc(df_tr2$diagnosis, prob_tr, levels = c("B","M"), direction = "<")
  roc_te <- roc(df_te2$diagnosis, prob_te, levels = c("B","M"), direction = "<")
  
  auc_tr <- as.numeric(auc(roc_tr))
  auc_te <- as.numeric(auc(roc_te))
  
  # 保存指标
  metrics <- data.frame(
    train_dataset = tag_train,
    test_dataset  = tag_test,
    AUC_train     = auc_tr,
    AUC_test      = auc_te,
    Acc_train     = cm_tr$overall["Accuracy"],
    Acc_test      = cm_te$overall["Accuracy"],
    Sens_train    = cm_tr$byClass["Sensitivity"],
    Sens_test     = cm_te$byClass["Sensitivity"],
    Spec_train    = cm_tr$byClass["Specificity"],
    Spec_test     = cm_te$byClass["Specificity"]
  )
  
  out_csv <- file.path(out_dir, paste0("cross_", tag_train, "_to_", tag_test, ".csv"))
  write.csv(metrics, out_csv, row.names = FALSE)
  
  # 画 ROC 对比图（训练 vs 测试）
  p_roc <- ggroc(
    list(Train = roc_tr, CrossTest = roc_te),
    size = 1.0
  ) +
    geom_abline(slope = 1, intercept = 0, linetype = 2, color = "gray50") +
    ggtitle(paste0("Cross-dataset ROC: train=", tag_train, ", test=", tag_test,
                   "\nAUC_train=", sprintf("%.3f", auc_tr),
                   " | AUC_test=", sprintf("%.3f", auc_te))) +
    theme_minimal(base_size = 13)
  
  ggsave(
    file.path(out_dir, paste0("ROC_", tag_train, "_to_", tag_test, ".png")),
    p_roc, width = 7, height = 6, dpi = 300
  )
  
  invisible(list(
    fit = fit_xgb,
    metrics = metrics,
    roc_train = roc_tr,
    roc_test  = roc_te
  ))
}

# 实际调用（路径换成你的）：
cross_bc_to_brca <- train_test_xgb_cross(
  train_csv = "C:/Users/RBY/Desktop/bc_data(1).csv",
  test_csv  = "C:/Users/RBY/Desktop/brca_clean.csv",
  tag_train = "bc",
  tag_test  = "brca"
)

cross_brca_to_bc <- train_test_xgb_cross(
  train_csv = "C:/Users/RBY/Desktop/brca_clean.csv",
  test_csv  = "C:/Users/RBY/Desktop/bc_data(1).csv",
  tag_train = "brca",
  tag_test  = "bc"
)

# ------------------------------------------------------
#  Q1 Upgrade 2: Ablation Study (Top-k features vs All)
# ------------------------------------------------------

run_ablation_topk <- function(csv_path, tag, k_vec = c(5, 10, 15), out_dir = "plots_q1_ablation") {
  dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)
  
  raw <- read.csv(csv_path, header = TRUE, stringsAsFactors = FALSE, check.names = FALSE)
  prep <- prepare_df(raw)
  df   <- prep$df
  num_cols <- prep$num_cols
  
  # 基于 RF importance 选 top feature
  set.seed(0)
  idx <- caret::createDataPartition(df$diagnosis, p = 0.7, list = FALSE)
  train_df <- df[idx, ]
  test_df  <- df[-idx, ]
  
  x_tr_all <- train_df[, num_cols, drop = FALSE]
  pp_rf    <- preProcess(x_tr_all, method = c("medianImpute","center","scale"))
  x_tr     <- predict(pp_rf, x_tr_all)
  rf_train <- data.frame(diagnosis = train_df$diagnosis, x_tr)
  
  rf_fit <- ranger::ranger(
    diagnosis ~ ., data = rf_train,
    importance = "impurity",
    num.trees = 500,
    mtry = max(1, floor(sqrt(ncol(x_tr)))),
    min.node.size = 5,
    seed = 0
  )
  imp <- sort(rf_fit$variable.importance, decreasing = TRUE)
  imp_df <- data.frame(feature = names(imp), importance = as.numeric(imp))
  
  # baseline：所有特征
  res_list <- list()
  
  eval_with_feats <- function(feats, label) {
    tr2 <- train_df[, c("diagnosis", feats)]
    te2 <- test_df[, c("diagnosis", feats)]
    
    ctrl <- trainControl(
      method = "cv",
      number = 5,
      classProbs = TRUE,
      summaryFunction = twoClassSummary
    )
    set.seed(0)
    fit_xgb <- caret::train(
      diagnosis ~ .,
      data = tr2,
      method = "xgbTree",
      metric = "ROC",
      trControl = ctrl,
      preProcess = c("center","scale")
    )
    
    prob_te <- predict(fit_xgb, newdata = te2, type = "prob")[, "M"]
    pred_te <- predict(fit_xgb, newdata = te2)
    cm <- confusionMatrix(pred_te, te2$diagnosis, positive = "M")
    roc_obj <- roc(te2$diagnosis, prob_te, levels = c("B","M"), direction = "<")
    
    data.frame(
      tag = tag,
      setting = label,
      AUC = as.numeric(auc(roc_obj)),
      Accuracy   = cm$overall["Accuracy"],
      Sensitivity = cm$byClass["Sensitivity"],
      Specificity = cm$byClass["Specificity"]
    )
  }
  
  # all features
  res_all <- eval_with_feats(num_cols, "All_features")
  res_list[[1]] <- res_all
  
  # top-k
  for (k in k_vec) {
    k_use <- min(k, nrow(imp_df))
    feats_k <- imp_df$feature[1:k_use]
    res_k <- eval_with_feats(feats_k, paste0("Top_", k_use, "_RF_features"))
    res_list[[length(res_list) + 1]] <- res_k
  }
  
  res_ablation <- do.call(rbind, res_list)
  write.csv(res_ablation, file.path(out_dir, paste0("ablation_", tag, ".csv")), row.names = FALSE)
  
  # 画 AUC 对比图
  p_auc <- ggplot(res_ablation, aes(x = setting, y = AUC, group = 1)) +
    geom_point(size = 3, colour = "#ff7f0e") +
    geom_line(linetype = 2) +
    labs(title = paste0("Ablation on Feature Sets — ", tag),
         x = "Feature Setting", y = "AUC (Test)") +
    theme_minimal(base_size = 13) +
    theme(axis.text.x = element_text(angle = 45, hjust = 1))
  
  ggsave(file.path(out_dir, paste0("ablation_auc_", tag, ".png")),
         p_auc, width = 8, height = 5.5, dpi = 300)
  
  invisible(res_ablation)
}

# 调用：
abl_bc   <- run_ablation_topk("C:/Users/RBY/Desktop/bc_data(1).csv", tag = "bc")
abl_brca <- run_ablation_topk("C:/Users/RBY/Desktop/brca_clean.csv",  tag = "brca")

