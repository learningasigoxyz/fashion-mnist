#!/usr/bin/env Rscript
#
# Benchmark GLM (glmnet), XGBoost, and CNN on Fashion-MNIST.
#
# CNN uses Rcpp for compiled conv/pool kernels + mini-batch SGD in R.
# (R torch requires a network-downloaded backend unavailable here.)
#
# Usage:
#   Rscript benchmark/benchmark_glm_xgb_cnn.R
#

suppressPackageStartupMessages({
  library(glmnet)
  library(xgboost)
  library(Matrix)
  library(Rcpp)
})

# ── data loader ───────────────────────────────────────────────────────────────

load_mnist_gz <- function(path, kind = "train") {
  lf  <- file.path(path, sprintf("%s-labels-idx1-ubyte.gz", kind))
  imf <- file.path(path, sprintf("%s-images-idx3-ubyte.gz", kind))

  con <- gzcon(file(lf, "rb"))
  hdr <- readBin(con, "integer", n = 2L, size = 4L, endian = "big")
  labels <- as.integer(readBin(con, "raw", n = hdr[2]))
  close(con)

  con  <- gzcon(file(imf, "rb"))
  hdr2 <- readBin(con, "integer", n = 4L, size = 4L, endian = "big")
  raw  <- readBin(con, "raw", n = hdr2[2] * hdr2[3] * hdr2[4])
  close(con)

  list(x = matrix(as.double(as.integer(raw)) / 255.0,
                  nrow = hdr2[2], byrow = TRUE),
       y = labels)
}

args       <- commandArgs(trailingOnly = FALSE)
script_arg <- args[grep("--file=", args)]
if (length(script_arg)) {
  script_dir <- dirname(normalizePath(sub("--file=", "", script_arg)))
  data_dir   <- file.path(dirname(script_dir), "data", "fashion")
} else {
  data_dir <- file.path(getwd(), "data", "fashion")
}

cat("Loading Fashion-MNIST ...\n")
train <- load_mnist_gz(data_dir, "train")
test  <- load_mnist_gz(data_dir, "t10k")
X_train <- train$x;  y_train <- train$y
X_test  <- test$x;   y_test  <- test$y
cat(sprintf("Train: %d x %d   Test: %d x %d\n",
            nrow(X_train), ncol(X_train), nrow(X_test), ncol(X_test)))

print_result <- function(name, acc, elapsed) {
  cat(sprintf("\n%s\n", strrep("=", 50)))
  cat(sprintf("Model        : %s\n", name))
  cat(sprintf("Test Accuracy: %.2f%%\n", acc * 100))
  cat(sprintf("Train Time   : %.1fs\n", elapsed))
  cat(sprintf("%s\n", strrep("=", 50)))
}
results <- list()

# ── 1. GLM ────────────────────────────────────────────────────────────────────

cat("\n--- GLM (glmnet multinomial ridge) ---\n")
t0 <- proc.time()[["elapsed"]]

col_means <- colMeans(X_train)
col_sds   <- apply(X_train, 2, sd); col_sds[col_sds == 0] <- 1.0
X_tr_sc   <- sweep(sweep(X_train, 2, col_means, "-"), 2, col_sds, "/")
X_te_sc   <- sweep(sweep(X_test,  2, col_means, "-"), 2, col_sds, "/")

fit_glm  <- glmnet(X_tr_sc, factor(y_train), family = "multinomial",
                   alpha = 0, standardize = FALSE, maxit = 5000)
best_lam <- min(fit_glm$lambda)
pred_glm <- predict(fit_glm, newx = X_te_sc, type = "class", s = best_lam)
acc_glm  <- mean(as.integer(as.character(pred_glm[, 1])) == y_test)
t_glm    <- proc.time()[["elapsed"]] - t0

print_result("GLM (glmnet multinomial ridge)", acc_glm, t_glm)
results[["GLM"]] <- c(acc = acc_glm, time = t_glm)

# ── 2. XGBoost ───────────────────────────────────────────────────────────────

cat("\n--- XGBoost ---\n")
t0 <- proc.time()[["elapsed"]]

dtrain    <- xgb.DMatrix(data = X_train, label = y_train)
dtest     <- xgb.DMatrix(data = X_test,  label = y_test)
params    <- list(booster = "gbtree", objective = "multi:softmax",
                  num_class = 10, max_depth = 6, eta = 0.1,
                  subsample = 0.8, colsample_bytree = 0.8,
                  tree_method = "hist",
                  nthread = parallel::detectCores())
model_xgb <- xgb.train(params = params, data = dtrain,
                        nrounds = 200, verbose = 0)
acc_xgb   <- mean(predict(model_xgb, dtest) == y_test)
t_xgb     <- proc.time()[["elapsed"]] - t0

print_result("XGBoost", acc_xgb, t_xgb)
results[["XGBoost"]] <- c(acc = acc_xgb, time = t_xgb)

# ── 3. CNN via Rcpp ───────────────────────────────────────────────────────────
#
# Architecture (NCHW vector layout throughout):
#   Conv(1→16, 3×3, pad=1) → ReLU → MaxPool(2×2)   → [N,16,14,14]
#   Conv(16→32, 3×3, pad=1) → ReLU → MaxPool(2×2)  → [N,32,7,7]
#   Flatten → Dense(1568→128) → ReLU → Dense(128→10)
#   Loss: softmax cross-entropy   Optimizer: SGD

cat("\n--- CNN (Rcpp kernels + R SGD) ---\n")

sourceCpp(code = '
// [[Rcpp::plugins(openmp)]]
#include <Rcpp.h>
#ifdef _OPENMP
#include <omp.h>
#endif
using namespace Rcpp;

// im2col: X (N,C,H,W) → col (N*oH*oW, C*kH*kW)
// [[Rcpp::export]]
NumericMatrix im2col_cpp(NumericVector X,
                         int N, int C, int H, int W,
                         int kH, int kW, int stride, int pad) {
  int oH = (H+2*pad-kH)/stride+1, oW = (W+2*pad-kW)/stride+1;
  NumericMatrix col(N*oH*oW, C*kH*kW);
  #pragma omp parallel for collapse(2) schedule(static)
  for (int n=0; n<N; n++)
  for (int c=0; c<C; c++)
  for (int kh=0; kh<kH; kh++)
  for (int kw=0; kw<kW; kw++) {
    int ci = c*kH*kW + kh*kW + kw;
    for (int oh=0; oh<oH; oh++) { int h = oh*stride - pad + kh;
    for (int ow=0; ow<oW; ow++) { int w = ow*stride - pad + kw;
      if (h>=0 && h<H && w>=0 && w<W)
        col(n*oH*oW + oh*oW + ow, ci) = X[n*C*H*W + c*H*W + h*W + w];
    }}
  }
  return col;
}

// col2im: dcol (N*oH*oW, C*kH*kW) → dX (N,C,H,W)
// Safe to parallelize over n: each n writes to a disjoint range of dX.
// [[Rcpp::export]]
NumericVector col2im_cpp(NumericMatrix dcol,
                         int N, int C, int H, int W,
                         int kH, int kW, int stride, int pad) {
  int oH = (H+2*pad-kH)/stride+1, oW = (W+2*pad-kW)/stride+1;
  NumericVector dX(N*C*H*W, 0.0);
  #pragma omp parallel for schedule(static)
  for (int n=0; n<N; n++)
  for (int c=0; c<C; c++)
  for (int kh=0; kh<kH; kh++)
  for (int kw=0; kw<kW; kw++) {
    int ci = c*kH*kW + kh*kW + kw;
    for (int oh=0; oh<oH; oh++) { int h = oh*stride - pad + kh;
    for (int ow=0; ow<oW; ow++) { int w = ow*stride - pad + kw;
      if (h>=0 && h<H && w>=0 && w<W)
        dX[n*C*H*W + c*H*W + h*W + w] += dcol(n*oH*oW + oh*oW + ow, ci);
    }}
  }
  return dX;
}

// Max-pool forward
// [[Rcpp::export]]
List maxpool_fwd_cpp(NumericVector X,
                     int N, int C, int H, int W,
                     int pH, int pW) {
  int oH = H/pH, oW = W/pW;
  NumericVector out(N*C*oH*oW, R_NegInf);
  IntegerVector mask(N*C*H*W, 0);
  #pragma omp parallel for collapse(2) schedule(static)
  for (int n=0; n<N; n++)
  for (int c=0; c<C; c++)
  for (int oh=0; oh<oH; oh++)
  for (int ow=0; ow<oW; ow++) {
    int oi = n*C*oH*oW + c*oH*oW + oh*oW + ow;
    for (int ph=0; ph<pH; ph++)
    for (int pw=0; pw<pW; pw++) {
      int h = oh*pH+ph, w = ow*pW+pw;
      int ii = n*C*H*W + c*H*W + h*W + w;
      if (X[ii] > out[oi]) { out[oi] = X[ii]; mask[ii] = 1; }
    }
  }
  return List::create(Named("out")=out, Named("mask")=mask);
}

// Max-pool backward
// [[Rcpp::export]]
NumericVector maxpool_bwd_cpp(NumericVector dout, IntegerVector mask,
                               int N, int C, int H, int W,
                               int pH, int pW) {
  int oH = H/pH, oW = W/pW;
  NumericVector dX(N*C*H*W, 0.0);
  #pragma omp parallel for collapse(2) schedule(static)
  for (int n=0; n<N; n++)
  for (int c=0; c<C; c++)
  for (int oh=0; oh<oH; oh++)
  for (int ow=0; ow<oW; ow++) {
    int oi = n*C*oH*oW + c*oH*oW + oh*oW + ow;
    for (int ph=0; ph<pH; ph++)
    for (int pw=0; pw<pW; pw++) {
      int h=oh*pH+ph, w=ow*pW+pw;
      int ii=n*C*H*W+c*H*W+h*W+w;
      if (mask[ii]) dX[ii] += dout[oi];
    }
  }
  return dX;
}
')

# ── R-level helpers ───────────────────────────────────────────────────────────

relu   <- function(x) { x[x < 0] <- 0; x }
relu_d <- function(pre_act) (pre_act > 0) * 1.0

softmax <- function(x) {
  ex <- exp(x - apply(x, 1, max))
  ex / rowSums(ex)
}

# Vectorised reshapes using aperm (no R-level loops).
# out_mat (N*oH*oW, F) → out_vec (N,F,oH*oW) NCHW
mat_to_nchw <- function(out_mat, N, F, oH, oW)
  as.double(aperm(array(as.double(out_mat), c(oH*oW, N, F)), c(1L, 3L, 2L)))

# out_vec (N,F,oH*oW) NCHW → dout_mat (N*oH*oW, F)
nchw_to_mat <- function(vec, N, F, oH, oW)
  matrix(as.double(aperm(array(vec, c(oH*oW, F, N)), c(1L, 3L, 2L))), N*oH*oW, F)

# conv_fwd: X is (N,C,H,W) flat NCHW vector, W_conv is (F, C*kH*kW)
conv_fwd <- function(X_vec, N, C, H, W, W_conv, b_conv, kH, kW, pad = 1L) {
  F   <- nrow(W_conv)
  oH  <- H + 2L*pad - kH + 1L
  oW  <- W + 2L*pad - kW + 1L
  col <- im2col_cpp(X_vec, N, C, H, W, kH, kW, 1L, pad)  # (N*oH*oW, C*kH*kW)
  out_mat <- col %*% t(W_conv) + rep(b_conv, each = nrow(col))  # (N*oH*oW, F)
  out_vec <- mat_to_nchw(out_mat, N, F, oH, oW)
  list(out = out_vec, col = col, N = N, C = C, H = H, W = W,
       F = F, oH = oH, oW = oW, kH = kH, kW = kW, pad = pad, W_conv = W_conv)
}

# conv_bwd: dout_vec is (N,F,oH,oW) flat NCHW vector
conv_bwd <- function(dout_vec, cache) {
  with(cache, {
    dout_mat <- nchw_to_mat(dout_vec, N, F, oH, oW)  # (N*oH*oW, F)
    dW   <- t(dout_mat) %*% col           # (F, C*kH*kW)
    db   <- colSums(dout_mat)             # (F,)
    dcol <- dout_mat %*% W_conv           # (N*oH*oW, C*kH*kW)
    dX   <- col2im_cpp(dcol, N, C, H, W, kH, kW, 1L, pad)
    list(dX = dX, dW = dW, db = db)
  })
}

ce_bwd <- function(logits, y_oh) {
  probs <- softmax(logits)
  N     <- nrow(probs)
  loss  <- -sum(y_oh * log(pmax(probs, 1e-15))) / N
  list(loss = loss, dlogits = (probs - y_oh) / N)
}

he_init <- function(fan_in, fan_out)
  matrix(rnorm(fan_in * fan_out, 0, sqrt(2.0 / fan_in)), fan_in, fan_out)

# ── weight init ───────────────────────────────────────────────────────────────
set.seed(42)
F1 <- 16L;  F2 <- 32L;  kH <- 3L;  kW <- 3L

W_c1  <- matrix(rnorm(F1 * 1  * kH * kW, 0, sqrt(2/(1 *kH*kW))), nrow = F1)
b_c1  <- rep(0.0, F1)
W_c2  <- matrix(rnorm(F2 * F1 * kH * kW, 0, sqrt(2/(F1*kH*kW))), nrow = F2)
b_c2  <- rep(0.0, F2)

flat_size <- F2 * 7L * 7L   # 1568
hidden    <- 128L
W_fc1 <- he_init(flat_size, hidden);  b_fc1 <- rep(0.0, hidden)
W_fc2 <- he_init(hidden, 10L);        b_fc2 <- rep(0.0, 10L)

# ── training ──────────────────────────────────────────────────────────────────
epochs     <- 10L
batch_size <- 128L
lr         <- 0.05   # SGD needs higher lr than Adam
n_train    <- nrow(X_train)

t0_cnn <- proc.time()[["elapsed"]]

for (ep in seq_len(epochs)) {
  perm       <- sample(n_train)
  epoch_loss <- 0.0; nb_count <- 0L

  for (i in seq(1L, n_train, by = batch_size)) {
    bi  <- perm[i:min(i + batch_size - 1L, n_train)]
    nb  <- length(bi)
    # xb: (N,1,28,28) NCHW flat vector.  C=1 so layout is just row-major images.
    xb  <- as.double(X_train[bi, ])   # (nb, 784) row-major → same as NCHW with C=1

    yb   <- y_train[bi]
    y_oh <- matrix(0.0, nb, 10L)
    y_oh[cbind(seq_len(nb), yb + 1L)] <- 1.0

    # ── forward ──────────────────────────────────────────────────────────────
    c1  <- conv_fwd(xb, nb, 1L, 28L, 28L, W_c1, b_c1, kH, kW)
    a1  <- relu(c1$out)
    p1  <- maxpool_fwd_cpp(a1, nb, F1, 28L, 28L, 2L, 2L)

    c2  <- conv_fwd(p1$out, nb, F1, 14L, 14L, W_c2, b_c2, kH, kW)
    a2  <- relu(c2$out)
    p2  <- maxpool_fwd_cpp(a2, nb, F2, 14L, 14L, 2L, 2L)

    # flatten (N,F2,7,7) → (N, flat_size), row per sample
    flat   <- matrix(p2$out, nrow = nb, ncol = flat_size, byrow = TRUE)
    z_fc1  <- flat %*% W_fc1 + rep(b_fc1, each = nb)
    a_fc1  <- relu(z_fc1)
    logits <- a_fc1 %*% W_fc2 + rep(b_fc2, each = nb)

    lsb <- ce_bwd(logits, y_oh)
    epoch_loss <- epoch_loss + lsb$loss; nb_count <- nb_count + 1L

    # ── backward ─────────────────────────────────────────────────────────────
    dl     <- lsb$dlogits                           # (nb, 10)

    dW_fc2 <- t(a_fc1) %*% dl;  db_fc2 <- colSums(dl)
    da_fc1 <- dl %*% t(W_fc2)
    dz_fc1 <- da_fc1 * relu_d(z_fc1)
    dW_fc1 <- t(flat) %*% dz_fc1;  db_fc1 <- colSums(dz_fc1)
    dflat  <- dz_fc1 %*% t(W_fc1)                  # (nb, flat_size)

    # unflatten (nb, flat_size) → (N,F2,7,7) NCHW vector
    dp2_out <- as.double(t(dflat))   # c(dflat[1,], dflat[2,], ...) ✓

    da2  <- maxpool_bwd_cpp(dp2_out, p2$mask, nb, F2, 14L, 14L, 2L, 2L)
    c2$W_conv <- W_c2
    cb2  <- conv_bwd(da2 * relu_d(c2$out), c2)

    da1  <- maxpool_bwd_cpp(cb2$dX, p1$mask, nb, F1, 28L, 28L, 2L, 2L)
    c1$W_conv <- W_c1
    cb1  <- conv_bwd(da1 * relu_d(c1$out), c1)

    # ── SGD ──────────────────────────────────────────────────────────────────
    W_c1  <- W_c1  - lr * cb1$dW;  b_c1  <- b_c1  - lr * cb1$db
    W_c2  <- W_c2  - lr * cb2$dW;  b_c2  <- b_c2  - lr * cb2$db
    W_fc1 <- W_fc1 - lr * dW_fc1;  b_fc1 <- b_fc1 - lr * db_fc1
    W_fc2 <- W_fc2 - lr * dW_fc2;  b_fc2 <- b_fc2 - lr * db_fc2
  }

  cat(sprintf("  Epoch %2d/%d  avg_loss=%.4f\n",
              ep, epochs, epoch_loss / nb_count))
}

# ── evaluation ────────────────────────────────────────────────────────────────
correct <- 0L
eval_bs <- 256L
for (i in seq(1L, nrow(X_test), by = eval_bs)) {
  bi  <- i:min(i + eval_bs - 1L, nrow(X_test))
  nb  <- length(bi)
  xb  <- as.double(X_test[bi, ])

  c1  <- conv_fwd(xb, nb, 1L, 28L, 28L, W_c1, b_c1, kH, kW)
  p1  <- maxpool_fwd_cpp(relu(c1$out), nb, F1, 28L, 28L, 2L, 2L)
  c2  <- conv_fwd(p1$out, nb, F1, 14L, 14L, W_c2, b_c2, kH, kW)
  p2  <- maxpool_fwd_cpp(relu(c2$out), nb, F2, 14L, 14L, 2L, 2L)
  flat <- matrix(p2$out, nrow = nb, ncol = flat_size, byrow = TRUE)
  z1   <- relu(flat %*% W_fc1 + rep(b_fc1, each = nb))
  lg   <- z1 %*% W_fc2 + rep(b_fc2, each = nb)
  correct <- correct + sum(max.col(lg) - 1L == y_test[bi])
}

acc_cnn <- correct / nrow(X_test)
t_cnn   <- proc.time()[["elapsed"]] - t0_cnn

print_result("CNN (Rcpp conv, SGD)", acc_cnn, t_cnn)
results[["CNN"]] <- c(acc = acc_cnn, time = t_cnn)

# ── summary ───────────────────────────────────────────────────────────────────
cat(sprintf("\n\n%s\n", strrep("=", 50)))
cat(sprintf("%-25s %10s %12s\n", "Model", "Accuracy", "Train Time"))
cat(sprintf("%s\n", strrep("-", 50)))
for (nm in names(results))
  cat(sprintf("%-25s %9.2f%% %10.1fs\n",
              nm, results[[nm]]["acc"] * 100, results[[nm]]["time"]))
cat(sprintf("%s\n", strrep("=", 50)))
