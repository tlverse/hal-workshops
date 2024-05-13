expit <- function(x) exp(x) / (1 + exp(x))
logit <- function(x) log(x/(1-x))

generate_data <- function(n, p_cov, p_image, ratio_nonzero, trt_coef, coef_A, int_A, coef_Y) {
  p_nonzero <- round(ratio_nonzero * p_image)  # sparse img signal
  coef_A <- list(cov = rep(coef_A, p_cov), 
                 img = rep(0, p_nonzero))
  coef_Y <- list(cov = rep(coef_Y, p_cov), 
                 img = rep(coef_Y, p_nonzero)) 
  x1 <- rbinom(n*p_cov,1,0.5) %>% matrix(nrow = n)
  X <- cbind(x1)
  trt_prob <- int_A + X %*% coef_A$cov
  outcome_prob <- X %*% coef_Y$cov
  
  img_signal <- rbinom(n*p_nonzero, 1, 0.5) %>% matrix(nrow = n)
  img_noise <- rbinom(n*(p_image - p_nonzero), 1, 0.5) %>% matrix(nrow = n)
  img_mat <- cbind(img_signal, img_noise)
  X <- cbind(X, img_mat)  
  trt_prob <- trt_prob + img_signal %*% coef_A$img
  trt_prob <- trt_prob %>% expit
  outcome_prob <- outcome_prob + img_signal %*% coef_Y$img
  
  value_treat <- rbinom(n = n, size = 1, trt_prob)  # otherwise keep the imputed value
  X_output <- cbind(X, value_treat)
  
  colnames(X_output)[1:p_cov] <- paste0("X", 1:p_cov)
  colnames(X_output)[(1:p_image) + p_cov] <- paste0("img", 1:p_image)
  colnames(X_output)[p_cov + p_image + 1] <- "trt"
  
  outcome_prob <- outcome_prob + trt_coef * value_treat
  value_outcome <- outcome_prob + rnorm(length(outcome_prob))
  X_output <- data.frame(X_output, Y = value_outcome)
  
  return(X_output)
}


call_with_args <- function(fun, args, other_valid = list(), keep_all = FALSE,
                           silent = FALSE, ignore = c()) {
  
  # drop ignore args
  args <- args[!(names(args) %in% ignore)]
  if (!keep_all) {
    # catch arguments to be kept
    formal_args <- names(formals(fun))
    all_valid <- c(formal_args, other_valid)
    
    # find invalid arguments based on combination of formals and other_valid
    invalid <- names(args)[which(!(names(args) %in% all_valid))]
    
    # subset arguments to pass
    args <- args[which(names(args) %in% all_valid)]
    
    # return warnings when dropping arguments
    if (!silent & length(invalid) > 0) {
      message(sprintf(
        "Learner called function %s with unknown args: %s. These will be dropped.\nCheck the params supported by this learner.",
        as.character(substitute(fun)), paste(invalid, collapse = ", ")
      ))
    }
  }
  do.call(fun, args)
}




Lrnr_screener_name <- R6Class(
  classname = "Lrnr_screener_name",
  inherit = Lrnr_base, portable = TRUE, class = TRUE,
  public = list(
    initialize = function(var_name = NULL) {
      params <- args_to_list()
      private$.var_name <- var_name
      super$initialize(params = params)
    }
  ),
  private = list(
    .properties = c("binomial", "continuous", "categorical", "screener"),
    .var_name = NULL, 
    .train = function(task) {
      outcome_type <- self$get_outcome_type(task)
      X <- task$X
      Y <- outcome_type$format(task$Y)
      covs <- task$nodes$covariates
      
      args <- self$params
      
      selected <- names(X) == private$.var_name
      
      selected_names <- names(X)[selected]
      selected_covs <- sapply(covs, function(cov) any(grep(paste0("^", cov, "$"), selected_names)))
      fit_object <- list(selected = covs[selected_covs])
      return(fit_object)
    },
    .predict = function(task) {
      task$data[, private$.fit_object$selected, with = FALSE, drop = FALSE]
    },
    .chain = function(task) {
      return(task$next_in_chain(covariates = private$.fit_object$selected))
    }
  )
)





Lrnr_identity <- R6Class(
  classname = "Lrnr_identity", inherit = Lrnr_base,
  portable = TRUE, class = TRUE,
  public = list(
    initialize = function(...) {
      params <- list(...)
      super$initialize(params = params, ...)
    },
    
    print = function() {
      print(self$name)
    }
  ),
  
  private = list(
    .properties = c("continuous", "binomial", "categorical", "weights", "offset"),
    
    .train = function(task) {
      outcome_type <- self$get_outcome_type(task)
      y <- outcome_type$format(task$Y)
      weights <- task$weights
      
      if (task$has_node("offset")) {
        offset <- task$offset
        if (outcome_type$type == "categorical") {
          # todo: fix
          stop("offsets not yet supported for outcome_type='categorical'")
        }
      } else {
        offset <- rep(0, task$nrow)
      }
      
      if (outcome_type$type == "categorical") {
        y_levels <- outcome_type$levels
        means <- sapply(
          y_levels,
          function(level) weighted.mean(y == level, weights)
        )
        fit_object <- list(mean = pack_predictions(matrix(means, nrow = 1)))
      } else {
        temp <- 0
        fit_object <- list(mean = temp)
      }
      
      fit_object$training_offset <- task$has_node("offset")
      
      return(fit_object)
    },
    
    .predict = function(task = NULL) {
      if (ncol(task$X) != 1) {
        print(names(task$X))
        stop("not single covariate") 
      } else {
        predictions <- task$X[, 1] %>% unlist
      }
      
      if (self$fit_object$training_offset) {
        offset <- task$offset_transformed(NULL, for_prediction = TRUE)
        predictions <- predictions + offset
      }
      
      predictions <- as.matrix(predictions, ncol = 1)
      return(predictions)
    }
  )
)




Lrnr_glmnet_overfit_sparse_insert <- R6Class(
  classname = "Lrnr_glmnet",
  inherit = Lrnr_base, portable = TRUE, class = TRUE,
  public = list(
    initialize = function(lambda = NULL, type.measure = "deviance",
                          nfolds = 10, alpha = 1, nlambda = 100,
                          use_min = TRUE, ratio_overfit = 1, trt_name = "trt", trt_value = 1, stratify_cv = FALSE, ...) {
      super$initialize(params = args_to_list(), ...)
    }
  ),
  private = list(
    .properties = c(
      "continuous", "binomial", "categorical",
      "weights", "ids"
    ),
    .train = function(task) {
      args <- self$params
      
      outcome_type <- self$get_outcome_type(task)
      
      if (is.null(args$family)) {
        args$family <- outcome_type$glm_family()
      }
      
      if (args$family %in% "quasibinomial") {
        args$family <- stats::quasibinomial()
      }
      
      # specify data
      args$x <- as.matrix(task$X)
      args$x <- Matrix(args$x, sparse = T)
      args$y <- outcome_type$format(task$Y)
      
      if (task$has_node("weights")) {
        args$weights <- task$weights
      }
      
      if (task$has_node("offset")) {
        args$offset <- task$offset
      }
      
      if (task$has_node("id")) {
        args$foldid <- origami::folds2foldvec(task$folds)
      }
      
      if (args$stratify_cv) {
        if (outcome_type$type == "binomial" & is.null(args$foldid)) {
          folds <- origami::make_folds(
            n = length(args$y), strata_ids = args$y, fold_fun = folds_vfold,
            V = as.integer(args$nfolds)
          )
          args$foldid <- origami::folds2foldvec(folds)
        } else {
          warning(
            "stratify_cv is TRUE; but inner cross-validation folds cannot ",
            "be stratified. Either the outcome is not binomial, or foldid ",
            "has already been established (user specified foldid upon ",
            "initializing the learner, or it was set according to task id's)."
          )
        }
      }
      
      fit_object <- call_with_args(
        glmnet::cv.glmnet, args,
        other_valid = names(formals(glmnet::glmnet)),
        ignore = c("use_min", "stratify_cv")
      )
      # fit_object$glmnet.fit$call <- NULL
      cv_lambda <- fit_object$lambda.min
      
      args$lambda <- cv_lambda * args$ratio_overfit
      fit_object <- call_with_args(
        glmnet::glmnet, args,
        other_valid = names(formals(glmnet::glmnet)),
        ignore = c("use_min", "stratify_cv")
      )
      fit_object$call <- NULL
      
      return(fit_object)
    },
    .predict = function(task) {
      trt_name <- self$params$trt_name
      trt_value <- self$params$trt_value
      # browser()
      newx <- task$X
      if (any(colnames(newx) == trt_name)) newx[[trt_name]] <- trt_value
      args <- list(
        object = private$.fit_object, newx = as.matrix(newx), type = "response"
      )
      
      # set choice regularization penalty
      if (self$params$use_min) {
        args$s <- "lambda.min"
      } else {
        args$s <- "lambda.1se"
      }
      
      if (task$has_node("offset")) {
        if (private$.fit_object$offset) {
          args$newoffset <- task$offset
        } else {
          warning(
            "Prediction task has offset, but an offset was not included in ",
            "the task for training the glmnet learner. The prediction task's ",
            "offset will not be considered for prediction."
          )
        }
      }
      
      # get predictions via S3 method
      predictions <- do.call(stats::predict, args)
      
      # reformat predictions based on outcome type
      if (private$.training_outcome_type$type == "categorical") {
        cat_names <- dimnames(predictions)[[2]]
        # predictions is a 3-dim matrix, convert to 2-dim matrix
        dim(predictions) <- dim(predictions)[1:2]
        colnames(predictions) <- cat_names
        # pack predictions in a single column
        predictions <- pack_predictions(predictions)
      }
      return(predictions)
    },
    .required_packages = c("glmnet", "origami")
  )
)





Lrnr_glmnet_overfit_sparse_G <- R6Class(
  classname = "Lrnr_glmnet",
  inherit = Lrnr_base, portable = TRUE, class = TRUE,
  public = list(
    initialize = function(lambda = NULL, type.measure = "deviance",
                          nfolds = 10, alpha = 1, nlambda = 100,
                          use_min = TRUE, ratio_overfit = 1, force_DV = "trt", stratify_cv = FALSE, ...) {
      super$initialize(params = args_to_list(), ...)
    }
  ),
  private = list(
    .properties = c(
      "continuous", "binomial", "categorical",
      "weights", "ids"
    ),
    .train = function(task) {
      args <- self$params
      
      outcome_type <- self$get_outcome_type(task)

      if (is.null(args$family)) {
        args$family <- "binomial"
      }
      
      if (args$family %in% "quasibinomial") {
        args$family <- stats::quasibinomial()
      }
      
      # specify data
      args$x <- as.matrix(task$X)
      args$x <- Matrix(args$x, sparse = T)
      args$y <- outcome_type$format(task$Y)
      
      # browser()
      force_DV <- args$force_DV
      if (any(colnames(task$X) == force_DV)) {
        args$x <- args$x[, -which(colnames(args$x) == force_DV)]
        args$y <- task$X[[force_DV]]
      }
      
      if (task$has_node("weights")) {
        args$weights <- task$weights
      }
      
      if (task$has_node("offset")) {
        args$offset <- task$offset
      }
      
      if (task$has_node("id")) {
        args$foldid <- origami::folds2foldvec(task$folds)
      }
      
      if (args$stratify_cv) {
        if (outcome_type$type == "binomial" & is.null(args$foldid)) {
          folds <- origami::make_folds(
            n = length(args$y), strata_ids = args$y, fold_fun = folds_vfold,
            V = as.integer(args$nfolds)
          )
          args$foldid <- origami::folds2foldvec(folds)
        } else {
          warning(
            "stratify_cv is TRUE; but inner cross-validation folds cannot ",
            "be stratified. Either the outcome is not binomial, or foldid ",
            "has already been established (user specified foldid upon ",
            "initializing the learner, or it was set according to task id's)."
          )
        }
      }
      
      fit_object <- call_with_args(
        glmnet::cv.glmnet, args,
        other_valid = names(formals(glmnet::glmnet)),
        ignore = c("use_min", "stratify_cv")
      )
      # fit_object$glmnet.fit$call <- NULL
      cv_lambda <- fit_object$lambda.min
      
      # browser()
      args$lambda <- cv_lambda * args$ratio_overfit
      fit_object <- call_with_args(
        glmnet::glmnet, args,
        other_valid = names(formals(glmnet::glmnet)),
        ignore = c("use_min", "stratify_cv")
      )
      fit_object$call <- NULL
      
      return(fit_object)
    },
    .predict = function(task) {
      args <- list(
        object = private$.fit_object, newx = as.matrix(task$X), type = "response"
      )
      
      force_DV <- self$params$force_DV
      if (any(colnames(task$X) == force_DV)) {
        args$newx <- args$newx[, -which(colnames(args$newx) == force_DV)]
      }
      
      # set choice regularization penalty
      if (self$params$use_min) {
        args$s <- "lambda.min"
      } else {
        args$s <- "lambda.1se"
      }
      
      if (task$has_node("offset")) {
        if (private$.fit_object$offset) {
          args$newoffset <- task$offset
        } else {
          warning(
            "Prediction task has offset, but an offset was not included in ",
            "the task for training the glmnet learner. The prediction task's ",
            "offset will not be considered for prediction."
          )
        }
      }
      
      # get predictions via S3 method
      predictions <- do.call(stats::predict, args)
      
      # reformat predictions based on outcome type
      if (private$.training_outcome_type$type == "categorical") {
        cat_names <- dimnames(predictions)[[2]]
        # predictions is a 3-dim matrix, convert to 2-dim matrix
        dim(predictions) <- dim(predictions)[1:2]
        colnames(predictions) <- cat_names
        # pack predictions in a single column
        predictions <- pack_predictions(predictions)
      }
      return(predictions)
    },
    .required_packages = c("glmnet", "origami")
  )
)


Lrnr_mean_wrong <- R6Class(
  classname = "Lrnr_mean", inherit = Lrnr_base,
  portable = TRUE, class = TRUE,
  public = list(
    initialize = function(...) {
      params <- list(...)
      super$initialize(params = params, ...)
    },
    
    print = function() {
      print(self$name)
    }
  ),
  
  private = list(
    .properties = c("continuous", "binomial", "categorical", "weights", "offset"),
    
    .train = function(task) {
      outcome_type <- self$get_outcome_type(task)
      y <- outcome_type$format(task$Y)
      weights <- task$weights
      
      if (task$has_node("offset")) {
        offset <- task$offset
        if (outcome_type$type == "categorical") {
          # todo: fix
          stop("offsets not yet supported for outcome_type='categorical'")
        }
      } else {
        offset <- rep(0, task$nrow)
      }
      
      if (outcome_type$type == "categorical") {
        y_levels <- outcome_type$levels
        means <- sapply(
          y_levels,
          function(level) weighted.mean(y == level, weights)
        )
        fit_object <- list(mean = pack_predictions(matrix(means, nrow = 1)))
      } else {
        temp <- weighted.mean(y - offset, weights) + 0.1
        # if (temp > 0.9) temp <- 0.9  # 20220428 more comparable
        if (temp > 0.999999) temp <- 0.999999
        # if (temp < 0.01) temp <- 0.01
        fit_object <- list(mean = temp)
      }
      
      fit_object$training_offset <- task$has_node("offset")
      
      return(fit_object)
    },
    
    .predict = function(task = NULL) {
      predictions <- rep(private$.fit_object$mean, task$nrow)
      
      if (self$fit_object$training_offset) {
        offset <- task$offset_transformed(NULL, for_prediction = TRUE)
        predictions <- predictions + offset
      }
      
      predictions <- as.matrix(predictions, ncol = 1)
      return(predictions)
    }
  )
)
