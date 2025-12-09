#include "metagb.h"

int addme(int a, int b) {
    return (a + b);
}

void addvec(float *a, float *b, float *c, int n) {
    for (int i = 0; i < n; ++i) {
        c[i] = a[i] + b[i];
    }
}

void print_char(unsigned char *array, int n) {
    for (int i = 0; i < n; ++i) {
        printf("%xd ", array[i]);
    }
}

void print_bool(bool *array, int n) {
    for (int i = 0; i < n; ++i) {
        printf("%d ", array[i]);
    }
    printf("\n");
}

void print_float(float *array, int len, int d) {
    for (int i = 0; i < len; ++i) {
        printf("%.3f ", array[i]);
        if ((i + 1) % d == 0) {
            printf("\n");
        }
    }
    printf("\n");
}

void print_double(double *array, int len, int d) {
    for (int i = 0; i < len; i++) {
        printf("%.3f ", array[i]);
        if ((i + 1) % d == 0)
            printf("\n");
    }
    printf("\n");
}

void print_int(int *array, int len, int d) {
    for (int i = 0; i < len; i++) {
        printf("%d ", array[i]);
        if ((i + 1) % d == 0)
            printf("\n");
    }
    printf("\n");
}

int compareint(const void *a, const void *b) {
    return (*(int*)a - *(int*)b);
}

int comparedouble(const void *a, const void *b) {
    return (int)((*(double*)a - *(double*)b) > 0);
}

double find_median(double *array, int length) {
    qsort(array, length, sizeof(double), comparedouble);
    return array[length/2];
}

int build_tree_float(float *tree,
                     const int node_id,
                     const int depth,
                     const unsigned char* x,
                     const float* bins,
                     const float *grad,
                     const float *hessian,
                     const bool *active_idx,
                     const int *feature_idx,
                     const int max_bin,
                     const int num_samples,
                     const int num_frac_features,
                     const int num_total_features,
                     const int max_depth,
                     const int min_data_in_leaf
                     ) {
    /* printf("max_bin %d \n", max_bin); */
    /* printf("num_samples %d \n", num_samples); */
    /* printf("num_frac_features %d \n", num_frac_features); */
    /* printf("num_total_features %d \n", num_total_features); */
    /* printf("max_depth %d \n", max_depth); */

    /* for (int di = 0; di < num_samples; ++di) */
    /* for (int fi = 0 ; fi < num_frac_features; ++fi) { */
    /* if (x[di*num_total_features + feature_idx[fi]] >= max_bin) */
    /* printf("%d %d\n", di, fi); */
    /* } */
    /* print_char(x, num_samples*num_total_features); */
    /* print_bool(active_idx, num_samples); */
    int num_active_samples = 0;
    float grad_sum = 0., hessian_sum = 0.;
    for (int i = 0; i < num_samples; ++i) {
        if (active_idx[i]) {
            num_active_samples += 1;
            grad_sum += grad[i];
            hessian_sum += hessian[i];
        }
    }

    /* printf("%d %d \n", node_id, num_active_samples); */
    /* printf("grad_sum % 3f \n", grad_sum); */
    if ((num_active_samples <= min_data_in_leaf) || (depth >= max_depth)) {
        tree[5*node_id] = -1;
        tree[5*node_id + 1] = NAN;
        tree[5*node_id + 2] = -grad_sum / hessian_sum;
        tree[5*node_id + 3] = NAN;
        tree[5*node_id + 4] = NAN;
        return node_id + 1;
    }

    // grad_left and hessian_left:  num_frac_features x num_bins
    float *grad_left, *hessian_left;
    grad_left = malloc(num_frac_features*max_bin*sizeof(float));
    hessian_left = malloc(num_frac_features*max_bin*sizeof(float));
    memset(grad_left, 0, num_frac_features*max_bin*sizeof(float));
    memset(hessian_left, 0, num_frac_features*max_bin*sizeof(float));
    // access x, grad, hessian in sequential order
    // access grad_left, hessian_left in feature wise order
    //                                bin order is random

    /* print(grad_left, num_frac_features * max_bin, max_bin); */
    for (int di = 0; di < num_samples; ++di) {
        if(active_idx[di]) {
            for (int fi = 0; fi < num_frac_features; ++fi) {
                // use unsigned char directly as int
                /* printf("%d %d %d\n", di, feature_idx[fi], x[di*num_total_features + feature_idx[fi]]); */
                grad_left[fi*max_bin + (int)x[di*num_total_features + feature_idx[fi]]] += grad[di];
                hessian_left[fi*max_bin + (int)x[di*num_total_features + feature_idx[fi]]] += hessian[di];
            }
        }
    }



    // calculate cumsum
    for (int fi = 0; fi < num_frac_features; ++fi) {
        float grad_cumsum = 0., hessian_cumsum = 0.;
        for (int bin = 0; bin < max_bin; ++bin) {
            grad_cumsum += grad_left[fi*max_bin + bin];
            grad_left[fi*max_bin + bin] = grad_cumsum;
            hessian_cumsum += hessian_left[fi*max_bin + bin];
            hessian_left[fi*max_bin + bin] = hessian_cumsum;
        }
    }


    /* print(grad_left, num_frac_features * max_bin, max_bin); */
    /* print(hessian_left, num_frac_features * max_bin, max_bin); */


    float* best_score_across_split = malloc(num_frac_features * sizeof(float));
    int* best_split = malloc(num_frac_features * sizeof(int));
    memset(best_split, -1, num_frac_features*sizeof(int));
    for (int fi = 0; fi < num_frac_features; ++fi)
        best_score_across_split[fi] = -INFINITY;
    /* print(best_score_across_split, num_frac_features, num_frac_features); */
    for (int fi = 0; fi < num_frac_features; ++fi) {
        for (int bin = 0; bin < max_bin; ++bin) {
            float grad_right = grad_sum - grad_left[fi*max_bin + bin];
            float hessian_right = hessian_sum - hessian_left[fi*max_bin + bin];
            float score = grad_left[fi*max_bin + bin] * grad_left[fi*max_bin + bin] / (hessian_left[fi*max_bin + bin] + 1e-8) + grad_right * grad_right / (hessian_right + 1e-8);
            /* printf("%d %d %.3f \n", fi, bin, score); */
            if (score > best_score_across_split[fi]){
                best_score_across_split[fi] = score;
                best_split[fi] = bin;
            }
        }
    }
      
    /* printf("grad_left "); */
    /* printf("hessian_left "); */
    /* print(hessian_left, num_frac_features * max_bin, max_bin); */
    // free grad_left, hessian_left, score
    free(grad_left);
    free(hessian_left);
    
    float best_score_across_feature = -INFINITY;
    int best_feature_idx = 0, best_feature = 0, best_feature_threshold = 0;
    double best_feature_threshold_real = NAN;
    for (int fi = 0; fi < num_frac_features; ++fi) {
        if (best_score_across_split[fi] > best_score_across_feature) {
            best_score_across_feature = best_score_across_split[fi];
            best_feature_idx = fi;
        }
    }
    best_feature = feature_idx[best_feature_idx];
    best_feature_threshold = best_split[best_feature_idx];
    best_feature_threshold_real = bins[best_feature_threshold*num_total_features + best_feature];
    // printf("%.3f\n", best_score_across_feature);
    // printf("%d %.3f %d %d \n", node_id, best_score_across_feature, best_feature, best_feature_threshold);

    // free array
    free(best_score_across_split);
    free(best_split);


    bool *active_idx_left = malloc(num_samples * sizeof(bool)), *active_idx_right = malloc(num_samples * sizeof(bool));
    int num_active_samples_left = 0, num_active_samples_right = 0;
    for(int di = 0; di < num_samples; ++di) {
        active_idx_left[di] = active_idx[di] && (x[di*num_total_features + best_feature] <= best_feature_threshold);
        if (active_idx_left[di]) num_active_samples_left += 1;
        active_idx_right[di] = active_idx[di] && (x[di*num_total_features + best_feature] > best_feature_threshold);
        if (active_idx_right[di]) num_active_samples_right +=1;
    }


    /* printf("%d %.3f %d %d %d \n", node_id, best_score_across_feature, best_feature, num_active_samples_left, num_active_samples_right); */

    /* printf("left and right %d %d \n", num_active_samples_left, num_active_samples_right); */
    if ((num_active_samples_left == 0) || (num_active_samples_right == 0)) {
        tree[5*node_id] = -1;
        tree[5*node_id + 1] = NAN;
        tree[5*node_id + 2] = -grad_sum / hessian_sum;
        tree[5*node_id + 3] = NAN;
        tree[5*node_id + 4] = NAN;

        free(active_idx_left);
        free(active_idx_right);

        return node_id + 1;
    } else{
        tree[5*node_id] = best_feature;
        tree[5*node_id + 1] = best_feature_threshold_real;
        tree[5*node_id + 2] = NAN;
        tree[5*node_id + 3] = node_id + 1;
        tree[5*node_id + 4] = NAN;
        
        int node_id_right = build_tree_float(tree,
                                             node_id + 1,
                                             depth + 1,
                                             x,
                                             bins,
                                             grad,
                                             hessian,
                                             active_idx_left,
                                             feature_idx,
                                             max_bin,
                                             num_samples,
                                             num_frac_features,
                                             num_total_features,
                                             max_depth,
                                             min_data_in_leaf
                                             );
        tree[5*node_id + 4] = node_id_right;
        int node_id_next = build_tree_float(tree, node_id_right,
                                            depth + 1,
                                            x,
                                            bins,
                                            grad,
                                            hessian,
                                            active_idx_right,
                                            feature_idx,
                                            max_bin,
                                            num_samples,
                                            num_frac_features,
                                            num_total_features,
                                            max_depth,
                                            min_data_in_leaf
                                            );
        free(active_idx_left);
        free(active_idx_right);

        return node_id_next;
    }
}



int build_tree_double(double *tree,
                      const int node_id,
                      const int depth,
                      const unsigned char* x,
                      const double *bins,
                      const double *grad,
                      const double *hessian,
                      const bool *active_idx,
                      const int *feature_idx,
                      const int max_bin,
                      const int num_samples,
                      const int num_frac_features,
                      const int num_total_features,
                      const int max_depth,
                      const int min_data_in_leaf
                      ) {
    /* printf("max_bin %d\n", max_bin); */
    /* printf("num_samples %d\n", num_samples); */
    /* printf("num_frac_features %d\n", num_frac_features); */
    /* printf("num_total_features %d\n", num_total_features); */
    /* printf("max_depth %d \n", max_depth); */

    /*    for (int di = 0; di < num_samples; ++di) */
    /*        for (int fi = 0; fi < num_frac_features; ++fi) { */
    /*            if (x[di*num_total_features + feature_idx[fi]] >= max_bin) */
    /*                printf("%d %d\n", di, fi); */
    /*        } */
    /* print_char(x, num_samples*num_total_features); */
    /* print_bool(active_idx, num_samples); */
    int num_active_samples = 0;
    double grad_sum = 0., hessian_sum = 0.;
    for (int i = 0; i < num_samples; ++i) {
        if (active_idx[i]) {
            num_active_samples += 1;
            grad_sum += grad[i];
            hessian_sum += hessian[i];
        }
    }

    /* printf("%d %d \n", node_id, num_active_samples); */
    /* printf("grad_sum %.3f \n", grad_sum); */
    if ((num_active_samples <= min_data_in_leaf) || (depth >= max_depth)) {
        tree[5*node_id] = -1;
        tree[5*node_id + 1] = NAN;
        tree[5*node_id + 2] = -grad_sum / hessian_sum;
        tree[5*node_id + 3] = NAN;
        tree[5*node_id + 4] = NAN;
        return node_id + 1;
    }

    // grad_left and hessian_left: num_frac_features x num_bins
    double *grad_left, *hessian_left;
    grad_left = malloc(num_frac_features*max_bin*sizeof(double));
    hessian_left = malloc(num_frac_features*max_bin*sizeof(double));
    memset(grad_left, 0, num_frac_features*max_bin*sizeof(double));
    memset(hessian_left, 0, num_frac_features*max_bin*sizeof(double));
    // access x, grad, hessian in sequential order
    // access grad_left, hessian_left in feature wise order
    //                                bin order is random

    /* print(grad_left, num_frac_features * max_bin, max_bin); */
    for (int di = 0; di < num_samples; ++di) {
        if(active_idx[di]) {
            for (int fi = 0; fi < num_frac_features; ++fi) {
                // use unsigned char directly as int
                /* printf("%d %d %d\n", di, feature_idx[fi], x[di*num_total_features + feature_idx[fi]]); */
                grad_left[fi*max_bin + (int)x[di*num_total_features + feature_idx[fi]]] += grad[di];
                hessian_left[fi*max_bin + (int)x[di*num_total_features + feature_idx[fi]]] += hessian[di];
            }
        }
    }


    // calculate cumsum
    for (int fi = 0; fi < num_frac_features; ++fi) {
        double grad_cumsum = 0., hessian_cumsum = 0.;
        for (int bin = 0; bin < max_bin; ++bin) {
            grad_cumsum += grad_left[fi*max_bin + bin];
            grad_left[fi*max_bin + bin] = grad_cumsum;
            hessian_cumsum += hessian_left[fi*max_bin + bin];
            hessian_left[fi*max_bin + bin] = hessian_cumsum;
        }
    }


    /* print(grad_left, num_frac_features * max_bin, max_bin); */
    /* print(hessian_left, num_frac_features * max_bin, max_bin); */


    double* best_score_across_split = malloc(num_frac_features * sizeof(double));
    int* best_split = malloc(num_frac_features * sizeof(int));
    memset(best_split, -1, num_frac_features*sizeof(int));
    for (int fi = 0; fi < num_frac_features; ++fi)
        best_score_across_split[fi] = -INFINITY;
    /* print(best_score_across_split, num_frac_features, num_frac_features); */

    for (int fi = 0; fi < num_frac_features; ++fi) {
        for (int bin = 0; bin < max_bin; ++bin) {
            double grad_right = grad_sum - grad_left[fi*max_bin + bin];
            double hessian_right = hessian_sum - hessian_left[fi*max_bin + bin];
            double score = grad_left[fi*max_bin+bin]*grad_left[fi*max_bin + bin] / (hessian_left[fi*max_bin + bin] + 1e-8) + grad_right * grad_right / (hessian_right + 1e-8);
            /* printf("%d %d %f %f\n", fi, bin, score); */
            if (score > best_score_across_split[fi]) {
                best_score_across_split[fi] = score;
                best_split[fi] = bin;
            }
        }
    }

    /* printf("grad_left "); */
    /* printf("hessian_left "); */
    /* print(hessian_left, num_frac_features * max_bin, max_bin); */
    // free grad_left, hessian_left, score
    free(grad_left);
    free(hessian_left);

    double best_score_across_feature = -INFINITY;
    int best_feature_idx = 0, best_feature = 0, best_feature_threshold = 0;
    double best_feature_threshold_real = NAN;
    for (int fi = 0; fi < num_frac_features; ++fi) {
        if (best_score_across_split[fi] > best_score_across_feature) {
            best_score_across_feature = best_score_across_split[fi];
            best_feature_idx = fi;
        }
    }
    best_feature = feature_idx[best_feature_idx];
    best_feature_threshold = best_split[best_feature_idx];
    best_feature_threshold_real = bins[best_feature_threshold*num_total_features + best_feature];
    // printf("%.3f\n", best_score_across_feature);
    // printf("%d %.3f %d %d \n", node_id, best_score_across_feature, best_feature, best_feature_threshold);
    
    // free array
    free(best_score_across_split);
    free(best_split);
    

    bool *active_idx_left = malloc(num_samples * sizeof(bool)), *active_idx_right = malloc(num_samples * sizeof(bool));
    int num_active_samples_left = 0, num_active_samples_right = 0;
    for(int di = 0; di < num_samples; ++di) {
        active_idx_left[di] = active_idx[di] && (x[di*num_total_features + best_feature] <= best_feature_threshold);
        if (active_idx_left[di]) num_active_samples_left += 1;
        active_idx_right[di] = active_idx[di] && (x[di*num_total_features + best_feature] > best_feature_threshold);
        if (active_idx_right[di]) num_active_samples_right +=1;
    }
    
    /* printf("%d %.3f %d %d %d \n", node_id, best_score_across_feature, best_feature, num_active_samples_left, num_active_samples_right); */      



    if ((num_active_samples_left == 0) || (num_active_samples_right == 0)) {
        tree[5*node_id] = -1;
        tree[5*node_id + 1] = NAN;
        tree[5*node_id + 2] = -grad_sum / hessian_sum;
        tree[5*node_id + 3] = NAN;
        tree[5*node_id + 4] = NAN;
        
        free(active_idx_left);
        free(active_idx_right);
        
        return node_id + 1;
    } else {
        tree[5*node_id] = best_feature;
        tree[5*node_id + 1] = best_feature_threshold_real;
        tree[5*node_id + 2] = NAN;
        tree[5*node_id + 3] = node_id + 1;
        tree[5*node_id + 4] = NAN;
        
        int node_id_right = build_tree_double(tree,
                                              node_id + 1,
                                              depth + 1,
                                              x,
                                              bins,
                                              grad,
                                              hessian,
                                              active_idx_left,
                                              feature_idx,
                                              max_bin,
                                              num_samples,
                                              num_frac_features,
                                              num_total_features,
                                              max_depth,
                                              min_data_in_leaf
                                            );
        tree[5*node_id + 4] = node_id_right;
        int node_id_next = build_tree_double(tree,
                                            node_id_right,
                                            depth + 1,
                                            x,
                                            bins,
                                            grad,
                                            hessian,
                                            active_idx_right,
                                            feature_idx,
                                            max_bin,
                                            num_samples,
                                            num_frac_features,
                                            num_total_features,
                                            max_depth,
                                            min_data_in_leaf
                                            );
    free(active_idx_left);
    free(active_idx_right);
    
    return node_id_next;
    }
}


















































































































































































































void train(const unsigned char *restrict x,
    const double *restrict x_origin,
    const double *restrict bins,
    const double *restrict y,
    double *restrict tree,
    int num_samples,
    int num_features,
    int num_trees,
    int max_depth,
    int max_bin,
    int min_data_in_leaf,
    double bagging_fraction,
    double feature_fraction,
    double learning_rate
) {

    assert(num_samples >= 1);
    assert(num_features >= 1);
    assert(bagging_fraction > 0);
    assert(feature_fraction > 0);
    assert(learning_rate > 0);
    assert(max_depth > 0);
    int num_nodes_per_tree = (int)pow(2., max_depth + 1) - 1;
    double *yhat = malloc(num_samples*sizeof(double));
    double *yi = malloc(num_samples*sizeof(double));
    double *grad= malloc(num_samples*sizeof(double));
    double *hessian = malloc(num_samples*sizeof(double));
    bool *active_idx = malloc(num_samples*sizeof(bool));
    int *feature_idx = malloc(num_features*sizeof(int));
    memset(yhat, 0, num_samples*sizeof(double));
    memset(yi, 0, num_samples*sizeof(double));
    memset(grad, 0, num_samples*sizeof(double));
    for (int di = 0; di < num_samples; ++di)
        hessian[di] = 1;

    for (int i = 0; i < num_trees * num_nodes_per_tree * 5; ++i)
        tree[i] = NAN;
    
    for (int tree_idx = 0; tree_idx < num_trees; ++tree_idx) {
        /* double grad_square_sum = 0; */
        for (int di = 0; di < num_samples; ++di) {
            grad[di] = yhat[di] - y[di];
            /* grad_square_sum += grad[di]*grad[di]; */
        }
        /* printf("%.3f\n", sqrt(grad_square_sum/num_samples)); */
        
        double *tree_i = tree + tree_idx * num_nodes_per_tree * 5;
        // sample active
        
        int num_active_samples = 0;
        for (int di = 0; di < num_samples; ++di) {
            double r = ((double)rand()/RAND_MAX);
            if (r <= bagging_fraction) {
                active_idx[di] = true;
                num_active_samples++;
            }
            else
                active_idx[di] = false;
        }
        if (num_active_samples == 0) {
            printf("bagging_fraction is too small, randomly select 1 sample");
            active_idx[rand()%num_samples] = true;
            num_active_samples = 1;
        }

        int num_frac_features = 0;
        for (int fi = 0; fi < num_features; ++fi) {
            double r = ((double)rand()/RAND_MAX);
            if (r <= feature_fraction) {
                feature_idx[num_frac_features] = fi;
                num_frac_features++;
            }
        }
        
        if (num_frac_features == 0) {
            printf("feature_fraction is too small, randomly select 1 feature");
            feature_idx[0] = rand() % num_features;
            num_frac_features = 1;
        }
        

        build_tree_double(tree_i,
                          0,
                          0,
                          x,
                          bins,
                          grad,
                          hessian,
                          active_idx,
                          feature_idx,
                          max_bin,
                          num_samples,
                          num_frac_features,
                          num_features,
                          max_depth,
                          min_data_in_leaf
                          );
        /* print_double(tree_i, num_nodes_per_tree*5, 5); */

        // prediction
        memset(yi, 0, num_samples*sizeof(double));
        predict(x_origin,
                tree_i,
                yi,
                num_samples,
                num_features,
                num_trees,
                num_nodes_per_tree,
                learning_rate,
                0,
                1
                );

        for (int di = 0; di < num_samples; ++di) {
            yhat[di] += yi[di];
        }
    }

    free(yhat);
    free(yi);
    free(grad);
    free(hessian);
    free(active_idx);
    free(feature_idx);

}
































































































































void predict(const double *restrict x,
    const double *restrict tree,
    double* restrict y_pred,
    int num_samples,
    int num_features,
    int num_trees,
    int num_nodes_per_tree,
    double learning_rate,
    int start_tree_idx,
    int end_tree_idx
) {

    memset(y_pred, 0, num_samples*sizeof(double));
    if (end_tree_idx == -1)
        end_tree_idx = num_trees;

    for (int sample_idx = 0; sample_idx < num_samples; ++sample_idx) {
        for (int tree_idx = start_tree_idx; tree_idx < end_tree_idx; ++tree_idx) {
            const double *tree_i = tree + tree_idx * num_nodes_per_tree * 5;
            int current_node = 0;
            while ((int)tree_i[current_node*5] != -1) {
                assert(!isnan(tree_i[current_node*5]));
                int feature_idx = (int)tree_i[current_node*5];
                assert((feature_idx >= 0) && (feature_idx < num_features));
                current_node = (x[sample_idx * num_features + feature_idx] <= tree_i[current_node*5+1]) ? (int)tree_i[current_node*5+3] : (int)tree_i[current_node*5+4];
                assert((current_node >=0) && (current_node < num_nodes_per_tree));
            }
            y_pred[sample_idx] += learning_rate * tree_i[current_node*5+2];
        }
    }
}

int build_tree_double_custom(double *tree,
    const int node_id,
    const int depth,
    const unsigned char* x,
    const double *bins,
    const double *x_grad,
    const double *xx_hessian,
    const bool *cond_active_idx,
    const bool *pred_active_idx,
    const int *cond_feature_idx,
    const int *pred_feature_idx,
    const int cond_depth_sep,
    const int pred_depth_sep,
    const int depth_sep,
    const int max_bin,
    const int num_samples,
    const int num_frac_features_cond,
    const int num_frac_features_pred,
    const int num_total_features,
    const int max_depth,
    const int min_data_in_leaf,
    const int leaf_use_all_pred_feature,
    const int use_delta,
    const int cond_shallow_use_prior_only,
    const int score_agg,
    const double min_gain_to_split,
    const double lambda_l2,
    const int normalize,
    const int split_sample,
    const int debug
) {
    /* printf("max_bin %d \n", max_bin); */
    /* printf("num_samples %d \n", num_samples); */
    /* printf("num_frac_features %d \n", num_frac_features); */
    /* printf("num_total_features %d \n", num_total_features); */
    /* printf("max_depth %d \n", max_depth); */

    /*  for (int di = 0; di < num_samples; ++di) */
    /*    for (int fi = 0 ; fi < num_frac_features; ++fi) { */
    /*      if (x[di*num_total_features + feature_idx[fi]] >= max_bin) */
    /*        printf("%d %d\n", di, fi); */
    /*  } */
    /*  print_char(x, num_samples*num_total_features); */
    /*  print_bool(active_idx, num_samples); */
    int num_active_samples = 0;
    double* grad_sum = malloc(num_frac_features_pred*sizeof(double));
    double* hessian_sum = malloc(num_frac_features_pred*sizeof(double));
    memset(grad_sum, 0, num_frac_features_pred*sizeof(double));
    memset(hessian_sum, 0, num_frac_features_pred*sizeof(double));

    for (int di = 0; di < num_samples; ++di) {
        if(pred_active_idx[di]) {
            for (int pi = 0; pi < num_frac_features_pred; ++pi) {
                grad_sum[pi] += x_grad[di*num_total_features + pred_feature_idx[pi]];
                hessian_sum[pi] += xx_hessian[di*num_total_features + pred_feature_idx[pi]];
            }
            num_active_samples++;
        }
    }
    if (debug)
        printf("node_id %d depth %d %d \n",
                node_id, depth, num_active_samples);
    /* printf("grad_sum %.3f \n", grad_sum); */
    /* printf("grad_sum\n"); */
    if (debug >= 2) {
        printf("grad and hessian sum\n");
        print_double(grad_sum, num_frac_features_pred, num_frac_features_pred);
        print_double(hessian_sum, num_frac_features_pred, num_frac_features_pred);
    }


    double score = -INFINITY;
    double leaf_weight = NAN;
    int best_feature_idx_pred_leaf = 0;
    int pi_start = 0, pi_end = num_frac_features_pred;
    if (!leaf_use_all_pred_feature) {
        pi_start = pred_depth_sep;
        /* printf("leaf not using all features\n"); */
    }
    for (int pi = pi_start; pi < pi_end; ++pi) {
        double score_pi = 0;
        if (normalize)
            score_pi = grad_sum[pi] * grad_sum[pi] / (hessian_sum[pi] * (lambda_l2 + 1) + 1e-8);
        else
            score_pi = grad_sum[pi] * grad_sum[pi] / (hessian_sum[pi] + lambda_l2 + 1e-8);
        if (score_pi > score ) {
            score = score_pi;
            best_feature_idx_pred_leaf = pi;
            if (normalize)
                leaf_weight = -grad_sum[pi] / (hessian_sum[pi] * (lambda_l2 + 1) + 1e-8);
            else
                leaf_weight = -grad_sum[pi] / (hessian_sum[pi] + lambda_l2 + 1e-8);
        }
    }




    free(grad_sum);
    free(hessian_sum);

    if ((num_active_samples <= min_data_in_leaf) || (depth >= max_depth)) {
        tree[5*node_id] = pred_feature_idx[best_feature_idx_pred_leaf];
        tree[5*node_id + 1] = NAN;
        tree[5*node_id + 2] = leaf_weight;
        tree[5*node_id + 3] = NAN;
        tree[5*node_id + 4] = NAN;
        if (debug)
            printf("best_pred_feature %d leaf_weight %.3f\n",
                pred_feature_idx[best_feature_idx_pred_leaf],
                leaf_weight);
        
        return node_id + 1;
    }

    // grad_left and hessian_left: num_frac_features x num_bins
    double *grad_left, *hessian_left;
    grad_left = malloc(num_frac_features_cond*num_frac_features_pred*max_bin*sizeof(double));
    hessian_left = malloc(num_frac_features_cond*num_frac_features_pred*max_bin*sizeof(double));
    memset(grad_left, 0, num_frac_features_cond*num_frac_features_pred*max_bin*sizeof(double));
    memset(hessian_left, 0, num_frac_features_cond*num_frac_features_pred*max_bin*sizeof(double));
    // access grad, hessian in sequential order
    // access grad_left, hessian_left in feature wise order
    //                          bin order is random

    /* print(grad_left, num_frac_features * max_bin, max_bin); */
    int ci_start = 0, ci_end = num_frac_features_cond;
    bool* active_idx = cond_active_idx;
    pi_start = 0; pi_end = num_frac_features_pred;
    if (depth < depth_sep) {
        if (split_sample)
            active_idx = cond_active_idx;
        ci_start = 0;
        ci_end = cond_depth_sep;
        pi_start = 0;
        // pi_end = num_frac_features_pred;
        pi_end = pred_depth_sep;
    } else {
        if (split_sample)
            active_idx = pred_active_idx;
        // if cond_shallow_use_prior_only == 2:
        // depth < depth_sep: [0, cond_depth_sep]
        // depth >= depth_sep: [cond_depth_sep, num_frac_features_cond]
        // else
        // depth < depth_sep: [0, cond_depth_sep]
        // depth >= depth_sep: [0, num_frac_features_cond]
        if (cond_shallow_use_prior_only == 2)
            ci_start = cond_depth_sep;
        else
            ci_start = 0;
        ci_end = num_frac_features_cond;
        pi_start = pred_depth_sep;
        pi_end = num_frac_features_pred;
    }
    if (debug >= 2) {
        printf("active_idx at depth %d \n", depth);
        print_bool(cond_active_idx, num_samples);
        print_bool(pred_active_idx, num_samples);
        print_bool(active_idx, num_samples);
    }
    
    for (int di = 0; di < num_samples; ++di) {
        if(active_idx[di]) {
            for (int ci = ci_start; ci < ci_end; ++ci) {
                for (int pi = pi_start; pi < pi_end; ++pi) {
                    // use unsigned char directly as int
                    /* printf("%d %d %d\n", di, feature_idx[fi], x[di*num_total_features + feature_idx[fi]]); */
                    grad_left[ci*num_frac_features_pred*max_bin + pi*max_bin
                            + (int)x[di*num_total_features + cond_feature_idx[ci]]] \
                        += x_grad[di*num_total_features + pred_feature_idx[pi]];
                    hessian_left[ci*num_frac_features_pred*max_bin + pi*max_bin
                            + (int)x[di*num_total_features + cond_feature_idx[ci]]] \
                        += xx_hessian[di*num_total_features + pred_feature_idx[pi]];
                }
            }
        }
    }
    
    /* if (debug >= 2) { */
    /*     printf("grad_left\n"); */
    /*     print_double(grad_left, num_frac_features_cond*num_frac_features_pred*max_bin, max_bin); */
    /*     printf("hessian_left\n"); */
    /*     print_double(hessian_left, num_frac_features_cond*num_frac_features_pred*max_bin, max_bin); */
    /* } */

    // calculate cumsum
    for (int ci = ci_start; ci < ci_end; ++ci) {
        for (int pi = pi_start; pi < pi_end; ++pi) {
            double grad_cumsum = 0., hessian_cumsum = 0.;
            for (int bin = 0; bin < max_bin; ++bin) {
                grad_cumsum += grad_left[ci*num_frac_features_pred*max_bin + pi*max_bin + bin];
                grad_left[ci*num_frac_features_pred*max_bin + pi*max_bin + bin] = grad_cumsum;
                hessian_cumsum += hessian_left[ci*num_frac_features_pred*max_bin + pi*max_bin + bin];
                hessian_left[ci*num_frac_features_pred*max_bin + pi*max_bin + bin] = hessian_cumsum;
            }
        }
    }

  

    /* print(grad_left, num_frac_features * max_bin, max_bin); */
    /* print(hessian_left, num_frac_features * max_bin, max_bin); */


    double* score_all = malloc(num_frac_features_cond * num_frac_features_pred * max_bin * sizeof(double));
    double* best_score_across_split = malloc(num_frac_features_cond * num_frac_features_pred * sizeof(double));
    int* best_split = malloc(num_frac_features_cond * num_frac_features_pred * sizeof(int));
    memset(best_split, -1, num_frac_features_cond * num_frac_features_pred*sizeof(int));
    for(int i = 0; i < num_frac_features_cond*num_frac_features_pred; ++i)
        best_score_across_split[i] = -INFINITY;
    /* print(best_score_across_split, num_frac_features, num_frac_features); */

    for (int ci = ci_start; ci < ci_end; ++ci) {
        for (int pi = pi_start; pi < pi_end; ++pi) {
            int idx_end = ci*num_frac_features_pred*max_bin + pi*max_bin + max_bin - 1;
            double score_base = 0.;
            if (normalize)
                score_base = grad_left[idx_end]*grad_left[idx_end]/(hessian_left[idx_end] * (lambda_l2 + 1) + 1e-8);
            else
                score_base = grad_left[idx_end]*grad_left[idx_end]/(hessian_left[idx_end] + lambda_l2 + 1e-8);
            for (int bin = 0; bin < max_bin; ++bin) {
                int idx_start = ci*num_frac_features_pred*max_bin + pi*max_bin + bin;
                double grad_right = grad_left[idx_end] - grad_left[idx_start];
                double hessian_right = hessian_left[idx_end] - hessian_left[idx_start];
                double score = 0;
                if (normalize)
                    score = grad_left[idx_start]*grad_left[idx_start] / (hessian_left[idx_start] * (lambda_l2 + 1) + 1e-8) + grad_right * grad_right / (hessian_right * (lambda_l2 + 1) + 1e-8);
                else
                    score = grad_left[idx_start]*grad_left[idx_start] / (hessian_left[idx_start] + lambda_l2 + 1e-8) + grad_right * grad_right / (hessian_right + lambda_l2 + 1e-8);
                if (use_delta)
                    score -= score_base;
                
                score_all[idx_start] = score;
                /* printf("%d %d %.3f \n", fi, bin, score); */
                /* printf("%.3f ", score); */
                /* if (score > best_score_across_split[ci*num_frac_features_pred + pi]) { */
                /*   best_score_across_split[ci*num_frac_features_pred + pi] = score; */
                /*   best_split[ci*num_frac_features_pred + pi] = bin; */
                /* } */
            }
            /* printf("\n "); */
        }
    }
    
    /* if (debug >= 3) */
    /*   print_double(score_all, num_frac_features_cond*num_frac_features_pred*max_bin, max_bin); */

    if (score_agg == 0) {
        // max across the bins
        for (int ci = ci_start; ci < ci_end; ++ci)
            for (int pi = pi_start; pi < pi_end; ++pi)
                for (int bin = 0; bin < max_bin; ++bin) {
                    int idx_start = ci*num_frac_features_pred*max_bin + pi*max_bin + bin;
                    double score = score_all[idx_start];
                    if (score > best_score_across_split[ci*num_frac_features_pred + pi]) {
                        best_score_across_split[ci*num_frac_features_pred + pi] = score;
                        best_split[ci*num_frac_features_pred + pi] = bin;
                    }
                }

    } else if (score_agg == 1) {
        // avg across pred features, then take max across bin
        for (int ci = ci_start; ci < ci_end; ++ci) {
            for (int bin = 0; bin < max_bin; ++bin) {
                double score_total = 0;
                for (int pi = pi_start; pi < pi_end; ++pi)
                    score_total += score_all[ci*num_frac_features_pred*max_bin + pi*max_bin + bin]/(pi_end - pi_start + 1e-8);
                /* if (debug >= 2) */
                /*   printf("%.3f ", score_total); */
                if (score_total > best_score_across_split[ci*num_frac_features_pred + 0]) {
                    best_score_across_split[ci*num_frac_features_pred + 0] = score_total;
                    best_split[ci*num_frac_features_pred + 0] = bin;
                }
            }
        }
    } else if (score_agg == 2) {
        double *compare_array = malloc(num_frac_features_pred*sizeof(double));
        for (int ci = ci_start; ci < ci_end; ++ci) {
            for (int bin = 0; bin < max_bin; ++bin) {
                int pi_count = 0;
                for (int pi = pi_start; pi < pi_end; ++pi) {
                    compare_array[pi_count] = score_all[ci*num_frac_features_pred*max_bin + pi*max_bin + bin];
                    pi_count += 1;
                }
                if (debug >= 2) {
                    print_double(compare_array, pi_end-pi_start, pi_count);
                }
                double score_median = find_median(compare_array, pi_count);
                if (debug >= 2) {
                    print_double(compare_array, pi_end-pi_start, pi_count);
                    printf("median %.3f \n", score_median);
                }
                if (score_median > best_score_across_split[ci*num_frac_features_pred + 0]) {
                    best_score_across_split[ci*num_frac_features_pred + 0] = score_median;
                    best_split[ci*num_frac_features_pred + 0] = bin;
                }
            }
        }
        free(compare_array);
    } else if (score_agg == 3) {
        // random pred features, then take max across bin
        for (int ci = ci_start; ci < ci_end; ++ci) {
            for (int bin = 0; bin < max_bin; ++bin) {
                int pi = rand()%(pi_end - pi_start) + pi_start;
                double score_pi_rand = score_all[ci*num_frac_features_pred*max_bin + pi*max_bin + bin];
                if (score_pi_rand > best_score_across_split[ci*num_frac_features_pred + 0]) {
                    best_score_across_split[ci*num_frac_features_pred + 0] = score_pi_rand;
                    best_split[ci*num_frac_features_pred + 0] = bin;
                }
            }
        }
    } else if (score_agg == 10) {
        // random select bin
        for (int ci = ci_start; ci < ci_end; ++ci)
            for (int pi = pi_start; pi < pi_end; ++pi) {
                int bin = rand() % max_bin;
                int idx_start = ci*num_frac_features_pred*max_bin + pi*max_bin + bin;
                double score = score_all[idx_start];
                best_score_across_split[ci*num_frac_features_pred + pi] = score;
                best_split[ci*num_frac_features_pred + pi] = bin;
            }
    } else if (score_agg == 11) {
        // avg across pred features, then take random across bin
        for (int ci = ci_start; ci < ci_end; ++ci) {
            int bin = rand() % max_bin;
            double score_total = 0;
            for (int pi = pi_start; pi < pi_end; ++pi)
                score_total += score_all[ci*num_frac_features_pred*max_bin + pi*max_bin + bin]/(pi_end - pi_start + 1e-8);
            best_score_across_split[ci*num_frac_features_pred + 0] = score_total;
            best_split[ci*num_frac_features_pred + 0] = bin;
        }
    } else if (score_agg == 12) {
        double *compare_array = malloc(num_frac_features_pred*sizeof(double));
        for (int ci = ci_start; ci < ci_end; ++ci) {
            int bin = rand() % max_bin;
            int pi_count = 0;
            for (int pi = pi_start; pi < pi_end; ++pi) {
                compare_array[pi_count] = score_all[ci*num_frac_features_pred*max_bin + pi*max_bin + bin];
                pi_count += 1;
            }
            double score_median = find_median(compare_array, pi_count);
            best_score_across_split[ci*num_frac_features_pred + 0] = score_median;
            best_split[ci*num_frac_features_pred + 0] = bin;
        }
        free(compare_array);
    } if (score_agg == 13) {
        // random pred features, then take max across bin
        for (int ci = ci_start; ci < ci_end; ++ci) {
            int bin = rand() % max_bin;
            int pi = rand()%(pi_end - pi_start) + pi_start;
            double score_pi_bin_rand = score_all[ci*num_frac_features_pred*max_bin + pi*max_bin + bin];
            best_score_across_split[ci*num_frac_features_pred + 0] = score_pi_bin_rand;
            best_split[ci*num_frac_features_pred + 0] = bin;
        }
    }


    if (debug >= 2) {
        print_double(best_score_across_split,
                    num_frac_features_cond*num_frac_features_pred,
                    num_frac_features_pred);
        print_int(best_split,
                num_frac_features_cond*num_frac_features_pred,
                num_frac_features_pred);
    }
    

        
    /* printf("best_score_across_split \n"); */
    /* print_double(best_score_across_split, num_frac_features_cond*num_frac_features_pred, num_frac_features_pred); */
    /* printf("best_split \n"); */
    /* print_int(best_split, num_frac_features_cond*num_frac_features_pred, num_frac_features_pred); */
    
    

    /* printf("grad_left "); */
    /* printf("hessian_left "); */
    /* print(hessian_left, num_frac_features * max_bin, max_bin); */
    /* free grad_left, hessian_left, score */
    free(grad_left);
    free(hessian_left);

    double best_score_across_feature = -INFINITY;
    int best_feature_idx_cond = 0, best_feature_idx_pred = 0, best_feature_cond = 0, best_feature_pred = 0, best_feature_cond_threshold = 0;
    double best_feature_cond_threshold_real = NAN;
    if ((score_agg == 0) || (score_agg == 10)) {
        for (int ci = ci_start; ci < ci_end; ++ci) {
            for (int pi = pi_start; pi < pi_end; ++pi) {
                /* for (int pi = num_frac_features_pred - 1; pi < num_frac_features_pred; ++pi) { */
                if (best_score_across_split[ci*num_frac_features_pred + pi] > best_score_across_feature) {
                    best_score_across_feature = best_score_across_split[ci*num_frac_features_pred + pi];
                    best_feature_idx_cond = ci;
                    best_feature_idx_pred = pi;
                }
            }
        }
    } else if ((score_agg == 1) || (score_agg == 2) || (score_agg == 3) || (score_agg == 11) || (score_agg == 12) || (score_agg == 13)) {
        for (int ci = ci_start; ci < ci_end; ++ci) {
            if (best_score_across_split[ci*num_frac_features_pred + 0] > best_score_across_feature) {
                best_score_across_feature = best_score_across_split[ci*num_frac_features_pred + 0];
                best_feature_idx_cond = ci;
            }
        }
    }
    /* printf("best_feature_idx_cond %d\n", best_feature_idx_cond); */
    /* printf("best_feature_idx_pred %d\n", best_feature_idx_pred); */

    best_feature_cond = cond_feature_idx[best_feature_idx_cond];
    best_feature_pred = pred_feature_idx[best_feature_idx_pred];
    best_feature_cond_threshold = best_split[best_feature_idx_cond*num_frac_features_pred + best_feature_idx_pred];
    best_feature_cond_threshold_real = bins[best_feature_cond_threshold*num_total_features + best_feature_cond];
    if (debug) {
        printf("best_score_across_feature %.6f\n", best_score_across_feature);
        printf("depth %d best_feature_idx_cond %d best_feature_cond %d best_feature_idx_pred %d best_feature_pred %d\n",
                depth, best_feature_idx_cond, best_feature_cond,
                best_feature_idx_pred, best_feature_pred);
    }
    

    /* printf("best_feature_cond_threshold %d\n", */
    /*        best_feature_cond_threshold); */
    /* printf("best_feature_cond_threshold_real %.3f\n", */
    /*        best_feature_cond_threshold_real); */
    // printf("%.3f\n", best_score_across_feature);
    // printf("%d %.3f %d %d \n", node_id, best_score_across_feature, best_feature, best_feature_threshold);
    
    // free array
    free(score_all);
    free(best_score_across_split);
    free(best_split);
    

    bool *cond_active_idx_left = malloc(num_samples * sizeof(bool)), *cond_active_idx_right = malloc(num_samples * sizeof(bool));
    int num_cond_active_samples_left = 0, num_cond_active_samples_right = 0;
    for(int di = 0; di < num_samples; ++di) {
        cond_active_idx_left[di] = cond_active_idx[di] && (x[di*num_total_features + best_feature_cond] <= best_feature_cond_threshold);
        if (cond_active_idx_left[di]) num_cond_active_samples_left += 1;
        cond_active_idx_right[di] = cond_active_idx[di] && (x[di*num_total_features + best_feature_cond] > best_feature_cond_threshold);
        if (cond_active_idx_right[di]) num_cond_active_samples_right +=1;
    }


    bool *pred_active_idx_left = malloc(num_samples * sizeof(bool)), *pred_active_idx_right = malloc(num_samples * sizeof(bool));
    int num_pred_active_samples_left = 0, num_pred_active_samples_right = 0;
    for(int di = 0; di < num_samples; ++di) {
        pred_active_idx_left[di] = pred_active_idx[di] && (x[di*num_total_features + best_feature_cond] <= best_feature_cond_threshold);
        if (pred_active_idx_left[di]) num_pred_active_samples_left += 1;
        pred_active_idx_right[di] = pred_active_idx[di] && (x[di*num_total_features + best_feature_cond] > best_feature_cond_threshold);
        if (pred_active_idx_right[di]) num_pred_active_samples_right +=1;
    }
    
    if (debug) {
        printf("cond_active_left %d cond_active_right %d pred_active_left %d pred_active_right %d\n",
                num_cond_active_samples_left,
                num_cond_active_samples_right,
                num_pred_active_samples_left,
                num_pred_active_samples_right);
    }
    
    
    /* printf("%d %.3f %d %d %d \n", node_id, best_score_across_feature, best_feature, num_active_samples_left, num_active_samples_right); */
    
    /* printf("left and right %d %d \n", num_active_samples_left, num_active_samples_right); */
    if ((best_score_across_feature < min_gain_to_split) || (num_cond_active_samples_left == 0) || (num_cond_active_samples_right == 0)
    || (num_pred_active_samples_left == 0) || (num_pred_active_samples_right == 0) ) {
        tree[5*node_id] = pred_feature_idx[best_feature_idx_pred_leaf];
        tree[5*node_id + 1] = NAN;
        tree[5*node_id + 2] = leaf_weight;
        tree[5*node_id + 3] = NAN;
        tree[5*node_id + 4] = NAN;
        
        free(cond_active_idx_left);
        free(cond_active_idx_right);
        free(pred_active_idx_left);
        free(pred_active_idx_right);
        
        return node_id + 1;
    } else{
        tree[5*node_id] = best_feature_cond;
        tree[5*node_id + 1] = best_feature_cond_threshold_real;
        tree[5*node_id + 2] = NAN;
        tree[5*node_id + 3] = node_id + 1;
        tree[5*node_id + 4] = NAN;
        int node_id_right = build_tree_double_custom(tree,
                                                    node_id + 1,
                                                    depth + 1,
                                                    x,
                                                    bins,
                                                    x_grad,
                                                    xx_hessian,
                                                    cond_active_idx_left,
                                                    pred_active_idx_left,
                                                    cond_feature_idx,
                                                    pred_feature_idx,
                                                    cond_depth_sep,
                                                    pred_depth_sep,
                                                    depth_sep,
                                                    max_bin,
                                                    num_samples,
                                                    num_frac_features_cond,
                                                    num_frac_features_pred,
                                                    num_total_features,
                                                    max_depth,
                                                    min_data_in_leaf,
                                                    leaf_use_all_pred_feature,
                                                    use_delta,
                                                    cond_shallow_use_prior_only,
                                                    score_agg,
                                                    min_gain_to_split,
                                                    lambda_l2,
                                                    normalize,
                                                    split_sample,
                                                    debug
                                                    );
        tree[5*node_id + 4] = node_id_right;
        int node_id_next = build_tree_double_custom(tree,
                                                    node_id_right,
                                                    depth + 1,
                                                    x,
                                                    bins,
                                                    x_grad,
                                                    xx_hessian,
                                                    cond_active_idx_right,
                                                    pred_active_idx_right,
                                                    cond_feature_idx,
                                                    pred_feature_idx,
                                                    cond_depth_sep,
                                                    pred_depth_sep,
                                                    depth_sep,
                                                    max_bin,
                                                    num_samples,
                                                    num_frac_features_cond,
                                                    num_frac_features_pred,
                                                    num_total_features,
                                                    max_depth,
                                                    min_data_in_leaf,
                                                    leaf_use_all_pred_feature,
                                                    use_delta,
                                                    cond_shallow_use_prior_only,
                                                    score_agg,
                                                    min_gain_to_split,
                                                    lambda_l2,
                                                    normalize,
                                                    split_sample,
                                                    debug
                                                    );
        free(cond_active_idx_left);
        free(cond_active_idx_right);
        free(pred_active_idx_left);
        free(pred_active_idx_right);
        
        return node_id_next;
    }
}
        
        



void predict_custom(const double *restrict x,
                    const double *restrict tree,
                    double* restrict y_pred,
                    int num_samples,
                    int num_features,
                    int num_trees,
                    int num_nodes_per_tree,
                    double learning_rate,
                    int start_tree_idx,
                    int end_tree_idx
                    ) {

    memset(y_pred, 0, num_samples*sizeof(double));
    if (end_tree_idx == -1)
        end_tree_idx = num_trees;

    for (int sample_idx = 0; sample_idx < num_samples; ++sample_idx) {
        for (int tree_idx = start_tree_idx; tree_idx < end_tree_idx; ++tree_idx) {
            const double *tree_i = tree + tree_idx * num_nodes_per_tree * 5;
            int current_node = 0;
            while (isnan(tree_i[current_node*5+2])) {
                assert(!isnan(tree_i[current_node*5]));
                int feature_idx = (int)tree_i[current_node*5];
                assert((feature_idx >= 0) && (feature_idx < num_features));
                current_node = (x[(long)sample_idx * (long)num_features + (long)feature_idx] <= tree_i[current_node*5+1]) ? (int)tree_i[current_node*5+3] : (int)tree_i[current_node*5+4];
                assert((current_node >=0) && (current_node < num_nodes_per_tree));
            }
            // tree_i[current_node*5] stores the feature,
            // tree_i[current_node*5+2] stores the weight
            y_pred[sample_idx] += learning_rate * x[(long)sample_idx * (long)num_features + (long)tree_i[current_node*5]] * tree_i[current_node*5+2];
        }
    }
}


void train_custom(const unsigned char *restrict x,
    const double *restrict x_origin,
    const double *restrict bins,
    const double *restrict y,
    double *restrict tree,
    int* prior_cond_feature_idx,
    double* prior_cond_feature_fraction,
    int num_prior_cond_features,
    int depth_sep,
    int use_random_depth_sep,
    int cond_shallow_use_prior_only,
    int pred_deep_use_prior_only,
    int num_samples,
    int num_features,
    int bias_feature_idx,
    int num_trees,
    int max_depth,
    int max_bin,
    int min_data_in_leaf,
    double bagging_fraction,
    double cond_feature_fraction,
    double pred_feature_fraction,
    double learning_rate,
    int leaf_use_all_pred_feature,
    int cond_pred_share,
    int use_delta,
    int score_agg,
    double min_gain_to_split,
    double lambda_l2,
    int normalize,
    int split_sample,
    int debug
) {

    assert(num_samples >= 1);
    assert(num_features >= 1);
    assert(bagging_fraction > 0);
    assert(cond_feature_fraction > 0);
    assert(pred_feature_fraction > 0);
    assert(learning_rate > 0);
    assert(max_depth > 0);
    if (debug) {
        printf("num_samples %d \n", num_samples);
        printf("num_features %d \n", num_features);
        printf("bagging_fraction %.3f \n", bagging_fraction);
        printf("cond_feature_fraction %.3f \n", cond_feature_fraction);
        printf("pred_feature_fraction %.3f \n", pred_feature_fraction);
        printf("learning_rate %.4f \n", learning_rate);
        printf("max_depth %d \n", max_depth);
        printf("split_sample %d \n", split_sample);
    }
    int num_nodes_per_tree = (int)pow(2., max_depth + 1) - 1;
    double *yhat = malloc(num_samples*sizeof(double));
    double *yi = malloc(num_samples*sizeof(double));
    double *x_grad = malloc(num_samples*num_features*sizeof(double));
    double *xx_hessian = malloc(num_samples*num_features*sizeof(double));
    bool *cond_active_idx = malloc(num_samples*sizeof(bool));
    bool *pred_active_idx = malloc(num_samples*sizeof(bool));
    int *cond_feature_idx = malloc(num_features*sizeof(int));
    int *pred_feature_idx = malloc(num_features*sizeof(int));
    if (!(yhat && yi && x_grad && xx_hessian && cond_active_idx && pred_active_idx
        && cond_feature_idx && pred_feature_idx)) {
      printf("malloc failure \n");
      free(yhat);
      free(yi);
      free(x_grad);
      free(xx_hessian);
      free(cond_active_idx);
      free(pred_active_idx);
      free(cond_feature_idx);
      free(pred_feature_idx);
      return;
    }
  
    memset(yhat, 0, num_samples*sizeof(double));
    memset(yi, 0, num_samples*sizeof(double));
    memset(x_grad, 0, num_samples*num_features*sizeof(double));
    for (int di = 0; di < num_samples; ++di)
        for (int fi = 0; fi < num_features; ++fi) {
            xx_hessian[di*num_features + fi] = x_origin[di*num_features + fi] * x_origin[di*num_features + fi];
        }
    
    /* print_double(xx_hessian, num_samples*num_features, num_features); */
    
    for (int i = 0; i < num_trees * num_nodes_per_tree * 5; ++i)
        tree[i] = NAN;
    
    srand(1);
    /* double r = ((double)rand()/RAND_MAX); */
    /* printf("%.3f\n", r); */
    /* r = ((double)rand()/RAND_MAX); */
    /* printf("%.3f\n", r); */
    /* r = ((double)rand()/RAND_MAX); */
    /* printf("%.3f\n", r); */

    for (int tree_idx = 0; tree_idx < num_trees; ++tree_idx) {
        if (debug)
            printf("tree_idx %d \n", tree_idx);
        /* double grad_square_sum = 0; */
        for (int di = 0; di < num_samples; ++di) {
            for (int fi = 0; fi < num_features; ++fi) {
                x_grad[di*num_features + fi] = (yhat[di] - y[di]) * x_origin[di*num_features + fi];
            }
            /* grad_square_sum += (yhat[di] - y[di])*(yhat[di] - y[di]); */
        }
        /* printf("%.3f\n", sqrt(grad_square_sum/num_samples)); */
        
        double *tree_i = tree + tree_idx * num_nodes_per_tree * 5;
        
        // sample active
        int num_active_samples = 0;
        for (int di = 0; di < num_samples; ++di) {
            double r = ((double)rand()/RAND_MAX);
            if (r <= bagging_fraction) {
                if (split_sample == 1) {
                    double r_split = ((double)rand()/RAND_MAX);
                    /* if (debug) */
                    /* printf("%.3f\n", r_split); */
                    if (r_split <= 0.5) {
                        cond_active_idx[di] = true;
                        pred_active_idx[di] = false;
                    } else {
                        cond_active_idx[di] = false;
                        pred_active_idx[di] = true;
                    }
                } else if (split_sample == 2) {
                    if(di <= num_samples / 2) {
                        cond_active_idx[di] = true;
                        pred_active_idx[di] = false;
                    } else {
                        cond_active_idx[di] = false;
                        pred_active_idx[di] = true;
                    }
                    } else {
                        cond_active_idx[di] = true;
                        pred_active_idx[di] = true;
                    }
                    num_active_samples++;
                }
                else {
                    cond_active_idx[di] = false;
                    pred_active_idx[di] = false;
            }
        }
                
        /* if (debug) { */
        /*   print_bool(cond_active_idx, num_samples); */
        /*   print_bool(pred_active_idx, num_samples); */
        /* } */
        if (num_active_samples == 0) {
            printf("bagging_fraction is too small, randomly select 1 sample");
            cond_active_idx[rand()%num_samples] = true;
            pred_active_idx[rand()%num_samples] = true;
            num_active_samples = 1;
        }
                
        // put prior cond_features at the front of the cond_feature_idx
        int num_prior_cond_features_sampled = 0, num_frac_features_cond = 0;
        for (int fi = 0; fi < num_prior_cond_features; ++fi) {
            double r = ((double)rand()/RAND_MAX);
            if (r < prior_cond_feature_fraction[fi]) {
                cond_feature_idx[num_frac_features_cond] = prior_cond_feature_idx[fi];
                num_frac_features_cond++;
            }
        }
        num_prior_cond_features_sampled = num_frac_features_cond;
        if (debug)
            print_int(cond_feature_idx, num_prior_cond_features_sampled, num_prior_cond_features_sampled);
        
        
        /* print_int(prior_cond_feature_idx, num_prior_cond_features, num_prior_cond_features); */
        

        for (int fi = 0; fi < num_features; ++fi) {
            if (fi != bias_feature_idx) {
                int *found_fi = (int*) bsearch(&fi, prior_cond_feature_idx,
                                            num_prior_cond_features,
                                            sizeof(int), compareint);
                if (found_fi == NULL) {
                    double r = ((double)rand()/RAND_MAX);
                    if (r <= cond_feature_fraction) {
                        cond_feature_idx[num_frac_features_cond] = fi;
                        num_frac_features_cond++;
                    }
                }
                /* else { */
                /*     printf("found %d in prior\n ", *found_fi); */
                /* } */
            }
        }
        if (num_frac_features_cond == num_prior_cond_features_sampled) {
            if (debug)
                printf("no cond features other than prior cond features are sampled\n");
            for (int fi = 0; fi < num_features; ++fi) {
                if (fi != bias_feature_idx) {
                    int *found_fi = (int*) bsearch(&fi, prior_cond_feature_idx,
                                                    num_prior_cond_features,
                                                    sizeof(int), compareint);
                    if (found_fi == NULL) {
                        cond_feature_idx[num_frac_features_cond] = fi;
                        num_frac_features_cond++;
                        break;
                    }
                }
            }
            if (num_frac_features_cond != (num_prior_cond_features_sampled + 1)) {
                printf("Error: all features are in prior cond features\n");
                exit(1);
            }
        }
        
        if (debug)
            print_int(cond_feature_idx, num_frac_features_cond, num_frac_features_cond);
        int cond_depth_sep = num_frac_features_cond;
        if (cond_shallow_use_prior_only) {
            cond_depth_sep = num_prior_cond_features_sampled;
        }
        
        if (num_frac_features_cond == 0) {
            printf("feature_fraction is too small, randomly select 1 feature");
            cond_feature_idx[0] = rand() % num_features;
            num_frac_features_cond = 1;
        }


        /* printf("num_frac_features_pred\n"); */
        int num_frac_features_pred = 0;
        if (!cond_pred_share) {
            for (int fi = 0; fi < num_features; ++fi) {
                if (fi != bias_feature_idx) {
                    double r = ((double)rand()/RAND_MAX);
                    if (r <= pred_feature_fraction) {
                        pred_feature_idx[num_frac_features_pred] = fi;
                        num_frac_features_pred++;
                    }
                } else {
                    pred_feature_idx[num_frac_features_pred] = fi;
                    num_frac_features_pred++;
                }
                /* printf("%d %d\n", fi, num_frac_features_pred); */
            }
        } else {
            for (int fi = num_prior_cond_features_sampled; fi < num_frac_features_cond; ++fi)
                pred_feature_idx[fi-num_prior_cond_features_sampled] = cond_feature_idx[fi];
            num_frac_features_pred = num_frac_features_cond - num_prior_cond_features_sampled;
            pred_feature_idx[num_frac_features_pred] = bias_feature_idx;
            num_frac_features_pred++;
        }
        int pred_depth_sep = 0;
        if (pred_deep_use_prior_only) {
            pred_depth_sep = num_frac_features_pred - 1;
        }

        int tree_idx_depth_sep = depth_sep;
        if (num_prior_cond_features_sampled == 0)
            tree_idx_depth_sep = 0;
        if (use_random_depth_sep) {
            tree_idx_depth_sep = rand()%(depth_sep + 1);
        }

        if (debug)
            printf("tree_idx_depth_sep %d\n ", tree_idx_depth_sep);

        if (debug)
            print_int(pred_feature_idx, num_frac_features_pred, num_frac_features_pred);

        build_tree_double_custom(tree_i,
                                0,
                                0,
                                x,
                                bins,
                                x_grad,
                                xx_hessian,
                                cond_active_idx,
                                pred_active_idx,
                                cond_feature_idx,
                                pred_feature_idx,
                                cond_depth_sep,
                                pred_depth_sep,
                                tree_idx_depth_sep,
                                max_bin,
                                num_samples,
                                num_frac_features_cond,
                                num_frac_features_pred,
                                num_features,
                                max_depth,
                                min_data_in_leaf,
                                leaf_use_all_pred_feature,
                                use_delta,
                                cond_shallow_use_prior_only,
                                score_agg,
                                min_gain_to_split,
                                lambda_l2,
                                normalize,
                                split_sample,
                                debug
                                );
        /* print_double(tree_i, num_nodes_per_tree*5, 5); */

        // prediction
        memset(yi, 0, num_samples*sizeof(double));
        predict_custom(x_origin,
                    tree_i,
                    yi,
                    num_samples,
                    num_features,
                    num_trees,
                    num_nodes_per_tree,
                    learning_rate,
                    0,
                    1
                    );

        for (int di = 0; di < num_samples; ++di) {
            yhat[di] += yi[di];
        }
    }

    free(yhat);
    free(yi);
    free(x_grad);
    free(xx_hessian);
    free(cond_active_idx);
    free(pred_active_idx);
    free(cond_feature_idx);
    free(pred_feature_idx);
}




int build_tree_double_memory(double *tree,
                            const int node_id,
                            const int depth,
                            const unsigned char* x,
                            const double *bins,
                            const double *grad,
                            const double *hessian,
                            const bool *cond_active_idx,
                            const bool *pred_active_idx,
                            const int *cond_feature_idx,
                            const int *pred_feature_idx,
                            const int cond_depth_sep,
                            const int pred_depth_sep,
                            const int depth_sep,
                            const int max_bin,
                            const int num_samples,
                            const int num_frac_features_cond,
                            const int num_frac_features_pred,
                            const int num_total_features,
                            const int max_depth,
                            const int min_data_in_leaf,
                            const int leaf_use_all_pred_feature,
                            const int use_delta,
                            const int cond_shallow_use_prior_only,
                            const int score_agg,
                            const double min_gain_to_split,
                            const double lambda_l2,
                            const int normalize,
                            const int split_sample,
                            const int debug
    ) {
    /* printf("max_bin %d \n", max_bin); */
    /* printf("num_samples %d \n", num_samples); */
    /* printf("num_frac_features %d \n", num_frac_features); */
    /* printf("num_total_features  %d \n", num_total_features); */
    /* printf("max_depth %d \n", max_depth); */

    /* for (int di = 0; di < num_samples; ++di) */
    /* for (int fi = 0 ; fi < num_frac_features; ++fi) { */
    /* if (x[di*num_total_features + feature_idx[fi]] >= max_bin) */
    /* printf("%d %d\n", di, fi); */
    /* } */
    /* print_char(x, num_samples*num_total_features); */
    /* print_bool(active_idx, num_samples); */
    int num_active_samples = 0;
    double* grad_sum = malloc(num_frac_features_pred*sizeof(double));
    double* hessian_sum = malloc(num_frac_features_pred*sizeof(double));
    memset(grad_sum, 0, num_frac_features_pred*sizeof(double));
    memset(hessian_sum, 0, num_frac_features_pred*sizeof(double));

    for (int di = 0; di < num_samples; ++di) {
        if(pred_active_idx[di]) {
            for (int pi = 0; pi < num_frac_features_pred; ++pi) {
                long di_pi = (long)di*(long)num_total_features + (long)pred_feature_idx[pi];
                int x_di_pi = x[di_pi];
                double x_di_pi_real = bins[x_di_pi*num_total_features + pred_feature_idx[pi]];
                grad_sum[pi] += x_di_pi_real * grad[di];
                hessian_sum[pi] += x_di_pi_real*x_di_pi_real * hessian[di];

                /* grad_sum[pi] += x_grad[di*num_total_features + pred_feature_idx[pi]]; */
                /* hessian_sum[pi] += xx_hessian[di*num_total_features + pred_feature_idx[pi]]; */
            }
            num_active_samples++;
        }
    }
    if (debug)
        printf("node_id %d depth %d %d \n",
            node_id, depth, num_active_samples);
    /* printf("grad_sum %.3f \n", grad_sum); */
    /* printf("grad_sum\n"); */
    if (debug >= 2) {
        printf("grad and hessian sum\n");
        print_double(grad_sum, num_frac_features_pred, num_frac_features_pred);
        print_double(hessian_sum, num_frac_features_pred, num_frac_features_pred);
    }


    double score = -INFINITY;
    double leaf_weight = NAN;
    int best_feature_idx_pred_leaf = 0;
    int pi_start = 0, pi_end = num_frac_features_pred;
    if (!leaf_use_all_pred_feature) {
        pi_start = pred_depth_sep;
        /* printf("leaf not using all features\n"); */
    }
    for (int pi = pi_start; pi < pi_end; ++pi) {
        double score_pi = 0;
        if (normalize)
            score_pi = grad_sum[pi] * grad_sum[pi] / (hessian_sum[pi] * (lambda_l2 + 1) + 1e-8);
        else
            score_pi = grad_sum[pi] * grad_sum[pi] / (hessian_sum[pi] + lambda_l2 + 1e-8);
        if (score_pi > score ) {
            score = score_pi;
            best_feature_idx_pred_leaf = pi;
            if (normalize)
                leaf_weight = -grad_sum[pi] / (hessian_sum[pi] * (lambda_l2 + 1) + 1e-8);
            else
                leaf_weight = -grad_sum[pi] / (hessian_sum[pi] + lambda_l2 + 1e-8);
        }
    }

    /* if (debug) */
    /* printf("leaf_weight computation done\n"); */

    free(grad_sum);
    free(hessian_sum);

    if ((num_active_samples <= min_data_in_leaf) || (depth >= max_depth)) {
        tree[5*node_id] = pred_feature_idx[best_feature_idx_pred_leaf];
        tree[5*node_id + 1] = NAN;
        tree[5*node_id + 2] = leaf_weight;
        tree[5*node_id + 3] = NAN;
        tree[5*node_id + 4] = NAN;
        if (debug)
            printf("best pred feature %d leaf_weight %.3f\n",
                pred_feature_idx[best_feature_idx_pred_leaf],
                leaf_weight);

        return node_id + 1;
    }

    // grad_left and hessian_left:  num_frac_features x num_bins
    double *grad_left, *hessian_left;
    grad_left = malloc(num_frac_features_cond*num_frac_features_pred*max_bin*sizeof(double));
    hessian_left = malloc(num_frac_features_cond*num_frac_features_pred*max_bin*sizeof(double));
    memset(grad_left, 0, num_frac_features_cond*num_frac_features_pred*max_bin*sizeof(double));
    memset(hessian_left, 0, num_frac_features_cond*num_frac_features_pred*max_bin*sizeof(double));
    // access x, grad, hessian in sequential order
    // access grad_left, hessian_left in feature wise order
    //                         bin order is random

    /* print(grad_left, num_frac_features * max_bin, max_bin); */
    int ci_start = 0, ci_end = num_frac_features_cond;
    bool* active_idx = cond_active_idx;
    pi_start = 0; pi_end = num_frac_features_pred;
    if (depth < depth_sep) {
        if (split_sample)
            active_idx = cond_active_idx;
        ci_start = 0;
        ci_end = cond_depth_sep;
        pi_start = 0;
        // pi_end = num_frac_features_pred;
        pi_end = pred_depth_sep;
    } else {
        if (split_sample)
            active_idx = pred_active_idx;
        // if cond_shallow_use_prior_only == 2:
        // depth < depth_sep: [0, cond_depth_sep]
        // depth >= depth_sep: [cond_depth_sep, num_frac_features_cond]
        // else
        // depth < depth_sep: [0, cond_depth_sep]
        // depth >= depth_sep: [0, num_frac_features_cond]
        if (cond_shallow_use_prior_only == 2)
            ci_start = cond_depth_sep;
        else
            ci_start = 0;
        ci_end = num_frac_features_cond;
        pi_start = pred_depth_sep;
        pi_end = num_frac_features_pred;
    }
    if (debug >= 2) {
        printf("active_idx at depth %d \n", depth);
        print_bool(cond_active_idx, num_samples);
        print_bool(pred_active_idx, num_samples);
        print_bool(active_idx, num_samples);
    }
    
    for (int di = 0; di < num_samples; ++di) {
        if(active_idx[di]) {
            for (int ci = ci_start; ci < ci_end; ++ci) {
                for (int pi = pi_start; pi < pi_end; ++pi) {
                    // use unsigned char directly as int
                    /* printf("%d %d %d\n", di, feature_idx[fi], x[di*num_total_features + feature_idx[fi]]); */
                    long di_ci = (long)di*(long)num_total_features + (long)cond_feature_idx[ci];
                    long di_pi = (long)di*(long)num_total_features + (long)pred_feature_idx[pi];
                    int x_di_ci = x[di_ci];
                    int x_di_pi = x[di_pi];
                    double x_di_pi_real = bins[x_di_pi*num_total_features + pred_feature_idx[pi]];
            
                    grad_left[ci*num_frac_features_pred*max_bin + pi*max_bin 
                        + x_di_ci] \
                        += grad[di] * x_di_pi_real;
                    hessian_left[ci*num_frac_features_pred*max_bin + pi*max_bin
                        + x_di_ci]    \
                        += x_di_pi_real * x_di_pi_real * hessian[di];
            
                    /* grad_left[ci*num_frac_features_pred*max_bin + pi*max_bin */
                    /* + (int)x[di*num_total_features + cond_feature_idx[ci]]] \ */
                    /* += x_grad[di*num_total_features + pred_feature_idx[pi]]; */
                    /* hessian_left[ci*num_frac_features_pred*max_bin + pi*max_bin */
                    /* + (int)x[di*num_total_features + cond_feature_idx[ci]]] \ */
                    /* += xx_hessian[di*num_total_features + pred_feature_idx[pi]]; */
                }
            }
        }
    }

    /* if (debug >= 2) { */
        /* printf("grad_left\n"); */
        /* print_double(grad_left, num_frac_features_cond*num_frac_features_pred*max_bin, max_bin); */
        /* printf("hessian_left\n"); */
        /* print_double(hessian_left, num_frac_features_cond*num_frac_features_pred*max_bin, max_bin); */
    /* } */

    // calculate cumsum
    for (int ci = ci_start; ci < ci_end; ++ci) {
        for (int pi = pi_start; pi < pi_end; ++pi) {
            double grad_cumsum = 0., hessian_cumsum = 0.;
            for (int bin = 0; bin < max_bin; ++bin) {
                grad_cumsum += grad_left[ci*num_frac_features_pred*max_bin + pi*max_bin + bin];
                grad_left[ci*num_frac_features_pred*max_bin + pi*max_bin + bin] = grad_cumsum;
                hessian_cumsum += hessian_left[ci*num_frac_features_pred*max_bin + pi*max_bin + bin];
                hessian_left[ci*num_frac_features_pred*max_bin + pi*max_bin + bin] = hessian_cumsum;
            }
        }
    }



    /* print(grad_left, num_frac_features * max_bin, max_bin); */
    /* print(hessian_left, num_frac_features * max_bin, max_bin); */


    double* score_all = malloc(num_frac_features_cond * num_frac_features_pred * max_bin * sizeof(double));
    double* best_score_across_split = malloc(num_frac_features_cond * num_frac_features_pred * sizeof(double));
    int* best_split = malloc(num_frac_features_cond * num_frac_features_pred * sizeof(int));
    memset(best_split, -1, num_frac_features_cond * num_frac_features_pred*sizeof(int));
    for (int i = 0; i < num_frac_features_cond*num_frac_features_pred; ++i)
        best_score_across_split[i] = -INFINITY;
    /* print(best_score_across_split, num_frac_features, num_frac_features); */

    for (int ci = ci_start; ci < ci_end; ++ci) {
        for (int pi = pi_start; pi < pi_end; ++pi) {
            int idx_end = ci*num_frac_features_pred*max_bin + pi*max_bin + max_bin - 1;
            double score_base = 0.;
            if (normalize)
                score_base = grad_left[idx_end]*grad_left[idx_end]/(hessian_left[idx_end] * (lambda_l2 + 1) + 1e-8);
            else
                score_base = grad_left[idx_end]*grad_left[idx_end]/(hessian_left[idx_end] + lambda_l2 + 1e-8);
            for (int bin = 0; bin < max_bin; ++bin) {
                int idx_start = ci*num_frac_features_pred*max_bin + pi*max_bin + bin;
                double grad_right = grad_left[idx_end] - grad_left[idx_start];
                double hessian_right = hessian_left[idx_end]- hessian_left[idx_start];
                double score = 0;
                if (normalize)
                    score = grad_left[idx_start]*grad_left[idx_start] / (hessian_left[idx_start] * (lambda_l2 + 1) + 1e-8) + grad_right * grad_right / (hessian_right * (lambda_l2 + 1) + 1e-8);
                else
                    score = grad_left[idx_start]*grad_left[idx_start] / (hessian_left[idx_start] + lambda_l2 + 1e-8) + grad_right * grad_right / (hessian_right + lambda_l2 + 1e-8);
                if (use_delta)
                    score -= score_base;

                score_all[idx_start] = score;
                /* printf("%d %d %.3f \n", fi, bin, score); */
                /* printf("%.3f \n", score); */
                /* if (score > best_score_across_split[ci*num_frac_features_pred + pi]) { */
                /* best_score_across_split[ci*num_frac_features_pred + pi] = score; */
                /* best_split[ci*num_frac_features_pred + pi] = bin; */
                /* } */
            }
            /* printf("\n "); */
        }
    }

    /* if (debug >= 3) */
    /* print_double(score_all, num_frac_features_cond*num_frac_features_pred*max_bin, max_bin); */

    if (score_agg == 0) {
        // max across the bins
        for (int ci = ci_start; ci < ci_end; ++ci)
            for (int pi = pi_start; pi < pi_end; ++pi)
                for (int bin = 0; bin < max_bin; ++bin) {
                    int idx_start = ci*num_frac_features_pred*max_bin + pi*max_bin + bin;
                    double score = score_all[idx_start];
                    if (score > best_score_across_split[ci*num_frac_features_pred + pi]) {
                        best_score_across_split[ci*num_frac_features_pred + pi] = score;
                        best_split[ci*num_frac_features_pred + pi] = bin;
                    }
                }

    } else if (score_agg == 1) {
        // avg across pred features, then take max across bin
        for (int ci = ci_start; ci < ci_end; ++ci) {
            for (int bin = 0; bin < max_bin; ++bin) {
                double score_total = 0;
                for (int pi = pi_start; pi < pi_end; ++pi)
                    score_total += score_all[ci*num_frac_features_pred*max_bin + pi*max_bin + bin]/(pi_end - pi_start + 1e-8);
                /* if (debug >= 2) */
                /* printf("%.3f ", score_total); */
                if (score_total > best_score_across_split[ci*num_frac_features_pred + 0]) {
                    best_score_across_split[ci*num_frac_features_pred + 0] = score_total;
                    best_split[ci*num_frac_features_pred + 0] = bin;
                }
            }
        }
    } else if (score_agg == 2) {
        double *compare_array = malloc(num_frac_features_pred*sizeof(double));
        for (int ci = ci_start; ci < ci_end; ++ci) {
            for (int bin = 0; bin < max_bin; ++bin) {
                int pi_count = 0;
                for (int pi = pi_start; pi < pi_end; ++pi) {
                    compare_array[pi_count] = score_all[ci*num_frac_features_pred*max_bin + pi*max_bin + bin];
                    pi_count += 1;
                }
                if (debug >= 2) {
                    print_double(compare_array, pi_end-pi_start, pi_count);
                }
                double score_median = find_median(compare_array, pi_count);
                if (debug >= 2) {
                    print_double(compare_array, pi_end-pi_start, pi_count);
                    printf("median %.3f \n", score_median);
                }
                if (score_median > best_score_across_split[ci*num_frac_features_pred + 0]) {
                    best_score_across_split[ci*num_frac_features_pred + 0] = score_median;
                    best_split[ci*num_frac_features_pred + 0] = bin;
               }
            }
        }
        free(compare_array);
    } else if (score_agg == 3) {
        // random pred features, then take max across bin
        for (int ci = ci_start; ci < ci_end; ++ci) {
            for (int bin = 0; bin < max_bin; ++bin) {
                int pi = rand()%(pi_end - pi_start) + pi_start;
                double score_pi_rand = score_all[ci*num_frac_features_pred*max_bin + pi*max_bin + bin];
                if (score_pi_rand > best_score_across_split[ci*num_frac_features_pred + 0]) {
                    best_score_across_split[ci*num_frac_features_pred + 0] = score_pi_rand;
                    best_split[ci*num_frac_features_pred + 0] = bin;
                }
            }
        }
    } else if (score_agg == 10) {
        // random select bin
        for (int ci = ci_start; ci < ci_end; ++ci)
            for (int pi = pi_start; pi < pi_end; ++pi) {
                int bin = rand() % max_bin;
                int idx_start = ci*num_frac_features_pred*max_bin + pi*max_bin + bin;
                double score = score_all[idx_start];
                best_score_across_split[ci*num_frac_features_pred + pi] = score;
                best_split[ci*num_frac_features_pred + pi] = bin;
            }
    } else if (score_agg == 11) {
        // avg across pred features, then take random across bin
        for (int ci = ci_start; ci < ci_end; ++ci) {
            int bin = rand() % max_bin;
            double score_total = 0;
            for (int pi = pi_start; pi < pi_end; ++pi)
                score_total += score_all[ci*num_frac_features_pred*max_bin + pi*max_bin + bin]/(pi_end - pi_start + 1e-8);
            best_score_across_split[ci*num_frac_features_pred + 0] = score_total;
            best_split[ci*num_frac_features_pred + 0] = bin;
        }
    } else if (score_agg == 12) {
        double *compare_array = malloc(num_frac_features_pred*sizeof(double));
        for (int ci = ci_start; ci < ci_end; ++ci) {
            int bin = rand() % max_bin;
            int pi_count = 0;
            for (int pi = pi_start; pi < pi_end; ++pi) {
                compare_array[pi_count] = score_all[ci*num_frac_features_pred*max_bin + pi*max_bin + bin];
                pi_count += 1;
            }
            double score_median = find_median(compare_array, pi_count);
            best_score_across_split[ci*num_frac_features_pred + 0] = score_median;
            best_split[ci*num_frac_features_pred + 0] = bin;
        }
        free(compare_array);
    } if (score_agg == 13) {
        // random pred features, then take max across bin
        for (int ci = ci_start; ci < ci_end; ++ci) {
            int bin = rand() % max_bin;
            int pi = rand()%(pi_end - pi_start) + pi_start;
            double score_pi_bin_rand = score_all[ci*num_frac_features_pred*max_bin + pi*max_bin + bin];
            best_score_across_split[ci*num_frac_features_pred + 0] = score_pi_bin_rand;
            best_split[ci*num_frac_features_pred + 0] = bin;
        }
    }

    
    if (debug >= 2) {
        print_double(best_score_across_split,
                    num_frac_features_cond*num_frac_features_pred,
                    num_frac_features_pred);
        print_int(best_split,
                num_frac_features_cond*num_frac_features_pred,
                num_frac_features_pred);
    }



    /* printf("best_score_across_split \n"); */
    /* print_double(best_score_across_split, num_frac_features_cond*num_frac_features_pred, num_frac_features_pred); */
    /* printf("best_split \n"); */
    /* print_int(best_split, num_frac_features_cond*num_frac_features_pred, num_frac_features_pred); */




    /* printf("grad_left "); */
    /* printf("hessian_left "); */
    /* print(hessian_left, num_frac_features * max_bin, max_bin); */
    // free grad_left, hessian_left, score
    free(grad_left);
    free(hessian_left);
    double best_score_across_feature = -INFINITY;
    int best_feature_idx_cond = 0, best_feature_idx_pred = 0, best_feature_cond = 0, best_feature_pred = 0, best_feature_cond_threshold = 0;
    double best_feature_cond_threshold_real = NAN;
    if ((score_agg == 0) || (score_agg == 10)) {
        for (int ci = ci_start; ci < ci_end; ++ci) {
            for (int pi = pi_start; pi < pi_end; ++pi) {
                /* for (int pi = num_frac_features_pred - 1; pi < num_frac_features_pred; ++pi) { */
                if (best_score_across_split[ci*num_frac_features_pred + pi] > best_score_across_feature) {
                    best_score_across_feature = best_score_across_split[ci*num_frac_features_pred + pi];
                    best_feature_idx_cond = ci;
                    best_feature_idx_pred = pi;
                }
            }
        }
    } else if ((score_agg == 1) || (score_agg == 2) || (score_agg == 3) || (score_agg == 11) || (score_agg == 12) || (score_agg == 13) ) {
        for (int ci = ci_start; ci < ci_end; ++ci) {
            if (best_score_across_split[ci*num_frac_features_pred + 0] > best_score_across_feature) {
                best_score_across_feature = best_score_across_split[ci*num_frac_features_pred + 0];
                best_feature_idx_cond = ci;
            }
        }
    }
    /* printf("best_feature_idx_cond %d\n", best_feature_idx_cond); */
    /* printf("best_feature_idx_pred %d\n", best_feature_idx_pred); */

    best_feature_cond = cond_feature_idx[best_feature_idx_cond];
    best_feature_pred = pred_feature_idx[best_feature_idx_pred];
    best_feature_cond_threshold = best_split[best_feature_idx_cond*num_frac_features_pred + best_feature_idx_pred];
    best_feature_cond_threshold_real = bins[best_feature_cond_threshold*num_total_features + best_feature_cond];
    if (debug) {
        printf("best_score_across_feature %.6f\n", best_score_across_feature);
        printf("depth %d best_feature_idx_cond %d best_feature_cond %d best_feature_idx_pred %d best_feature_pred %d\n",
            depth, best_feature_idx_cond, best_feature_cond,
            best_feature_idx_pred, best_feature_pred);
    }


    /* printf("best_feature_cond_threshold %d\n", */
    /* best_feature_cond_threshold); */
    /* printf("best_feature_cond_threshold_real %.3f\n", */
    /* best_feature_cond_threshold_real); */
    // printf("%.3f\n", best_score_across_feature);
    // printf("%d %.3f %d %d \n", node_id, best_score_across_feature, best_feature, best_feature_threshold);

    // free array
    free(score_all);
    free(best_score_across_split);
    free(best_split);


    bool *cond_active_idx_left = malloc(num_samples * sizeof(bool)), *cond_active_idx_right = malloc(num_samples * sizeof(bool));
    int num_cond_active_samples_left = 0, num_cond_active_samples_right = 0;
    for(int di = 0; di < num_samples; ++di) {
        long di_cond = (long)di*(long)num_total_features + (long)best_feature_cond;
        cond_active_idx_left[di] = cond_active_idx[di] && (x[di_cond] <= best_feature_cond_threshold);
        if (cond_active_idx_left[di]) num_cond_active_samples_left += 1;
        cond_active_idx_right[di] = cond_active_idx[di] && (x[di_cond] > best_feature_cond_threshold);
        if (cond_active_idx_right[di]) num_cond_active_samples_right +=1;

        /* cond_active_idx_left[di] = cond_active_idx[di] && (x[di*num_total_features + best_feature_cond] <= best_feature_cond_threshold); */
        /* if (cond_active_idx_left[di]) num_cond_active_samples_left += 1; */
        /* cond_active_idx_right[di] = cond_active_idx[di] && (x[di*num_total_features + best_feature_cond] > best_feature_cond_threshold); */
        /* if (cond_active_idx_right[di]) num_cond_active_samples_right += 1; */
    }


    bool *pred_active_idx_left = malloc(num_samples * sizeof(bool)), *pred_active_idx_right = malloc(num_samples * sizeof(bool));
    int num_pred_active_samples_left = 0, num_pred_active_samples_right = 0;
    for(int di = 0; di < num_samples; ++di) {
        long di_cond = (long)di*(long)num_total_features + (long)best_feature_cond;
        pred_active_idx_left[di] = pred_active_idx[di] && (x[di_cond] <= best_feature_cond_threshold);
        if (pred_active_idx_left[di]) num_pred_active_samples_left += 1;
        pred_active_idx_right[di] = pred_active_idx[di] && (x[di_cond] > best_feature_cond_threshold);
        if (pred_active_idx_right[di]) num_pred_active_samples_right += 1;
        /* pred_active_idx_left[di] = pred_active_idx[di] && (x[di*num_total_features + best_feature_cond] <= best_feature_cond_threshold); */
        /* if (pred_active_idx_left[di]) num_pred_active_samples_left += 1; */
        /* pred_active_idx_right[di] = pred_active_idx[di] && (x[di*num_total_features + best_feature_cond] > best_feature_cond_threshold); */
        /* if (pred_active_idx_right[di]) num_pred_active_samples_right += 1; */
    }

    if (debug) {
        printf("cond_active_left %d cond_active_right %d pred_active_left %d pred_active_right %d\n",
            num_cond_active_samples_left,
            num_cond_active_samples_right,
            num_pred_active_samples_left,
            num_pred_active_samples_right);
    }

    
    /* printf("%d %.3f %d %d %d \n", node_id, best_score_across_feature, best_feature, num_active_samples_left, num_active_samples_right); */

    /* printf("left and right %d %d \n", num_active_samples_left, num_active_samples_right); */
    if ((best_score_across_feature < min_gain_to_split) || (num_cond_active_samples_left == 0) || (num_cond_active_samples_right == 0) \
    || (num_pred_active_samples_left == 0) || (num_pred_active_samples_right == 0) ) {
        tree[5*node_id] = pred_feature_idx[best_feature_idx_pred_leaf];
        tree[5*node_id + 1] = NAN;
        tree[5*node_id + 2] = leaf_weight;
        tree[5*node_id + 3] = NAN;
        tree[5*node_id + 4] = NAN;

        free(cond_active_idx_left);
        free(cond_active_idx_right);
        free(pred_active_idx_left);
        free(pred_active_idx_right);

        return node_id + 1;
    } else{
        tree[5*node_id] = best_feature_cond;
        tree[5*node_id + 1] = best_feature_cond_threshold_real;
        tree[5*node_id + 2] = NAN;
        tree[5*node_id + 3] = node_id + 1;
        tree[5*node_id + 4] = NAN;
        int node_id_right = build_tree_double_memory(tree,
                                                    node_id + 1,
                                                    depth + 1,
                                                    x,
                                                    bins,
                                                    grad,
                                                    hessian,
                                                    cond_active_idx_left,
                                                    pred_active_idx_left,
                                                    cond_feature_idx,
                                                    pred_feature_idx,
                                                    cond_depth_sep,
                                                    pred_depth_sep,
                                                    depth_sep,
                                                    max_bin,
                                                    num_samples,
                                                    num_frac_features_cond,
                                                    num_frac_features_pred,
                                                    num_total_features,
                                                    max_depth,
                                                    min_data_in_leaf,
                                                    leaf_use_all_pred_feature,
                                                    use_delta,
                                                    cond_shallow_use_prior_only,
                                                    score_agg,
                                                    min_gain_to_split,
                                                    lambda_l2,
                                                    normalize,
                                                    split_sample,
                                                    debug
                                                    );
        tree[5*node_id + 4] = node_id_right;
        int node_id_next = build_tree_double_memory(tree,
                                                    node_id_right,
                                                    depth + 1,
                                                    x,
                                                    bins,
                                                    grad,
                                                    hessian,
                                                    cond_active_idx_right,
                                                    pred_active_idx_right,
                                                    cond_feature_idx,
                                                    pred_feature_idx,
                                                    cond_depth_sep,
                                                    pred_depth_sep,
                                                    depth_sep,
                                                    max_bin,
                                                    num_samples,
                                                    num_frac_features_cond,
                                                    num_frac_features_pred,
                                                    num_total_features,
                                                    max_depth,
                                                    min_data_in_leaf,
                                                    leaf_use_all_pred_feature,
                                                    use_delta,
                                                    cond_shallow_use_prior_only,
                                                    score_agg,
                                                    min_gain_to_split,
                                                    lambda_l2,
                                                    normalize,
                                                    split_sample,
                                                    debug
                                                    );
        free(cond_active_idx_left);
        free(cond_active_idx_right);
        free(pred_active_idx_left);
        free(pred_active_idx_right);

        return node_id_next;
    }
}





void train_memory(const unsigned char *restrict x,
    const double *restrict x_origin,
    const double *restrict bins,
    const double *restrict y,
    double *restrict tree,
    int* prior_cond_feature_idx,
    double* prior_cond_feature_fraction,
    int num_prior_cond_features,
    int depth_sep,
    int use_random_depth_sep,
    int cond_shallow_use_prior_only,
    int pred_deep_use_prior_only,
    int num_samples,
    int num_features,
    int bias_feature_idx,
    int num_trees,
    int max_depth,
    int max_bin,
    int min_data_in_leaf,
    double bagging_fraction,
    double cond_feature_fraction,
    double pred_feature_fraction,
    double learning_rate,
    int leaf_use_all_pred_feature,
    int cond_pred_share,
    int use_delta,
    int score_agg,
    double min_gain_to_split,
    double lambda_l2,
    int normalize,
    int split_sample,
    int debug
    ) {

    assert(num_samples >= 1);
    assert(num_features >= 1);
    assert(bagging_fraction > 0);
    assert(cond_feature_fraction > 0);
    assert(pred_feature_fraction > 0);
    assert(learning_rate > 0);
    assert(max_depth > 0);
    if (debug) {
        printf("num_samples %d \n", num_samples);
        printf("num_features %d \n", num_features);
        printf("bagging_fraction %.3f \n", bagging_fraction);
        printf("cond_feature_fraction %.3f \n", cond_feature_fraction);
        printf("pred_feature_fraction %.3f \n", pred_feature_fraction);
        printf("learning_rate %.4f \n", learning_rate);
        printf("max_depth %d \n", max_depth);
        printf("split_sample %d \n", split_sample);
    }
    int num_nodes_per_tree = (int)pow(2., max_depth + 1) - 1;
    double *yhat = malloc(num_samples*sizeof(double));
    double *yi = malloc(num_samples*sizeof(double));
    double *grad = malloc(num_samples*sizeof(double));
    double *hessian = malloc(num_samples*sizeof(double));
    bool *cond_active_idx = malloc(num_samples*sizeof(bool));
    bool *pred_active_idx = malloc(num_samples*sizeof(bool));
    int *cond_feature_idx = malloc(num_features*sizeof(int));
    int *pred_feature_idx = malloc(num_features*sizeof(int));
    if (!(yhat && yi && grad && hessian && cond_active_idx && pred_active_idx
        && cond_feature_idx && pred_feature_idx)) {
        printf("malloc failure \n");
        free(yhat);
        free(yi);
        free(grad);
        free(hessian);
        free(cond_active_idx);
        free(pred_active_idx);
        free(cond_feature_idx);
        free(pred_feature_idx);
        return;
    }
  
    memset(yhat, 0, num_samples*sizeof(double));
    memset(yi, 0, num_samples*sizeof(double));
    memset(grad, 0, num_samples*sizeof(double));
    for (int di = 0; di < num_samples; ++di)
        hessian[di] = 1;
  
    /* for (int di = 0; di < num_samples; ++di) */
    /* for (int fi = 0; fi < num_features; ++fi) { */
    /* xx_hessian[di*num_features + fi] = x_origin[di*num_features + fi] * x_origin[di*num_features + fi]; */
    /* } */
    
    /* print_double(xx_hessian, num_samples*num_features, num_features); */
    
    for (int i = 0; i < num_trees * num_nodes_per_tree * 5; ++i)
        tree[i] = NAN;
  
    srand(1);
    /* double r = ((double)rand()/RAND_MAX); */
    /* printf("%.3f\n", r); */
    /* r = ((double)rand()/RAND_MAX); */
    /* printf("%.3f\n", r); */
    /* r = ((double)rand()/RAND_MAX); */
    /* printf("%.3f\n", r); */

    for (int tree_idx = 0; tree_idx < num_trees; ++tree_idx) {
        if (debug)
            printf("tree_idx %d \n", tree_idx);
        /* double grad_square_sum = 0; */
        for (int di = 0; di < num_samples; ++di) {
            grad[di] = yhat[di] - y[di];
            /* for (int fi = 0; fi < num_features; ++fi) { */
            /* x_grad[di*num_features + fi] = (yhat[di] - y[di]) * x_origin[di*num_features + fi]; */
            /* } */
            /* grad_square_sum += (yhat[di] - y[di])*(yhat[di] - y[di]); */
        }
        /* printf("%.3f\n", sqrt(grad_square_sum/num_samples)); */
    
        double *tree_i = tree + tree_idx * num_nodes_per_tree * 5;
    
        // sample active
        int num_active_samples = 0;
        for (int di = 0; di < num_samples; ++di) {
            double r = ((double)rand()/RAND_MAX);
            if (r <= bagging_fraction) {
                if (split_sample == 1) {
                    double r_split = ((double)rand()/RAND_MAX);
                    /* if (debug) */
                    /* printf("%.3f\n", r_split); */
                    if (r_split <= 0.5) {
                        cond_active_idx[di] = true;
                        pred_active_idx[di] = false;
                    } else {
                        cond_active_idx[di] = false;
                        pred_active_idx[di] = true;
                    }
                } else if (split_sample == 2) {
                    if(di <= num_samples / 2) {
                        cond_active_idx[di] = true;
                        pred_active_idx[di] = false;
                    } else {
                        cond_active_idx[di] = false;
                        pred_active_idx[di] = true;
                    }
                } else {
                    cond_active_idx[di] = true;
                    pred_active_idx[di] = true;
                }
                num_active_samples++;
            }
            else {
                cond_active_idx[di] = false;
                pred_active_idx[di] = false;
            }
        }
        
        /* if (debug) { */
        /* print_bool(cond_active_idx, num_samples); */
        /* print_bool(pred_active_idx, num_samples); */
        /* } */
    if (num_active_samples == 0) {
        printf("bagging_fraction is too small, randomly select 1 sample");
        cond_active_idx[rand()%num_samples] = true;
        pred_active_idx[rand()%num_samples] = true;
        num_active_samples = 1;
    }
    
    // put prior cond_features at the front of the cond_feature_idx
    int num_prior_cond_features_sampled = 0, num_frac_features_cond = 0;
    for (int fi = 0; fi < num_prior_cond_features; ++fi) {
        double r = ((double)rand()/RAND_MAX);
        if (r < prior_cond_feature_fraction[fi]) {
            cond_feature_idx[num_frac_features_cond] = prior_cond_feature_idx[fi];
            num_frac_features_cond++;
        }
    }
    num_prior_cond_features_sampled = num_frac_features_cond;
    if (debug)
        print_int(cond_feature_idx, num_prior_cond_features_sampled, num_prior_cond_features_sampled);
    

    /* print_int(prior_cond_feature_idx, num_prior_cond_features, num_prior_cond_features); */
    

    for (int fi = 0; fi < num_features; ++fi) {
        if (fi != bias_feature_idx) {
            int *found_fi = (int*) bsearch(&fi, prior_cond_feature_idx,
                                           num_prior_cond_features,
                                           sizeof(int), compareint);
            if (found_fi == NULL) {
                double r = ((double)rand()/RAND_MAX);
                if (r <= cond_feature_fraction) {
                    cond_feature_idx[num_frac_features_cond] = fi;
                    num_frac_features_cond++;
                }
            }
            /* else { */
            /* printf("found %d in prior\n ", *found_fi); */
            /* } */
            }
        }
        if (num_frac_features_cond == num_prior_cond_features_sampled) {
            if (debug)
                printf("no cond features other than prior cond features are sampled\n");
            for (int fi = 0; fi < num_features; ++fi) {
                if (fi != bias_feature_idx) {
                    int *found_fi = (int*) bsearch(&fi, prior_cond_feature_idx,
                                                    num_prior_cond_features,
                                                    sizeof(int), compareint);
                    if (found_fi == NULL) {
                        cond_feature_idx[num_frac_features_cond] = fi;
                        num_frac_features_cond++;
                        break;
                    }
                }
            }
            if (num_frac_features_cond != (num_prior_cond_features_sampled + 1)) {
                printf("Error: all features are in prior cond features\n");
                exit(1);
            }
        }
            
        if (debug)
            print_int(cond_feature_idx, num_frac_features_cond, num_frac_features_cond);
        int cond_depth_sep = num_frac_features_cond;
        if (cond_shallow_use_prior_only) {
            cond_depth_sep = num_prior_cond_features_sampled;
        }
        
        if (num_frac_features_cond == 0) {
            printf("feature_fraction is too small, randomly select 1 feature");
            cond_feature_idx[0] = rand() % num_features;
            num_frac_features_cond = 1;
        }
            
            
        /* printf("num_frac_features_pred\n"); */
        int num_frac_features_pred = 0;
        if (!cond_pred_share) {
            for (int fi = 0; fi < num_features; ++fi) {
                if (fi != bias_feature_idx) {
                    double r = ((double)rand()/RAND_MAX);
                    if (r <= pred_feature_fraction) {
                        pred_feature_idx[num_frac_features_pred] = fi;
                        num_frac_features_pred++;
                    }
                } else {
                    pred_feature_idx[num_frac_features_pred] = fi;
                    num_frac_features_pred++;
                }
                /* printf("%d %d\n", fi, num_frac_features_pred); */
            }
        } else {
            for (int fi = num_prior_cond_features_sampled; fi < num_frac_features_cond; ++fi)
                pred_feature_idx[fi-num_prior_cond_features_sampled] = cond_feature_idx[fi];
            num_frac_features_pred = num_frac_features_cond - num_prior_cond_features_sampled;
            pred_feature_idx[num_frac_features_pred] = bias_feature_idx;
            num_frac_features_pred++;
        }
        int pred_depth_sep = 0;
        if (pred_deep_use_prior_only) {
            pred_depth_sep = num_frac_features_pred - 1;
        }
            
        int tree_idx_depth_sep = depth_sep;
        if (num_prior_cond_features_sampled == 0)
            tree_idx_depth_sep = 0;
        if (use_random_depth_sep) {
            tree_idx_depth_sep = rand()%(depth_sep + 1);
        }
            
        if (debug)
            printf("tree_idx_depth_sep %d\n ", tree_idx_depth_sep);
            
        if (debug)
            print_int(pred_feature_idx, num_frac_features_pred, num_frac_features_pred);

        build_tree_double_memory(tree_i,
                                0,
                                0,
                                x,
                                bins,
                                grad,
                                hessian,
                                cond_active_idx,
                                pred_active_idx,
                                cond_feature_idx,
                                pred_feature_idx,
                                cond_depth_sep,
                                pred_depth_sep,
                                tree_idx_depth_sep,
                                max_bin,
                                num_samples,
                                num_frac_features_cond,
                                num_frac_features_pred,
                                num_features,
                                max_depth,
                                min_data_in_leaf,
                                leaf_use_all_pred_feature,
                                use_delta,
                                cond_shallow_use_prior_only,
                                score_agg,
                                min_gain_to_split,
                                lambda_l2,
                                normalize,
                                split_sample,
                                debug
                                );
        /* print_double(tree_i, num_nodes_per_tree*5, 5); */

        // prediction
        memset(yi, 0, num_samples*sizeof(double));
        predict_custom(x_origin,
                        tree_i,
                        yi,
                        num_samples,
                        num_features,
                        num_trees,
                        num_nodes_per_tree,
                        learning_rate,
                        0,
                        1
                        );

        for (int di = 0; di < num_samples; ++di) {
            yhat[di] += yi[di];
        }
    }

    free(yhat);
    free(yi);
    free(grad);
    free(hessian);
    free(cond_active_idx);
    free(pred_active_idx);
    free(cond_feature_idx);
    free(pred_feature_idx);

}