// Permission is hereby granted, free of charge, to any person obtaining a
// copy of this software and associated documentation files (the
// "Software"), to deal in the Software without restriction, including
// without limitation the rights (to use, copy, modify, merge, publish,
// distribute, sublicense, and/or sell copies of the Software, and to
// permit persons to whom the Software is furnished to do so, subject to
// the following conditions:
//
// The above copyright notice and this permission notice shall be included
// in all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
// OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
// MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
// NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
// LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
// OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
// WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#include "ligra.h"

#include <adept_source.h>
#include <adept_arrays.h>
#include <random>
#include <vector>
#include <map>
#include <cilk/reducer.h>
#include <cilk/reducer_opadd.h>

using adept::Matrix;
using adept::Real;

#include "parse_pubmed.cpp"

double PARAM_ADAM_B1 = 0.9;
double PARAM_ADAM_B2 = 0.999;
double PARAM_ADAM_EPSILON = 1e-8;



/*
	Utility Functions
*/

void zero_init(Matrix& mat) {
  for (int i = 0; i < mat.dimensions()[0]; i++) {
    for (int j = 0; j < mat.dimensions()[1]; j++) {
      mat(i,j) = 0.0;
    }
  }
}

void random_init(std::default_random_engine& gen, Matrix& mat) {
  std::uniform_real_distribution<double> distribution(0.0, 1.0/(mat.dimensions()[0]*mat.dimensions()[1]));
  for (int i = 0; i < mat.dimensions()[0]; i++) {
    for (int j = 0; j < mat.dimensions()[1]; j++) {
      mat(i,j) = distribution(gen);
    }
  }
}

void apply_gradient_update_ADAM(std::vector<Matrix>& weights, std::vector<Matrix>& d_weights, std::vector<Matrix>& vel, std::vector<Matrix>& mom,
                                double mul, double lr, int t) {

  double lr_t = lr * (sqrt(1.0-pow(PARAM_ADAM_B2, t)) / (1.0-pow(PARAM_ADAM_B1, t)));

  cilk_for (int i = 0; i < weights.size(); i++) {
    cilk_for (int j = 0; j < weights[i].dimensions()[0]; j++) {
      cilk_for (int k = 0; k < weights[i].dimensions()[1]; k++) {
        double g = d_weights[i](j,k) * mul;
        double m = mom[i](j,k);
        double v = vel[i](j,k);

        double m_t = PARAM_ADAM_B1 * m + (1.0 - PARAM_ADAM_B1) * g;
        double v_t = PARAM_ADAM_B2 * v + (1.0 - PARAM_ADAM_B2) * (g*g);

        double new_val = weights[i](j,k) - lr_t * m_t / (sqrt(v_t) + PARAM_ADAM_EPSILON);

        mom[i](j,k) = m_t;
        vel[i](j,k) = v_t;
        weights[i](j,k) = new_val;
      }
    }
  }
}

/*
	Gradient table reducer
*/

#include "gradient_table_reducer.cpp"


/*
	Functions/Operations plus functions to propagate their adjoints backwards.
*/

void softmax(Matrix& input, Matrix& output) {
  // max val divided out for numerical stability
  Real mval = maxval(input);
  output = exp(input-mval)/sum(exp(input-mval));
}

void d_softmax(Matrix& input, Matrix& d_input, Matrix& output, Matrix& d_output) {
  d_input = -1.0*output * sum(d_output*output) + output*d_output;
  zero_init(d_output);
}

void crossentropy(Matrix& yhat, Matrix& y, double& output) {
  double loss_sum = 0.0;
  double n = y.dimensions()[0]*y.dimensions()[1];
  for (int i = 0; i < y.dimensions()[0]; i++) {
    for (int j = 0; j < y.dimensions()[1]; j++) {
      loss_sum += -1.0 * y(i,j)*log(yhat(i,j) + 1e-12) - (1.0-y(i,j))*log(1-yhat(i,j) + 1e-12);
    }
  }
  output = loss_sum / n;
}

void d_crossentropy(Matrix& yhat, Matrix& d_yhat, Matrix& y, double& d_output) {
   double n = y.dimensions()[0]*y.dimensions()[1];
   for (int i = 0; i < y.dimensions()[0]; i++) {
    for (int j = 0; j < y.dimensions()[1]; j++) {
      d_yhat(i,j) = d_output*(-1.0 * y(i,j) / (yhat(i,j) + 1e-12) + (1.0-y(i,j))*1.0/(1-yhat(i,j)+1e-12)) * (1.0/n);
    }
  }
}

void sqloss(Matrix& input1, Matrix& input2, double& output) {
  Matrix diff = input1-input2;
  output = sum(diff*diff);
}

void d_sqloss(Matrix& input1, Matrix& d_input1, Matrix& input2,
              double& d_output) {
  d_input1 = 2*(input1-input2) * d_output;
  d_output = 0.0;
}



template <class vertex>
struct GCN_applyweights_F {

  graph<vertex>& GA;

  uintE* Parents;
  Matrix& weights;
  Matrix* prev_vertex_embeddings;
  Matrix* next_vertex_embeddings;
  bool first;
  GCN_applyweights_F(uintE* _Parents, graph<vertex>& _GA,
        Matrix& _weights, Matrix* _next_vertex_embeddings,
        Matrix* _prev_vertex_embeddings) : Parents(_Parents), GA(_GA), weights(_weights),
                                          next_vertex_embeddings(_next_vertex_embeddings),
                                          prev_vertex_embeddings(_prev_vertex_embeddings) {}

  inline bool operator() (uintE v) {
    uintE n = v;
    next_vertex_embeddings[v] = (weights ** prev_vertex_embeddings[n]);
    return 1;
  }
};

template <class vertex>
struct d_GCN_applyweights_F {

  graph<vertex>& GA;

  uintE* Parents;

  Matrix& weights;
  Matrix* next_vertex_embeddings;
  Matrix* prev_vertex_embeddings;

  Matrix* d_weights;

  Matrix* d_next_vertex_embeddings;
  Matrix* d_prev_vertex_embeddings;

  ArrayReducer* reducer;

  d_GCN_applyweights_F(uintE* _Parents, graph<vertex>& _GA,
        Matrix& _weights, Matrix* _d_weights,
        Matrix* _next_vertex_embeddings, Matrix* _d_next_vertex_embeddings,
        Matrix* _prev_vertex_embeddings, Matrix* _d_prev_vertex_embeddings,
        ArrayReducer* _reducer) :
                                          Parents(_Parents), GA(_GA),
                                          weights(_weights), d_weights(_d_weights),
                                          next_vertex_embeddings(_next_vertex_embeddings),
                                          d_next_vertex_embeddings(_d_next_vertex_embeddings),
                                          prev_vertex_embeddings(_prev_vertex_embeddings),
                                          d_prev_vertex_embeddings(_d_prev_vertex_embeddings),
                                          reducer(_reducer) {}

  inline bool operator() (uintE v) {
    ArrayReducerView* view = reducer->view();
    Matrix& d_weights_view = *(view->get_view(d_weights));
    uintE n = v;
    for (int j = 0; j < weights.dimensions()[0]; j++) {
      if (d_next_vertex_embeddings[v](j,0) == 0.0) continue;
      for (int k = 0; k < weights.dimensions()[1]; k++) {
        d_weights_view(j,k) += prev_vertex_embeddings[n](k,0) * d_next_vertex_embeddings[v](j,0);
      }
    }
    Matrix& d_prev_vertex_embeddings_n = *(view->get_view(&(d_prev_vertex_embeddings[n])));
    // propagate to d_prev_vertex_embeddings[n]
    for (int j = 0; j < weights.dimensions()[0]; j++) {
      if (d_next_vertex_embeddings[v](j,0) == 0.0) continue;
      for (int k = 0; k < weights.dimensions()[1]; k++) {
        d_prev_vertex_embeddings_n(k,0) += weights(j,k) * d_next_vertex_embeddings[v](j,0);
      }
    }
    return 1;
  }
};




template <class vertex>
struct GCN_F {

  graph<vertex>& GA;

  uintE* Parents;
  Matrix& weights;
  Matrix& skip_weights;
  Matrix* next_vertex_embeddings;
  Matrix* prev_vertex_embeddings;
  Matrix* prevprev_vertex_embeddings;
  bool first;
  GCN_F(uintE* _Parents, graph<vertex>& _GA,
        Matrix& _weights, Matrix& _skip_weights, Matrix* _next_vertex_embeddings,
        Matrix* _prev_vertex_embeddings, Matrix* _prevprev_vertex_embeddings, bool _first) : Parents(_Parents), GA(_GA), weights(_weights), skip_weights(_skip_weights),
                                          next_vertex_embeddings(_next_vertex_embeddings),
                                          prev_vertex_embeddings(_prev_vertex_embeddings), prevprev_vertex_embeddings(_prevprev_vertex_embeddings), first(_first) {}

  inline bool operator() (uintE v) {
    uintE* neighbors = reinterpret_cast<uintE*>(GA.V[v].getOutNeighbors());
    int in_degree = GA.V[v].getOutDegree();

    // self edge
    {
      uintE n = v;
      //double edge_weight = 1.0/sqrt(in_degree + GA.V[n].getOutDegree() + 1);
      //printf("skip_weights dims (%d,%d), prev_vertex dims (%d,%d)\n", skip_weights.dimensions()[0],
      //       skip_weights.dimensions()[1], prev_vertex_embeddings[n].dimensions()[0],
      //       prev_vertex_embeddings[n].dimensions()[1]);
      next_vertex_embeddings[v] = skip_weights ** prevprev_vertex_embeddings[n];
    }

    //if (!first) {
      // pack the neighbors into a matrix.
      if (in_degree > 0) {
        //int prev_nrows = prev_vertex_embeddings[v].dimensions()[0];
        //Matrix neighbor_embeddings = Matrix(prev_nrows, in_degree);
        //Matrix ones = Matrix(in_degree, 1);
        for (int i = 0; i < in_degree; i++) {
        //  ones(i,0) = 1.0;
          uintE n = neighbors[i];
          double edge_weight = 1.0/sqrt(in_degree + GA.V[n].getOutDegree());
        //  for (int j = 0; j < prev_nrows; j++) {
        //    neighbor_embeddings(j,i) = prev_vertex_embeddings[n](j,0)*edge_weight;
        //  }
          next_vertex_embeddings[v] += prev_vertex_embeddings[n]*edge_weight;//(weights ** neighbor_embeddings) ** ones;
        }
      }

      //for (int i = 0; i < in_degree; i++) {
      //  uintE n = neighbors[i];
      //  double edge_weight = 1.0/sqrt(in_degree + GA.V[n].getOutDegree());
      //  next_vertex_embeddings[v] += (weights ** prev_vertex_embeddings[n]) * edge_weight;
      //}
      if (!first) {
        next_vertex_embeddings[v] = fmax(0.0, next_vertex_embeddings[v]);
      }
    //}
    return 1;
  }
};

template <class vertex>
struct d_GCN_F {

  graph<vertex>& GA;

  uintE* Parents;

  Matrix& weights;
  Matrix& skip_weights;
  Matrix* next_vertex_embeddings;
  Matrix* prev_vertex_embeddings;
  Matrix* prevprev_vertex_embeddings;
  bool first;

  Matrix* d_weights;
  Matrix* d_skip_weights;

  Matrix* d_next_vertex_embeddings;

  Matrix* d_prev_vertex_embeddings;
  Matrix* d_prevprev_vertex_embeddings;

  ArrayReducer* reducer;

  d_GCN_F(uintE* _Parents, graph<vertex>& _GA,
        Matrix& _weights, Matrix* _d_weights,
        Matrix& _skip_weights, Matrix* _d_skip_weights,
        Matrix* _next_vertex_embeddings, Matrix* _d_next_vertex_embeddings,
        Matrix* _prev_vertex_embeddings, Matrix* _d_prev_vertex_embeddings,
        Matrix* _prevprev_vertex_embeddings, Matrix* _d_prevprev_vertex_embeddings,
        ArrayReducer* _reducer, bool _first) :
                                          Parents(_Parents), GA(_GA),
                                          weights(_weights), d_weights(_d_weights),
                                          skip_weights(_skip_weights), d_skip_weights(_d_skip_weights),
                                          next_vertex_embeddings(_next_vertex_embeddings),
                                          d_next_vertex_embeddings(_d_next_vertex_embeddings),
                                          prev_vertex_embeddings(_prev_vertex_embeddings),
                                          d_prev_vertex_embeddings(_d_prev_vertex_embeddings),
                                          prevprev_vertex_embeddings(_prevprev_vertex_embeddings), d_prevprev_vertex_embeddings(_d_prevprev_vertex_embeddings),
                                          reducer(_reducer), first(_first) {}

  inline bool operator() (uintE v) {

    uintE* neighbors = reinterpret_cast<uintE*>(GA.V[v].getOutNeighbors());
    int in_degree = GA.V[v].getOutDegree();

    // reverse-mode of fmax(0.0, next_vertex_embeddings[v]).
    if (!first) {
      for (int i = 0; i < next_vertex_embeddings[v].dimensions()[0]; i++) {
        if (next_vertex_embeddings[v](i,0) <= 0.0) {
          d_next_vertex_embeddings[v](i,0) = 0.0;
        }
      }
    }


    ArrayReducerView* view = reducer->view();

    //Matrix& d_weights_view = *(view->get_view(d_weights));
    Matrix& d_skip_weights_view = *(view->get_view(d_skip_weights));


    // self edge.
    {
      uintE n = v;
      for (int j = 0; j < weights.dimensions()[0]; j++) {
        if (d_next_vertex_embeddings[v](j,0) == 0.0) continue;
        for (int k = 0; k < weights.dimensions()[1]; k++) {
          d_skip_weights_view(j,k) += prevprev_vertex_embeddings[n](k,0) * d_next_vertex_embeddings[v](j,0);
        }
      }
      Matrix& d_prevprev_vertex_embeddings_n = *(view->get_view(&(d_prevprev_vertex_embeddings[n])));
      // propagate to d_prev_vertex_embeddings[n]
      //d_prevprev_vertex_embeddings_n += d_next_vertex_embeddings[v]; //* 1.0/sqrt(in_degree + GA.V[n].getOutDegree() + 1);
      for (int j = 0; j < weights.dimensions()[0]; j++) {
        if (d_next_vertex_embeddings[v](j,0) == 0.0) continue;
        for (int k = 0; k < weights.dimensions()[1]; k++) {
          d_prevprev_vertex_embeddings_n(k,0) += skip_weights(j,k) * d_next_vertex_embeddings[v](j,0);
        }
      }
    }

    //if (!first) {
      // reverse of matrix multiplies.
      for (int i = 0; i < in_degree; i++) {
        uintE n = neighbors[i];
        double edge_weight = 1.0/sqrt(in_degree + GA.V[n].getOutDegree());
        //for (int j = 0; j < weights.dimensions()[0]; j++) {
        //  if (d_next_vertex_embeddings[v](j,0) == 0.0) continue;
        //  for (int k = 0; k < weights.dimensions()[1]; k++) {
        //    d_weights_view(j,k) += /*prev_vertex_embeddings[n](k,0) **/ d_next_vertex_embeddings[v](j,0) * edge_weight;
        //  }
        //}
        Matrix& d_prev_vertex_embeddings_n = *(view->get_view(&(d_prev_vertex_embeddings[n])));
        // propagate to d_prev_vertex_embeddings[n]
        d_prev_vertex_embeddings_n += d_next_vertex_embeddings[v]*edge_weight;
        //for (int j = 0; j < weights.dimensions()[0]; j++) {
        //  if (d_next_vertex_embeddings[v](j,0) == 0.0) continue;
        //  for (int k = 0; k < weights.dimensions()[1]; k++) {
        //    d_prev_vertex_embeddings_n(k,0) += /*weights(j,k) * */d_next_vertex_embeddings[v](j,0) * edge_weight;
        //  }
        //}
      }
    //}
    return 1;
  }
};


/*
	Main compute function.
*/

template <class vertex>
void Compute(graph<vertex>& GA, commandLine P) {
  long start = P.getOptionLongValue("-r",0);
  long n = GA.n;
  //creates Parents array, initialized to all -1, except for start
  uintE* Parents = newA(uintE,n);
  parallel_for(long i=0;i<n;i++) Parents[i] = UINT_E_MAX;
  Parents[start] = start;

  bool* vIndices = static_cast<bool*>(malloc(sizeof(bool)*GA.n));


  int n_vertices = GA.n;
  int feature_dim = 500;
  bool* is_train = static_cast<bool*>(calloc(n_vertices, sizeof(bool)));
  bool* is_val = static_cast<bool*>(calloc(n_vertices, sizeof(bool)));
  bool* is_test = static_cast<bool*>(calloc(n_vertices, sizeof(bool)));


  std::vector<Matrix> groundtruth_labels;
  std::vector<Matrix> feature_vectors;

  for (int i = 0; i < n_vertices; i++) {
    Matrix tmp(3,1);
    zero_init(tmp);
    groundtruth_labels.push_back(tmp);

    Matrix tmp2(feature_dim,1);
    zero_init(tmp2);
    feature_vectors.push_back(tmp2);
  }

  // parse the data from the graph.
  parse_pubmed_data("pubmed.trainlabels", "pubmed.vallabels",
                    "pubmed.testlabels", "pubmed_features", is_train,
                    is_val, is_test, groundtruth_labels,
                    feature_vectors);

  double learning_rate = 0.1;

  std::vector<int> gcn_embedding_dimensions;
  gcn_embedding_dimensions.push_back(feature_dim);
  gcn_embedding_dimensions.push_back(32);
  gcn_embedding_dimensions.push_back(3);

  std::vector<Matrix> layer_weights, layer_skip_weights, d_layer_weights, d_layer_skip_weights,
                      layer_weights_momentum, layer_weights_velocity, layer_skip_weights_momentum,
                      layer_skip_weights_velocity;

  for (int i = 0; i < gcn_embedding_dimensions.size()-1; i++) {
    layer_weights.push_back(Matrix(gcn_embedding_dimensions[i+1], gcn_embedding_dimensions[i]));
    layer_skip_weights.push_back(Matrix(gcn_embedding_dimensions[i+1], gcn_embedding_dimensions[i]));
    d_layer_weights.push_back(Matrix(gcn_embedding_dimensions[i+1], gcn_embedding_dimensions[i]));
    d_layer_skip_weights.push_back(Matrix(gcn_embedding_dimensions[i+1], gcn_embedding_dimensions[i]));
    layer_weights_momentum.push_back(Matrix(gcn_embedding_dimensions[i+1], gcn_embedding_dimensions[i]));
    layer_weights_velocity.push_back(Matrix(gcn_embedding_dimensions[i+1], gcn_embedding_dimensions[i]));
    layer_skip_weights_velocity.push_back(Matrix(gcn_embedding_dimensions[i+1], gcn_embedding_dimensions[i]));
    layer_skip_weights_momentum.push_back(Matrix(gcn_embedding_dimensions[i+1], gcn_embedding_dimensions[i]));
  }

  for (int i = 0; i < layer_weights.size(); i++) {
    zero_init(layer_weights_momentum[i]);
    zero_init(layer_weights_velocity[i]);
    zero_init(layer_skip_weights_momentum[i]);
    zero_init(layer_skip_weights_velocity[i]);
  }

  std::default_random_engine generator(1000);
  for (int i = 0; i < layer_weights.size(); i++) {
    random_init(generator, layer_weights[i]);
    random_init(generator, layer_skip_weights[i]);
  }

  for (int64_t i = 0; i < GA.n; i++) {
    vIndices[i] = true;
  }
  vertexSubset Frontier(n, vIndices); //creates initial frontier

  for (int iter = 0; iter < 30; iter++) {
    std::vector<Matrix*> embedding_list, d_embedding_list, pre_embedding_list, d_pre_embedding_list;
    embedding_list.push_back(&(feature_vectors[0]));
    for (int i = 0; i < gcn_embedding_dimensions.size(); i++) {
      if (i > 0) embedding_list.push_back(new Matrix[GA.n]);
      pre_embedding_list.push_back(new Matrix[GA.n]);
      d_pre_embedding_list.push_back(new Matrix[GA.n]);
      d_embedding_list.push_back(new Matrix[GA.n]);
    }

    // Forward
    for (int i = 0; i < embedding_list.size()-1; i++) {
      Matrix* prev_vertex_embeddings = embedding_list[i];
      Matrix* next_vertex_embeddings = embedding_list[i+1];

      Matrix& weights = layer_weights[i];
      Matrix& skip_weights = layer_skip_weights[i];

      cilk_for (int j = 0; j < GA.n; j++) {
        next_vertex_embeddings[j] = Matrix(weights.dimensions()[0],1);
        pre_embedding_list[i][j] = Matrix(weights.dimensions()[0],1);
        zero_init(next_vertex_embeddings[j]);
        zero_init(pre_embedding_list[i][j]);
      }
      bool first = (i == embedding_list.size()-2);

      vertexMap(Frontier, GCN_applyweights_F<vertex>(Parents, GA, weights, pre_embedding_list[i],
                                        prev_vertex_embeddings));
      vertexMap(Frontier, GCN_F<vertex>(Parents, GA, weights, skip_weights, next_vertex_embeddings,
                                        pre_embedding_list[i], prev_vertex_embeddings, first));
    }

    Matrix* final_vertex_embeddings = new Matrix[GA.n];
    cilk_for (int i = 0; i < GA.n; i++) {
      final_vertex_embeddings[i] = Matrix(3,1);
      softmax(embedding_list[embedding_list.size()-1][i], final_vertex_embeddings[i]);
    }

    double* losses = new double[GA.n];
    double total_loss = 0.0;

    cilk::reducer_opadd<double> total_loss_reducer(total_loss);

    int batch_size = 0;
    int total_val_correct = 0;
    int total_val = 0;

    cilk::reducer_opadd<int> batch_size_reducer(batch_size);
    cilk::reducer_opadd<int> total_val_correct_reducer(total_val_correct);
    cilk::reducer_opadd<int> total_val_reducer(total_val);

    cilk_for (int i = 0; i < GA.n; i++) {
      if (!is_train[i]) {
        losses[i] = 0.0;
        if (is_val[i]) continue;
        crossentropy(final_vertex_embeddings[i], groundtruth_labels[i], losses[i]);
        int argmax = 0;
        int gt_label = 0;
        double maxval = -1;
        for (int j = 0; j < final_vertex_embeddings[i].dimensions()[0]; j++) {
          if (final_vertex_embeddings[i](j,0) > maxval || j == 0) {
            argmax = j;
            maxval = final_vertex_embeddings[i](j,0);
          }
          if (groundtruth_labels[i](j,0) > 0.5) gt_label = j;
        }
        if (gt_label == argmax) {
          *total_val_correct_reducer += 1;
        }
        *total_val_reducer += 1;
        continue;
      }
      *batch_size_reducer += 1;
      crossentropy(final_vertex_embeddings[i], groundtruth_labels[i], losses[i]);
      *total_loss_reducer += losses[i];
    }
    total_loss = total_loss_reducer.get_value();
    batch_size = batch_size_reducer.get_value();
    total_val_correct = total_val_correct_reducer.get_value();
    total_val = total_val_reducer.get_value();

    printf("epoch %d: \ttotal loss is %f test accuracy %f\n", iter+1, total_loss/batch_size, (1.0*total_val_correct) / total_val);

    // now do reverse.
    Matrix* d_final_vertex_embeddings = new Matrix[GA.n];
    cilk_for (int i = 0; i < GA.n; i++) {
      double d_loss = 1.0/batch_size;
      if (!is_train[i]) d_loss = 0.0;
      d_final_vertex_embeddings[i] = Matrix(final_vertex_embeddings[i].dimensions()[0], final_vertex_embeddings[i].dimensions()[1]);
      zero_init(d_final_vertex_embeddings[i]);
      d_crossentropy(final_vertex_embeddings[i], d_final_vertex_embeddings[i], groundtruth_labels[i], d_loss);
    }

    Matrix* d_next_vertex_embeddings = d_embedding_list[d_embedding_list.size()-1];// new Matrix[GA.n];
    Matrix* next_vertex_embeddings = embedding_list[d_embedding_list.size()-1];// new Matrix[GA.n];
    cilk_for (int i = 0; i < GA.n; i++) {
      d_softmax(next_vertex_embeddings[i], d_next_vertex_embeddings[i],
                final_vertex_embeddings[i], d_final_vertex_embeddings[i]);
    }

    for (int i = embedding_list.size()-2; i >= 0; --i) {
      bool first = (i == embedding_list.size()-2);
      Matrix* d_weights = &(d_layer_weights[i]);
      Matrix* d_skip_weights = &(d_layer_skip_weights[i]);
      Matrix& weights = layer_weights[i];
      Matrix& skip_weights = layer_skip_weights[i];
      zero_init(*d_weights);
      zero_init(*d_skip_weights);

      Matrix* prev_vertex_embeddings = embedding_list[i];
      Matrix* next_vertex_embeddings = embedding_list[i+1];
      Matrix* d_prev_vertex_embeddings = d_embedding_list[i];
      Matrix* d_next_vertex_embeddings = d_embedding_list[i+1];

      cilk_for (int j = 0; j < GA.n; j++) {
        d_prev_vertex_embeddings[j] = Matrix(prev_vertex_embeddings[j].dimensions()[0], prev_vertex_embeddings[j].dimensions()[1]);
        d_pre_embedding_list[i][j] = Matrix(pre_embedding_list[i][j].dimensions()[0], pre_embedding_list[i][j].dimensions()[1]);
        zero_init(d_prev_vertex_embeddings[j]);
        zero_init(d_pre_embedding_list[i][j]);
      }

      ArrayReducer reducer;

      vertexMap(Frontier, d_GCN_F<vertex>(Parents, GA, weights, d_weights, skip_weights, d_skip_weights,
                                          next_vertex_embeddings, d_next_vertex_embeddings,
                                          pre_embedding_list[i], d_pre_embedding_list[i],
                                          prev_vertex_embeddings, d_prev_vertex_embeddings,
                                          &reducer, first));
      {
        ArrayReducerView* view = reducer.view();
        cilk_for (int j = 0; j < GA.n; j++) {
          if (view->has_view(&(d_pre_embedding_list[i][j]))) {
            d_pre_embedding_list[i][j] = *(view->get_view(&(d_pre_embedding_list[i][j])));
            delete view->get_view(&(d_pre_embedding_list[i][j]));
          }
          if (view->has_view(&(d_prev_vertex_embeddings[j]))) {
            d_prev_vertex_embeddings[j] = *(view->get_view(&(d_prev_vertex_embeddings[j])));
            //delete view->get_view(&(d_prev_vertex_embeddings[j]));
          }
        }
      }

      vertexMap(Frontier, d_GCN_applyweights_F<vertex>(Parents, GA, weights, d_weights,
                                          pre_embedding_list[i], d_pre_embedding_list[i],
                                          prev_vertex_embeddings, d_prev_vertex_embeddings,
                                          &reducer));

      ArrayReducerView* view = reducer.view();
      cilk_for (int i = 0; i < GA.n; i++) {
        if (view->has_view(&(d_prev_vertex_embeddings[i]))) {
          d_prev_vertex_embeddings[i] = *(view->get_view(&(d_prev_vertex_embeddings[i])));
          delete view->get_view(&(d_prev_vertex_embeddings[i]));
        }
      }
      *d_weights = *(view->get_view(d_weights));
      *d_skip_weights = *(view->get_view(d_skip_weights));
      delete view->get_view(d_weights);
      delete view->get_view(d_skip_weights);
    }


    apply_gradient_update_ADAM(layer_weights, d_layer_weights, layer_weights_velocity, layer_weights_momentum, 1.0, learning_rate, iter+1);
    apply_gradient_update_ADAM(layer_skip_weights, d_layer_skip_weights, layer_skip_weights_velocity, layer_skip_weights_momentum, 1.0, learning_rate, iter+1);

    for (int i = 0; i < embedding_list.size(); i++) {
      if (i > 0) delete[] embedding_list[i];

      delete[] d_embedding_list[i];
      delete[] pre_embedding_list[i];
      delete[] d_pre_embedding_list[i];
    }

    delete[] final_vertex_embeddings;
    delete[] losses;
    delete[] d_final_vertex_embeddings;
  }
  Frontier.del();

  free(is_train);
  free(is_val);
  free(is_test);
  free(Parents);
}
