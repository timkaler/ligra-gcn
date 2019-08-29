// This code is part of the project "Ligra: A Lightweight Graph Processing
// Framework for Shared Memory", presented at Principles and Practice of 
// Parallel Programming, 2013.
// Copyright (c) 2013 Julian Shun and Guy Blelloch
//
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
//#include "Eigen/Dense"
#include <adept_source.h>
#include <adept_arrays.h>
#include <cilk/reducer.h>
#include <cilk/reducer_opadd.h>

//int sched_yield(void) { for (int i=0; i< 4000; i++) _mm_pause(); return 0;}

using adept::Matrix;
using adept::Real;



void zero_init(Matrix& mat) {
  for (int i = 0; i < mat.dimensions()[0]; i++) {
    for (int j = 0; j < mat.dimensions()[1]; j++) {
      mat(i,j) = 0.0;
    }
  }
}


#include "parse_pubmed.cpp"

#include <random>
#include <vector>
#include <map>
#include "plad_reducer.cpp"

struct BFS_F {
  uintE* Parents;
  BFS_F(uintE* _Parents) : Parents(_Parents) {}
  inline bool update (uintE s, uintE d) { //Update
    if(Parents[d] == UINT_E_MAX) { Parents[d] = s; return 1; }
    else return 0;
  }
  inline bool updateAtomic (uintE s, uintE d){ //atomic version of Update
    return (CAS(&Parents[d],UINT_E_MAX,s));
  }
  //cond function checks if vertex has been visited yet
  inline bool cond (uintE d) { return (Parents[d] == UINT_E_MAX); } 
};

template <class vertex>
struct d_GCN_F {

  graph<vertex>& GA;

  uintE* Parents;

  Matrix& weights;
  Matrix* next_vertex_embeddings;
  Matrix* prev_vertex_embeddings;

  //Matrix& d_weights;


  Matrix* d_weights;

  Matrix* d_next_vertex_embeddings;

  Matrix* d_prev_vertex_embeddings;

  ArrayReducer* reducer;


  d_GCN_F(uintE* _Parents, graph<vertex>& _GA,
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

    uintE* neighbors = reinterpret_cast<uintE*>(GA.V[v].getInNeighbors());
    int in_degree = GA.V[v].getInDegree();

    // reverse-mode of fmax(0.0, next_vertex_embeddings[v]).
    for (int i = 0; i < next_vertex_embeddings[v].dimensions()[0]; i++) {
      if (next_vertex_embeddings[v](i,0) <= 0.0) {
        d_next_vertex_embeddings[v](i,0) = 0.0;
      }
    }


    ArrayReducerView* view = reducer->view();

    Matrix& d_weights_view = *(view->get_view(d_weights));



    // reverse of matrix multiplies.
    for (int i = 0; i < in_degree; i++) {
      uintE n = neighbors[i];
      for (int j = 0; j < weights.dimensions()[0]; j++) {
        //std::cout << prev_vertex_embeddings[n] << std::endl;
        for (int k = 0; k < weights.dimensions()[1]; k++) {
          d_weights_view(j,k) += prev_vertex_embeddings[n](k,0) * d_next_vertex_embeddings[v](j,0);
          //d_prev_vertex_embeddings[n](k,0) += weights(j,k) * d_next_vertex_embeddings[v](k,0);
        }
      }
      Matrix& d_prev_vertex_embeddings_n = *(view->get_view(&(d_prev_vertex_embeddings[n])));
      // propagate to d_prev_vertex_embeddings[n]
      for (int k = 0; k < weights.dimensions()[1]; k++) {
        for (int j = 0; j < weights.dimensions()[0]; j++) {
          d_prev_vertex_embeddings_n(k,0) += weights(j,k) * d_next_vertex_embeddings[v](j,0);
        }
      }
    }


    // set next_vertex_embeddings[v] to zero.
    //for (int i = 0; i < next_vertex_embeddings[v].dimensions()[0]; i++) {
    //  d_next_vertex_embeddings[v](i,0) = 0.0;
    //}

    return 1;
  }





  inline bool updateAtomic (uintE s, uintE d){ //atomic version of Update
    return (CAS(&Parents[d],UINT_E_MAX,s));
  }
  //cond function checks if vertex has been visited yet
  inline bool cond (uintE d) { return (Parents[d] == UINT_E_MAX); } 

};





template <class vertex>
struct GCN_F {

  graph<vertex>& GA;

  uintE* Parents;
  Matrix& weights;
  Matrix* next_vertex_embeddings;
  Matrix* prev_vertex_embeddings;

  GCN_F(uintE* _Parents, graph<vertex>& _GA,
        Matrix& _weights, Matrix* _next_vertex_embeddings,
        Matrix* _prev_vertex_embeddings) : Parents(_Parents), GA(_GA), weights(_weights),
                                          next_vertex_embeddings(_next_vertex_embeddings),
                                          prev_vertex_embeddings(_prev_vertex_embeddings) {}
  //float** next_vertex_embeddings;
  inline bool update (uintE s, uintE d) {
    printf("Processing edge %d -> %d\n", s, d);
    if(Parents[d] == UINT_E_MAX) { Parents[d] = s; return 1; }
    else return 0;
    //return 1;
  }




  inline bool operator() (uintE v) {
    //printf("processing vertex %d\n", v);
    uintE* neighbors = reinterpret_cast<uintE*>(GA.V[v].getInNeighbors());
    int in_degree = GA.V[v].getInDegree();

    for (int i = 0; i < in_degree; i++) {
      uintE n = neighbors[i];
      next_vertex_embeddings[v] += weights ** prev_vertex_embeddings[n];
    }

    next_vertex_embeddings[v] = fmax(0.0, next_vertex_embeddings[v]);

    return 1;
  }

  //inline bool reverse (uintE v, Matrix* d_next_vertex_embeddings, Matrix& d_weights, Matrix* d_prev_vertex_embeddings) {

  //  uintE* neighbors = reinterpret_cast<uintE*>(GA.V[v].getInNeighbors());
  //  int in_degree = GA.V[v].getInDegree();

  //  // reverse-mode of fmax(0.0, next_vertex_embeddings[v]).
  //  for (int i = 0; i < next_vertex_embeddings[v].dimensions()[0]; i++) {
  //    if (next_vertex_embeddings[v](i,0) <= 0.0) {
  //      d_next_vertex_embeddings[v](i,0) = 0.0;
  //    }
  //  }

  //  // reverse of matrix multiplies.
  //  for (int i = 0; i < in_degree; i++) {
  //    uintE n = neighbors[i];
  //    for (int j = 0; j < weights.dimensions()[0]; j++) {
  //      for (int k = 0; k < weights.dimensions()[1]; k++) {
  //        d_weights(j,k) += prev_vertex_embeddings[n](k,0) * d_next_vertex_embeddings[v](j,0);
  //        //d_prev_vertex_embeddings[n](k,0) += weights(j,k) * d_next_vertex_embeddings[v](k,0);
  //      }
  //    }
  //    // propagate to d_prev_vertex_embeddings[n]
  //    for (int k = 0; k < weights.dimensions()[1]; k++) {
  //      for (int j = 0; j < weights.dimensions()[0]; j++) {
  //        d_prev_vertex_embeddings[n](k,0) += weights(j,k) * d_next_vertex_embedding[v](j,0);
  //      }
  //    }
  //  }


  //  // set next_vertex_embeddings[v] to zero.
  //  for (int i = 0; i < next_vertex_embeddings[v].dimensions()[0]; i++) {
  //    next_vertex_embeddings[v](i,0) = 0.0;
  //  }

  //  return 1;
  //}





  inline bool updateAtomic (uintE s, uintE d){ //atomic version of Update
    return (CAS(&Parents[d],UINT_E_MAX,s));
  }
  //cond function checks if vertex has been visited yet
  inline bool cond (uintE d) { return (Parents[d] == UINT_E_MAX); } 

};



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
      //loss_sum += -1.0 * y(i,j)*log(yhat(i,j) + 1e-12) - (1.0-y(i,j))*log(1-yhat(i,j) + 1e-12);
      d_yhat(i,j) = d_output*(-1.0 * y(i,j) / (yhat(i,j) + 1e-12) - (1.0-y(i,j))*1.0/(1-yhat(i,j)+1e-12)) * (1.0/n);
    }
  }
}

//void d_softmax(Matrix& input, Matrix& d_input, Matrix& output, Matrix& d_output) {
//  d_input = -1.0*output * sum(d_output*output) + output*d_output;
//  zero_init(d_output);
//}



void sqloss(Matrix& input1, Matrix& input2, double& output) {
  Matrix diff = input1-input2;
  output = sum(diff*diff);
}

void d_sqloss(Matrix& input1, Matrix& d_input1, Matrix& input2,
              double& d_output) {
  d_input1 = 2*(input1-input2) * d_output;
  d_output = 0.0;
}



void random_init(std::default_random_engine& gen, Matrix& mat) {
  std::uniform_real_distribution<double> distribution(0.0, 1.0/sqrt(mat.dimensions()[0]*mat.dimensions()[1]));
  for (int i = 0; i < mat.dimensions()[0]; i++) {
    for (int j = 0; j < mat.dimensions()[1]; j++) {
      mat(i,j) = distribution(gen);
    }
  }
}



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
  bool* is_train = (bool*) calloc(n_vertices, sizeof(bool));
  bool* is_val = (bool*) calloc(n_vertices, sizeof(bool));
  bool* is_test = (bool*) calloc(n_vertices, sizeof(bool));


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

  //std::cout << feature_vectors[0] << std::endl;


  Matrix weights = Matrix(3,500);

  std::default_random_engine generator(1000);

  random_init(generator, weights);

  for (int64_t i = 0; i < GA.n; i++) {
    vIndices[i] = true;
  }
  vertexSubset Frontier(n, vIndices); //creates initial frontier

  for (int iter = 0; iter < 100; iter++) {
    Matrix* next_vertex_embeddings = new Matrix[GA.n];
    Matrix* prev_vertex_embeddings = &(feature_vectors[0]);

    cilk_for (int i = 0; i < GA.n; i++) {
      next_vertex_embeddings[i] = Matrix(3,1);
      zero_init(next_vertex_embeddings[i]);
    }

    //while(!Frontier.isEmpty()){ //loop until frontier is empty
    vertexMap(Frontier, GCN_F<vertex>(Parents, GA, weights, next_vertex_embeddings,
                                      prev_vertex_embeddings));


    Matrix* final_vertex_embeddings = new Matrix[GA.n];
    cilk_for (int i = 0; i < GA.n; i++) {
      final_vertex_embeddings[i] = Matrix(3,1);
      softmax(next_vertex_embeddings[i], final_vertex_embeddings[i]);
    }


    double* losses = new double[GA.n];
    double total_loss = 0.0;
    cilk::reducer_opadd<double> total_loss_reducer(total_loss);
    cilk_for (int i = 0; i < GA.n; i++) {
      if (!is_train[i]) {
        losses[i] = 0.0;
        continue;
      }
      //sqloss(final_vertex_embeddings[i], groundtruth_labels[i], losses[i]);
      crossentropy(final_vertex_embeddings[i], groundtruth_labels[i], losses[i]);
      *total_loss_reducer += losses[i];
    }
    total_loss = total_loss_reducer.get_value();

    printf("total loss is %f\n", total_loss);


    // now do reverse.
    Matrix* d_final_vertex_embeddings = new Matrix[GA.n];
    cilk_for (int i = 0; i < GA.n; i++) {
      double d_loss = 1.0;
      if (!is_train[i]) d_loss = 0.0;
      //d_sqloss(final_vertex_embeddings[i], d_final_vertex_embeddings[i], groundtruth_labels[i], d_loss);
      d_final_vertex_embeddings[i] = Matrix(final_vertex_embeddings[i].dimensions()[0], final_vertex_embeddings[i].dimensions()[1]);
      zero_init(d_final_vertex_embeddings[i]);
      d_crossentropy(final_vertex_embeddings[i], d_final_vertex_embeddings[i], groundtruth_labels[i], d_loss);
    }

    Matrix* d_next_vertex_embeddings = new Matrix[GA.n];
    cilk_for (int i = 0; i < GA.n; i++) {
      d_softmax(next_vertex_embeddings[i], d_next_vertex_embeddings[i],
                final_vertex_embeddings[i], d_final_vertex_embeddings[i]);
    }

    Matrix d_weights = Matrix(weights.dimensions()[0], weights.dimensions()[1]);

    //cilk::reducer_opadd<Matrix> d_weights(Matrix(weights.dimensions()[0], weights.dimensions()[1]));


    zero_init(d_weights);

    //Matrix* d_next_vertex_embeddings = new Matrix[GA.n];
    Matrix* d_prev_vertex_embeddings = new Matrix[GA.n];
    cilk_for (int i = 0; i < GA.n; i++) {
      d_prev_vertex_embeddings[i] = Matrix(prev_vertex_embeddings[i].dimensions()[0], prev_vertex_embeddings[i].dimensions()[1]);
      //zero_init(d_next_vertex_embeddings[i]);
      zero_init(d_prev_vertex_embeddings[i]);
    }


    ArrayReducer reducer;
    vertexMap(Frontier, d_GCN_F<vertex>(Parents, GA, weights, &d_weights,
                                        next_vertex_embeddings, d_next_vertex_embeddings,
                                        prev_vertex_embeddings, d_prev_vertex_embeddings,
                                        &reducer));

    ArrayReducerView* view = reducer.view();
    cilk_for (int i = 0; i < GA.n; i++) {
      if (view->has_view(&(d_prev_vertex_embeddings[i]))) {
        d_prev_vertex_embeddings[i] = *(view->get_view(&(d_prev_vertex_embeddings[i])));
        delete view->get_view(&(d_prev_vertex_embeddings[i]));
      }
    }
    d_weights = *(view->get_view(&d_weights));
    delete view->get_view(&d_weights);

    cilk_for (int i = 0; i < weights.dimensions()[0]; i++) {
      cilk_for (int j = 0; j < weights.dimensions()[1]; j++) {
        weights(i,j) -= 0.01 * (d_weights)(i,j);
      }
    }

    delete[] next_vertex_embeddings;
    delete[] final_vertex_embeddings;
    delete[] losses;
    delete[] d_final_vertex_embeddings;
    delete[] d_next_vertex_embeddings;
    delete[] d_prev_vertex_embeddings;
  }
  //std::cout << d_weights << std::endl;
  //Frontier.del();
    //Frontier = output; //set new frontier
  //}
  Frontier.del();

  //free(vIndices);
  //free(is_train);
  //free(is_val);
  //free(is_test);
  free(Parents);
}
