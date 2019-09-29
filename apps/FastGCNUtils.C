

template <class vertex>
struct getVertexDegrees {
  graph<vertex>& GA;
  int64_t* vertexDegrees;
  getVertexDegrees(int64_t* _vertexDegrees, graph<vertex>& _GA)
                   : vertexDegrees(_vertexDegrees), GA(_GA) {}
  inline bool update (uintE v, uintE u) {
    vertexDegrees[v] += 1;//GA.V[v].getOutDegree() + GA.V[v].getInDegree();
    vertexDegrees[u] += 1;//GA.V[v].getOutDegree() + GA.V[v].getInDegree();
    //printf("got vertex degree! %d %d\n", v, vertexDegrees[v]);
    return false;
  }

  inline bool updateAtomic (uintE v, uintE u) {
    __sync_fetch_and_add(&(vertexDegrees[v]),1);
    __sync_fetch_and_add(&(vertexDegrees[u]),1);
    return false;
  }

  inline bool cond(uintE d) {
    return cond_true(d);
  }
};



/***
  Vertex RELU
***/
template <class vertex>
struct GCN_vertexRELU_F {
  graph<vertex>& GA;
  uintE* Parents;
  MatrixXf& weights;
  MatrixXf* prev_vertex_embeddings;
  MatrixXf* next_vertex_embeddings;
  bool first;

  GCN_vertexRELU_F(uintE* _Parents, graph<vertex>& _GA,
        MatrixXf& _weights, MatrixXf* _next_vertex_embeddings,
        MatrixXf* _prev_vertex_embeddings) : Parents(_Parents), GA(_GA), weights(_weights),
                                          next_vertex_embeddings(_next_vertex_embeddings),
                                          prev_vertex_embeddings(_prev_vertex_embeddings) {}
  // update with source s and destination d
  inline bool operator() (uintE v) {
    next_vertex_embeddings[v] = prev_vertex_embeddings[v].cwiseMax(0.0);
    return true;
  }
};

template <class vertex>
struct d_GCN_vertexRELU_F {
  graph<vertex>& GA;

  uintE* Parents;

  MatrixXf& weights;
  MatrixXf* next_vertex_embeddings;
  MatrixXf* prev_vertex_embeddings;

  MatrixXf* d_weights;
  MatrixXf* d_next_vertex_embeddings;
  MatrixXf* d_prev_vertex_embeddings;

  ArrayReducer* reducer;

  d_GCN_vertexRELU_F(uintE* _Parents, graph<vertex>& _GA,
        MatrixXf& _weights, MatrixXf* _d_weights,
        MatrixXf* _next_vertex_embeddings, MatrixXf* _d_next_vertex_embeddings,
        MatrixXf* _prev_vertex_embeddings, MatrixXf* _d_prev_vertex_embeddings,
        ArrayReducer* _reducer) :
                                          Parents(_Parents), GA(_GA),
                                          weights(_weights), d_weights(_d_weights),
                                          next_vertex_embeddings(_next_vertex_embeddings),
                                          d_next_vertex_embeddings(_d_next_vertex_embeddings),
                                          prev_vertex_embeddings(_prev_vertex_embeddings),
                                          d_prev_vertex_embeddings(_d_prev_vertex_embeddings),
                                          reducer(_reducer) {}
  // Invert the vertex RELU.
  inline bool operator() (uintE v) {
    for (int i = 0; i < next_vertex_embeddings[v].rows(); i++) {
      if (prev_vertex_embeddings[v](i,0) <= 0.0) {
        d_prev_vertex_embeddings[v](i,0) = 0.0;
      } else {
        d_prev_vertex_embeddings[v](i,0) = d_next_vertex_embeddings[v](i,0);
      }
      //d_next_vertex_embeddings[v](i,0);
    }
    return false;
  }
};


/***
   GCN Edge map using weight matrix.
***/

//template <class vertex>
//struct GCN_edgeMap_F {
//
//  graph<vertex>& GA;
//  uintE* Parents;
//  int64_t* vertexDegrees;
//  MatrixXf& weights;
//  MatrixXf& skip_weights;
//  MatrixXf* prev_vertex_embeddings;
//  MatrixXf* next_vertex_embeddings;
//
//  GCN_edgeMap_F(uintE* _Parents, int64_t* _vertexDegrees, graph<vertex>& _GA,
//        MatrixXf& _weights, MatrixXf& _skip_weights, MatrixXf* _next_vertex_embeddings,
//        MatrixXf* _prev_vertex_embeddings) : Parents(_Parents), vertexDegrees(_vertexDegrees), GA(_GA),
//                                          weights(_weights), skip_weights(_skip_weights),
//                                          next_vertex_embeddings(_next_vertex_embeddings),
//                                          prev_vertex_embeddings(_prev_vertex_embeddings) {}
//
//  // update with source s and destination d
//  // NOTE(TFK): This only works serially right now.
//  inline bool update(uintE s, uintE d) {
//    __sync_fetch_and_add(&tfk_debug_counter,1);
//    if (s == d) {
//      //return 0;
//      double edgeWeight = 1.0;//(1.0 / sqrt(GA.V[s].getOutDegree() + GA.V[d].getOutDegree()));
//      next_vertex_embeddings[s] += edgeWeight * (skip_weights * prev_vertex_embeddings[d]);
//    }//else {
//    //  //printf("s,d %d,%d\n", s,d);
//    //  assert(vertexDegrees[s] != 0);
//    //  assert(vertexDegrees[d] != 0);
//    //} /*else*/ 
//    {
//      //printf("regular edge degree %d %d \n", GA.V[s].getOutDegree() + GA.V[s].getInDegree(), GA.V[d].getOutDegree() + GA.V[d].getInDegree());
//      double edgeWeight = (1.0 / sqrt(vertexDegrees[s] * vertexDegrees[d]));
//      next_vertex_embeddings[s] += edgeWeight * (weights * prev_vertex_embeddings[d]);
//    }
//    return false;
//  }
//  inline bool updateAtomic(uintE s, uintE d) {
//    while (!__sync_bool_compare_and_swap(&Parents[s], UINT_E_MAX, 0)) {
//      //if (Parents[d] == 0) break;
//      continue;
//    }
//    bool ret = update(s,d);
//    __sync_bool_compare_and_swap(&Parents[s], 0, UINT_E_MAX);
//    return ret;
//  }
//  inline bool cond(uintE d) {
//    return cond_true(d);
//  }
//};
//
//template <class vertex>
//struct d_GCN_edgeMap_F {
//  graph<vertex>& GA;
//  uintE* Parents;
//  int64_t* vertexDegrees;
//  MatrixXf& weights;
//  MatrixXf* d_weights;
//  MatrixXf& skip_weights;
//  MatrixXf* d_skip_weights;
//
//  MatrixXf* prev_vertex_embeddings;
//  MatrixXf* d_prev_vertex_embeddings;
//  MatrixXf* next_vertex_embeddings;
//  MatrixXf* d_next_vertex_embeddings;
//
//  ArrayReducer* reducer;
//
//  d_GCN_edgeMap_F(uintE* _Parents, int64_t* _vertexDegrees, graph<vertex>& _GA,
//        MatrixXf& _weights, MatrixXf* _d_weights,
//        MatrixXf& _skip_weights, MatrixXf* _d_skip_weights,
//        MatrixXf* _next_vertex_embeddings, MatrixXf* _d_next_vertex_embeddings,
//        MatrixXf* _prev_vertex_embeddings, MatrixXf* _d_prev_vertex_embeddings,
//        ArrayReducer* _reducer) : Parents(_Parents), vertexDegrees(_vertexDegrees), GA(_GA), weights(_weights), d_weights(_d_weights),
//                                          skip_weights(_skip_weights), d_skip_weights(_d_skip_weights),
//                                          next_vertex_embeddings(_next_vertex_embeddings),
//                                          d_next_vertex_embeddings(_d_next_vertex_embeddings),
//                                          prev_vertex_embeddings(_prev_vertex_embeddings),
//                                          d_prev_vertex_embeddings(_d_prev_vertex_embeddings),
//                                          reducer(_reducer) {}
//
//  // update with source s and destination d
//  inline bool update(uintE s, uintE d) {
//
//    if (s == d) {
//      //return 0;
//      ArrayReducerView* view = reducer->view();
//      MatrixXf& d_skip_weights_view = *(view->get_view(d_skip_weights));
//      double edgeWeight = 1.0;//.(1.0 / sqrt(GA.V[s].getOutDegree() + GA.V[d].getOutDegree()));
//      for (int j = 0; j < weights.rows(); j++) {
//        if (d_next_vertex_embeddings[d](j,0) == 0.0) continue;
//        for (int k = 0; k < weights.cols(); k++) {
//          d_skip_weights_view(j,k) += edgeWeight*prev_vertex_embeddings[s](k,0) * d_next_vertex_embeddings[d](j,0);
//        }
//      }
//      MatrixXf& d_prev_vertex_embeddings_s = *(view->get_view(&(d_prev_vertex_embeddings[s])));
//      for (int j = 0; j < weights.rows(); j++) {
//        if (d_next_vertex_embeddings[d](j,0) == 0.0) continue;
//        for (int k = 0; k < weights.cols(); k++) {
//          d_prev_vertex_embeddings_s(k,0) += edgeWeight*skip_weights(j,k) * d_next_vertex_embeddings[d](j,0);
//        }
//      }
//    }// /*else*/
//    {
//      ArrayReducerView* view = reducer->view();
//      MatrixXf& d_weights_view = *(view->get_view(d_weights));
//      double edgeWeight = (1.0 / sqrt(vertexDegrees[s] * vertexDegrees[d]));
//      for (int j = 0; j < weights.rows(); j++) {
//        if (d_next_vertex_embeddings[s](j,0) == 0.0) continue;
//        for (int k = 0; k < weights.cols(); k++) {
//          d_weights_view(j,k) += edgeWeight*prev_vertex_embeddings[d](k,0) * d_next_vertex_embeddings[s](j,0);
//        }
//      }
//      MatrixXf& d_prev_vertex_embeddings_d = *(view->get_view(&(d_prev_vertex_embeddings[d])));
//      for (int j = 0; j < weights.rows(); j++) {
//        if (d_next_vertex_embeddings[s](j,0) == 0.0) continue;
//        for (int k = 0; k < weights.cols(); k++) {
//          d_prev_vertex_embeddings_d(k,0) += edgeWeight*weights(j,k) * d_next_vertex_embeddings[s](j,0);
//        }
//      }
//    }
//    return 0;
//  }
//
//  inline bool updateAtomic(uintE s, uintE d) {
//    return update(s,d);
//  }
//  inline bool cond(uintE d) {
//    return cond_true(d);
//  }
//};




