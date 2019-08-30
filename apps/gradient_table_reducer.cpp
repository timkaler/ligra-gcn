
class ArrayReducerView {
  public:
    std::map<Matrix*, Matrix*> views;

    ArrayReducerView() { views.clear(); }

    void accumulate(ArrayReducerView* right) {
      for (auto iter = right->views.begin(); iter != right->views.end(); ++iter) {
        if (this->views.find(iter->first) == this->views.end()) {
          this->views[iter->first] = new Matrix(iter->second->dimensions()[0], iter->second->dimensions()[1]);
          *(this->views[iter->first]) = *(iter->second);
        } else {
          *(this->views[iter->first]) += *(iter->second);
        }
        delete iter->second;
      }
    }

    bool has_view(Matrix* id) {
      return this->views.find(id) != this->views.end();
    }

    Matrix* get_view(Matrix* id) {
      if (this->views.find(id) == this->views.end()) {
        // need to create a new version of this Matrix view.
        this->views[id] = new Matrix(id->dimensions()[0], id->dimensions()[1]);
        zero_init(*(this->views[id]));
      }
      return this->views[id];
    }
};


struct _ArrayReducer : cilk::monoid_base<ArrayReducerView*> {
  public:
  static void reduce (ArrayReducerView** left, ArrayReducerView** right) {
    (*left)->accumulate(*(right));
  }

  static void identity (ArrayReducerView** p) {
    *p = new ArrayReducerView();
  }
};

typedef cilk::reducer<_ArrayReducer> ArrayReducer;

