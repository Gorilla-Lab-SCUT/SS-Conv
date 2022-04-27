#ifndef DATATYPE_H
#define DATATYPE_H
#include <cstdint>
#include <array>
#include <vector>
#include <queue>
#include <google/dense_hash_map>

using Int = int32_t;
template <Int dimension> using Point = std::array<Int, dimension>;

template <Int dimension> struct IntArrayHash{
    std::size_t operator()(Point<dimension> const &p) const{
        Int hash = 16777619;
        for(auto x : p){
            hash *= 2166136261;
            hash ^= x;
        }
        return hash;
    }
};

template <Int dimension> using SparseGridMap = google::dense_hash_map<Point<dimension>, Int, IntArrayHash<dimension>, std::equal_to<Point<dimension>>>; // <Key, Data, HashFcn, EqualKey, Alloc>

template <Int dimension> class SparseGrid{
public:
    Int ctr;
    SparseGridMap<dimension> mp;
    SparseGrid();
};

template <Int dimension> using SparseGrids = std::vector<SparseGrid<dimension>>;

using RuleBook = std::vector<std::vector<Int>>;

class ConnectedComponent{
public:
    std::vector<Int> pt_idxs;

    ConnectedComponent();
    void addPoint(Int pt_idx);
};

using ConnectedComponents = std::vector<ConnectedComponent>;



// ------------------------------------------------------------------
template <Int dimension> SparseGrid<dimension>::SparseGrid() : ctr(0) {
    // Sparsehash needs a key to be set aside and never used
    Point<dimension> empty_key;
    for(Int i = 0; i < dimension; i++){
        empty_key[i] = std::numeric_limits<Int>::min();
    }
    mp.set_empty_key(empty_key);
}

#endif //DATATYPE_H