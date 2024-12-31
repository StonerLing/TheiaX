// Copyright (C) 2015 The Regents of the University of California (Regents).
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
//     * Redistributions of source code must retain the above copyright
//       notice, this list of conditions and the following disclaimer.
//
//     * Redistributions in binary form must reproduce the above
//       copyright notice, this list of conditions and the following
//       disclaimer in the documentation and/or other materials provided
//       with the distribution.
//
//     * Neither the name of The Regents or University of California nor the
//       names of its contributors may be used to endorse or promote products
//       derived from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
//
// Please contact the author of this library if you have any questions.
// Author: Chris Sweeney (cmsweeney@cs.ucsb.edu)

#include <unordered_map>
#include <unordered_set>
#include <utility>

#include <gtest/gtest.h>

#include "theia/math/graph/normalized_graph_cut.h"
#include "theia/util/hash.h"

namespace theia {

// Create a simple scenario with 4 nodes that form a rectangle:
//    0 ------------------------- 5
//    |                           |
//    1 ------------------------- 6
//    |                           |
//    2 ------------------------- 7
//    |                           |
//    3 ------------------------- 8
//    |                           |
//    4 ------------------------- 9
// This should be very simple to partition.
TEST(NormalizedGraphCut, SimpleGraph) {
  typedef std::pair<int, int> IntPair;
  std::unordered_map<std::pair<int, int>, double> edge_weights;
  edge_weights.emplace(IntPair(0, 1), 1);
  edge_weights.emplace(IntPair(1, 2), 1);
  edge_weights.emplace(IntPair(2, 3), 1);
  edge_weights.emplace(IntPair(3, 4), 1);
  edge_weights.emplace(IntPair(5, 6), 1);
  edge_weights.emplace(IntPair(6, 7), 1);
  edge_weights.emplace(IntPair(7, 8), 1);
  edge_weights.emplace(IntPair(8, 9), 1);
  edge_weights.emplace(IntPair(0, 5), 0.01);
  edge_weights.emplace(IntPair(1, 6), 0.01);
  edge_weights.emplace(IntPair(2, 7), 0.01);
  edge_weights.emplace(IntPair(3, 8), 0.01);
  edge_weights.emplace(IntPair(4, 9), 0.01);

  NormalizedGraphCut<int>::Options options;
  NormalizedGraphCut<int> ncut(options);
  std::unordered_set<int> subgraph1, subgraph2;
  EXPECT_TRUE(ncut.ComputeCut(edge_weights, &subgraph1, &subgraph2, NULL));

  // Make sure that the subgraphs are split properly.
  EXPECT_EQ(subgraph1.size(), 5);
  EXPECT_EQ(subgraph2.size(), 5);

  const int node_0_subgraph = ContainsKey(subgraph1, 0) ? 1 : 2;
  const int node_1_subgraph = ContainsKey(subgraph1, 1) ? 1 : 2;
  const int node_2_subgraph = ContainsKey(subgraph1, 2) ? 1 : 2;
  const int node_3_subgraph = ContainsKey(subgraph1, 3) ? 1 : 2;
  const int node_4_subgraph = ContainsKey(subgraph1, 4) ? 1 : 2;
  const int node_5_subgraph = ContainsKey(subgraph1, 5) ? 1 : 2;
  const int node_6_subgraph = ContainsKey(subgraph1, 6) ? 1 : 2;
  const int node_7_subgraph = ContainsKey(subgraph1, 7) ? 1 : 2;
  const int node_8_subgraph = ContainsKey(subgraph1, 8) ? 1 : 2;
  const int node_9_subgraph = ContainsKey(subgraph1, 9) ? 1 : 2;

  EXPECT_EQ(node_0_subgraph, node_1_subgraph);
  EXPECT_EQ(node_1_subgraph, node_2_subgraph);
  EXPECT_EQ(node_2_subgraph, node_3_subgraph);
  EXPECT_EQ(node_3_subgraph, node_4_subgraph);
  EXPECT_EQ(node_5_subgraph, node_6_subgraph);
  EXPECT_EQ(node_6_subgraph, node_7_subgraph);
  EXPECT_EQ(node_7_subgraph, node_8_subgraph);
  EXPECT_EQ(node_8_subgraph, node_9_subgraph);
}

TEST(NormalizedGraphCut, SimpleGraph1) {
  typedef std::pair<int, int> IntPair;
  std::unordered_map<std::pair<int, int>, double> edge_weights;
  edge_weights.emplace(IntPair(1, 7), 100);
  edge_weights.emplace(IntPair(1, 4), 1);
  edge_weights.emplace(IntPair(1, 3), 100);
  edge_weights.emplace(IntPair(7, 3), 100);
  edge_weights.emplace(IntPair(3, 8), 1);
  edge_weights.emplace(IntPair(5, 4), 100);
  edge_weights.emplace(IntPair(5, 8), 100);
  edge_weights.emplace(IntPair(4, 8), 100);
  edge_weights.emplace(IntPair(2, 6), 80);
  edge_weights.emplace(IntPair(6, 9), 70);
  edge_weights.emplace(IntPair(6, 3), 40);
  edge_weights.emplace(IntPair(9, 8), 50);
  edge_weights.emplace(IntPair(2, 9), 20);
  edge_weights.emplace(IntPair(9, 7), 90);
  edge_weights.emplace(IntPair(2, 3), 110);
  edge_weights.emplace(IntPair(10, 1), 40);
  edge_weights.emplace(IntPair(10, 5), 60);
  edge_weights.emplace(IntPair(10, 6), 30);

  NormalizedGraphCut<int>::Options options;
  NormalizedGraphCut<int> ncut(options);
  std::unordered_set<int> subgraph1, subgraph2;
  EXPECT_TRUE(ncut.ComputeCut(edge_weights, &subgraph1, &subgraph2, NULL));
}

TEST(NormalizedGraphCut, FullyConnected) {
  typedef std::pair<int, int> IntPair;

  for (int num_nodes = 20; num_nodes < 50; ++num_nodes) {
    std::unordered_map<std::pair<int, int>, double> edge_weights;
    for (int i = 0; i < num_nodes; i++) {
      for (int j = i + 1; j < num_nodes; j++) {
        const double weight =
            std::max(5 * (i + j) + (i - j) * (i - j), 100) / 100.0;
        edge_weights[IntPair(i, j)] = weight;
      }
    }

    NormalizedGraphCut<int>::Options options;
    NormalizedGraphCut<int> ncut(options);
    std::unordered_set<int> subgraph1, subgraph2;
    EXPECT_TRUE(ncut.ComputeCut(edge_weights, &subgraph1, &subgraph2, NULL));
  }
}

}  // namespace theia
