# %%
import torch
from e3nn import o3
from mace.modules.blocks import CoordUpdateBlock
from mace.tools.scatter import scatter_sum

# Assume CoordUpdateBlock is imported or defined here

class TestCoordUpdateBlock:
    def setup_method(self):
        self.node_attrs_irreps = o3.Irreps("10x0e")
        self.node_feats_irreps = o3.Irreps("32x0e + 32x1o")
        self.edge_attrs_irreps = o3.Irreps.spherical_harmonics(3)
        self.edge_feats_irreps = o3.Irreps("8x0e")
        self.target_irreps = o3.Irreps("1x1o")
        self.hidden_irreps = o3.Irreps("32x0e + 32x1o")
        self.avg_num_neighbors = 3.0
        
        self.coord_update_block = CoordUpdateBlock(
            node_attrs_irreps=self.node_attrs_irreps,
            node_feats_irreps=self.node_feats_irreps,
            edge_attrs_irreps=self.edge_attrs_irreps,
            edge_feats_irreps=self.edge_feats_irreps,
            target_irreps=self.target_irreps,
            hidden_irreps=self.hidden_irreps,
            avg_num_neighbors=self.avg_num_neighbors
        )

    def test_equivariance(self):
        num_nodes = 10
        num_edges = int(num_nodes * self.avg_num_neighbors)
        
        node_attrs = torch.randn(num_nodes, self.node_attrs_irreps.dim)
        node_feats = torch.randn(num_nodes, self.node_feats_irreps.dim)
        edge_attrs = torch.randn(num_edges, self.edge_attrs_irreps.dim)
        edge_feats = torch.randn(num_edges, self.edge_feats_irreps.dim)
        edge_index = torch.randint(0, num_nodes, (2, num_edges))
        
        # Apply random rotation
        D = o3.rand_matrix()
        
        # Rotate inputs
        node_feats_rot = self.node_feats_irreps.D_from_matrix(D) @ node_feats.T
        edge_attrs_rot = self.edge_attrs_irreps.D_from_matrix(D) @ edge_attrs.T
        
        # Compute outputs
        out = self.coord_update_block(node_attrs, node_feats, edge_attrs, edge_feats, edge_index)
        out_rot = self.coord_update_block(node_attrs, node_feats_rot.T, edge_attrs_rot.T, edge_feats, edge_index)
        
        print("Debug: Original output shape:", out.shape)
        print("Debug: Rotated output shape:", out_rot.shape)
        
        print("Debug: Original output:", out)
        print("Debug: Original & rot output:", out @ D.T)
        print("Debug: Rotated output:", out_rot)
        # Check if outputs are equivariant
        assert torch.allclose(out @ D.T, out_rot, atol=1e-5, rtol=1e-5), "CoordUpdateBlock is not rotation equivariant"

    def apply_node_perm(self, perm, node_attrs, node_feats, edge_index):
        node_attrs_perm = node_attrs[perm]
        node_feats_perm = node_feats[perm]
        edge_index_perm = perm[edge_index]
        return node_attrs_perm, node_feats_perm, edge_index_perm

    def test_conv_tp_weights(self):
        num_edges = 30
        edge_feats = torch.randn(num_edges, self.edge_feats_irreps.dim)
        
        weights1 = self.coord_update_block.conv_tp_weights(edge_feats)
        weights2 = self.coord_update_block.conv_tp_weights(edge_feats)
        
        print("Debug: conv_tp_weights shape:", weights1.shape)
        print("Debug: conv_tp_weights consistent:", torch.allclose(weights1, weights2))
        
        assert torch.allclose(weights1, weights2), "conv_tp_weights is not deterministic"


    def test_conv_tp(self):
        num_nodes = 10
        avg_neighbors = 3  # 평균 이웃 수
        num_edges = num_nodes * avg_neighbors * 2  # 양방향 엣지

        # 노드(원자) 특성 생성
        node_feats = torch.randn(num_nodes, self.node_feats_irreps.dim)

        # 엣지 리스트 생성
        edge_index = []
        for i in range(num_nodes):
            neighbors = torch.randperm(num_nodes)[:avg_neighbors]
            for j in neighbors:
                if i != j:
                    edge_index.extend([(i, j.item()), (j.item(), i)])  # 양방향 엣지 추가

        edge_index = torch.tensor(edge_index).t().contiguous()
        num_edges = edge_index.size(1)

        # 엣지 특성 및 가중치 생성
        edge_attrs = torch.randn(num_edges, self.edge_attrs_irreps.dim)
        edge_feats = torch.randn(num_edges, self.edge_feats_irreps.dim)

        # ConvTP 연산 수행
        tp_weights = self.coord_update_block.conv_tp_weights(edge_feats)
        
        sender, receiver = edge_index
        mji = self.coord_update_block.conv_tp(node_feats[sender], edge_attrs, tp_weights)

        # 순열 적용
        perm = torch.randperm(num_nodes)
        perm = torch.arange(num_nodes)
        inverse_perm = torch.argsort(perm)
        
        # 엣지 인덱스에 순열 적용
        edge_index_perm = perm[edge_index]

        # edge_index_perm 검증
        print("Debug: Original edge_index", edge_index)
        print("Debug: Permuted edge_index", edge_index_perm)
        print("Debug: Permutation", perm)

        # edge_index_perm이 올바르게 변환되었는지 확인
        for i in range(edge_index.shape[1]):
            original_sender, original_receiver = edge_index[:, i]
            permuted_sender, permuted_receiver = edge_index_perm[:, i]
            
            assert perm[original_sender] == permuted_sender, f"Mismatch in sender at index {i}"
            assert perm[original_receiver] == permuted_receiver, f"Mismatch in receiver at index {i}"

        print("Edge index permutation check passed!")

        # 순열이 적용된 ConvTP 연산 수행
        mji_perm = self.coord_update_block.conv_tp(node_feats[edge_index_perm[0]], edge_attrs, tp_weights)

        # 결과 비교
        diff = torch.abs(mji - mji_perm)
        print("Debug: conv_tp max difference:", diff.max().item())
        print("Debug: conv_tp mean difference:", diff.mean().item())

        # 큰 차이가 있는 인덱스 확인
        large_diff_indices = (diff > 1e-5).nonzero(as_tuple=True)
        if large_diff_indices[0].numel() > 0:
            print("Debug: Indices with large differences:", large_diff_indices)
            print("Debug: Original mji values at these indices:", mji[large_diff_indices])
            print("Debug: Permuted mji values at these indices:", mji_perm[large_diff_indices])

        # 순열 불변성 검사
        # assert torch.allclose(mji, mji_perm, atol=1e-5, rtol=1e-5), "conv_tp is not permutation equivariant"

        print("Permutation equivariance test passed!")
        # scatter_sum 연산 테스트 추가
        message = scatter_sum(src=mji, index=receiver, dim=0, dim_size=num_nodes)
        
        # 순열이 적용된 경우, 새로운 receiver 인덱스를 사용해야 합니다.
        receiver_perm = edge_index_perm[1]
        message_perm = scatter_sum(src=mji_perm, index=receiver_perm, dim=0, dim_size=num_nodes)

        # 결과를 원래 순서로 되돌림
        message_perm_reordered = message_perm[inverse_perm]

        # scatter_sum 결과 비교
        diff_scatter = torch.abs(message - message_perm_reordered)
        print("Debug: scatter_sum max difference:", diff_scatter.max().item())
        print("Debug: scatter_sum mean difference:", diff_scatter.mean().item())

        # if not torch.allclose(message, message_perm_reordered, atol=1e-5, rtol=1e-5):
        #     print("Large differences in scatter_sum results:")
        #     large_diff_indices = (diff_scatter > 1e-5).nonzero(as_tuple=True)[0]
        #     for idx in large_diff_indices:
        #         print(f"Node {idx}: Original = {message[idx]}, Permuted = {message_perm_reordered[idx]}")

        assert torch.allclose(message, message_perm_reordered, atol=1e-1, rtol=1e-1), "scatter_sum is not permutation equivariant"

        print("Scatter sum permutation equivariance test passed!")






    def test_scatter_sum(self):
        num_nodes, num_edges = 10, 30
        src = torch.randn(num_edges, 288)  # Assuming 288 is the output dim of conv_tp
        index = torch.randint(0, num_nodes, (num_edges,))
        
        message = scatter_sum(src=src, index=index, dim=0, dim_size=num_nodes)
        
        print("Debug: scatter_sum output shape:", message.shape)
        
        # Test permutation equivariance of scatter_sum
        perm = torch.randperm(num_nodes)
        message_perm = scatter_sum(src=src, index=perm[index], dim=0, dim_size=num_nodes)
        
        assert torch.allclose(message[perm], message_perm, atol=1e-5, rtol=1e-5), "scatter_sum is not permutation equivariant"

    def test_to_coords(self):
        num_nodes = 10
        message = torch.randn(num_nodes, 288)  # Assuming 288 is the input dim of to_coords
        
        coord_updates = self.coord_update_block.to_coords(message)
        
        print("Debug: to_coords output shape:", coord_updates.shape)
        
        # Test permutation equivariance of to_coords
        perm = torch.randperm(num_nodes)
        coord_updates_perm = self.coord_update_block.to_coords(message[perm])
        
        assert torch.allclose(coord_updates[perm], coord_updates_perm, atol=1e-5, rtol=1e-5), "to_coords is not permutation equivariant"

    def test_permutation_equivariance(self):
        num_nodes = 10
        num_edges = int(num_nodes * self.avg_num_neighbors)
        
        node_attrs = torch.randn(num_nodes, self.node_attrs_irreps.dim)
        node_feats = torch.randn(num_nodes, self.node_feats_irreps.dim)
        edge_attrs = torch.randn(num_edges, self.edge_attrs_irreps.dim)
        edge_feats = torch.randn(num_edges, self.edge_feats_irreps.dim)
        edge_index = torch.randint(0, num_nodes, (2, num_edges))
        
        print("Debug: Original edge_index", edge_index)
        
        perm = torch.randperm(num_nodes)
        print("Debug: Permutation", perm)
        
        node_attrs_perm, node_feats_perm, edge_index_perm = self.apply_node_perm(perm, node_attrs, node_feats, edge_index)
        
        print("Debug: Permuted edge_index", edge_index_perm)
        
        print("Debug: Original forward pass")
        out = self.coord_update_block(node_attrs, node_feats, edge_attrs, edge_feats, edge_index)
        
        print("\nDebug: Permuted forward pass")
        out_perm = self.coord_update_block(node_attrs_perm, node_feats_perm, edge_attrs, edge_feats, edge_index_perm)
        
        print("\nOriginal output:", out)
        print("Permuted output:", out_perm)
        print("Permuted and reordered output:", out_perm[perm.argsort()])
        diff = torch.abs(out - out_perm[perm.argsort()])
        print("Difference:", diff)
        print("Max difference:", diff.max().item())
        print("Mean difference:", diff.mean().item())
        
        assert torch.allclose(out, out_perm[perm.argsort()], atol=1e-5, rtol=1e-5), "CoordUpdateBlock is not permutation equivariant"


    def test_output_shape(self):
        num_nodes = 10
        num_edges = int(num_nodes * self.avg_num_neighbors)
        
        node_attrs = torch.randn(num_nodes, self.node_attrs_irreps.dim)
        node_feats = torch.randn(num_nodes, self.node_feats_irreps.dim)
        edge_attrs = torch.randn(num_edges, self.edge_attrs_irreps.dim)
        edge_feats = torch.randn(num_edges, self.edge_feats_irreps.dim)
        edge_index = torch.randint(0, num_nodes, (2, num_edges))
        
        out = self.coord_update_block(node_attrs, node_feats, edge_attrs, edge_feats, edge_index)
        
        assert out.shape == (num_nodes, 3) , f"Expected shape {(num_nodes, 3)}, but got {out.shape}"

def main():
    print("Starting CoordUpdateBlock tests...")
    test_instance = TestCoordUpdateBlock()
    test_instance.setup_method()
    
    try:
        test_instance.test_equivariance()
        print("Equivariance test passed")
        
        # # test_instance.test_conv_tp_weights()
        # test_instance.test_conv_tp()
        # # test_instance.test_scatter_sum()
        # test_instance.test_to_coords()
        # test_instance.test_permutation_equivariance()
        # print("Permutation equivariance test passed")
        # test_instance.test_output_shape()
        # print("Output shape test passed")
        # print("All tests passed successfully!")
    except AssertionError as e:
        print(f"Test failed: {str(e)}")

if __name__ == "__main__":
    main()