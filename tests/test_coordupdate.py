# %%
import torch
from e3nn import o3
from mace.modules.blocks import CoordUpdateBlock

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
        
        # Check if outputs are equivariant
        assert torch.allclose(out @ D.T, out_rot, atol=1e-5, rtol=1e-5), "CoordUpdateBlock is not rotation equivariant"

    def test_permutation_equivariance(self):
        num_nodes = 10
        num_edges = int(num_nodes * self.avg_num_neighbors)
        
        node_attrs = torch.randn(num_nodes, self.node_attrs_irreps.dim)
        node_feats = torch.randn(num_nodes, self.node_feats_irreps.dim)
        edge_attrs = torch.randn(num_edges, self.edge_attrs_irreps.dim)
        edge_feats = torch.randn(num_edges, self.edge_feats_irreps.dim)
        edge_index = torch.randint(0, num_nodes, (2, num_edges))
        
        perm = torch.randperm(num_nodes)
        
        node_attrs_perm = node_attrs[perm]
        node_feats_perm = node_feats[perm]
        edge_index_perm = perm[edge_index]
        
        out = self.coord_update_block(node_attrs, node_feats, edge_attrs, edge_feats, edge_index)
        out_perm = self.coord_update_block(node_attrs_perm, node_feats_perm, edge_attrs, edge_feats, edge_index_perm)
        
        print("Original output:", out)
        print("Permuted output:", out_perm)
        print("Permuted and reordered output:", out_perm[perm.argsort()])
        diff = torch.abs(out - out_perm[perm.argsort()])
        print("Difference:", diff)
        print("Max difference:", diff.max())
        print("Mean difference:", diff.mean())
        
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
        test_instance.test_permutation_equivariance()
        print("Permutation equivariance test passed")
        test_instance.test_output_shape()
        print("Output shape test passed")
        print("All tests passed successfully!")
    except AssertionError as e:
        print(f"Test failed: {str(e)}")

if __name__ == "__main__":
    main()