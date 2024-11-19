from typing import Any, Dict

import pytest
import torch
import torch.nn.functional as F
from e3nn import o3

from mace import data, modules, tools
from e3nn.util import jit
from mace.cli.convert_cueq_e3nn import run as run_cueq_to_e3nn
from mace.cli.convert_e3nn_cueq import run as run_e3nn_to_cueq
from mace.tools import torch_geometric

try:
    import cuequivariance as cue  # pylint: disable=unused-import

    CUET_AVAILABLE = True
except ImportError:
    CUET_AVAILABLE = False

torch.set_default_dtype(torch.float64)


@pytest.mark.skipif(not CUET_AVAILABLE, reason="cuequivariance not installed")
class TestCueq:
    @pytest.fixture
    def model_config(self, interaction_cls_first, hidden_irreps) -> Dict[str, Any]:
        table = tools.AtomicNumberTable([6])
        print("interaction_cls_first", interaction_cls_first)
        print("hidden_irreps", hidden_irreps)
        return {
            "r_max": 5.0,
            "num_bessel": 8,
            "num_polynomial_cutoff": 6,
            "max_ell": 3,
            "interaction_cls": modules.interaction_classes[
                "RealAgnosticResidualInteractionBlock"
            ],
            "interaction_cls_first": interaction_cls_first,
            "num_interactions": 2,
            "num_elements": 1,
            "hidden_irreps": hidden_irreps,
            "MLP_irreps": o3.Irreps("16x0e"),
            "gate": F.silu,
            "atomic_energies": torch.tensor([1.0]),
            "avg_num_neighbors": 8,
            "atomic_numbers": table.zs,
            "correlation": 3,
            "radial_type": "bessel",
            "atomic_inter_scale": 1.0,
            "atomic_inter_shift": 0.0,
        }

    @pytest.fixture
    def batch(self, device: str):
        from ase import build

        table = tools.AtomicNumberTable([6])

        atoms = build.bulk("C", "diamond", a=3.567, cubic=True)
        import numpy as np

        displacement = np.random.uniform(-0.1, 0.1, size=atoms.positions.shape)
        atoms.positions += displacement
        atoms_list = [atoms.repeat((2, 2, 2))]

        configs = [data.config_from_atoms(atoms) for atoms in atoms_list]
        data_loader = torch_geometric.dataloader.DataLoader(
            dataset=[
                data.AtomicData.from_config(config, z_table=table, cutoff=5.0)
                for config in configs
            ],
            batch_size=1,
            shuffle=False,
            drop_last=False,
        )
        batch = next(iter(data_loader))
        return batch.to(device).to_dict()

    @pytest.mark.parametrize("device", ["cuda"])
    @pytest.mark.parametrize(
        "interaction_cls_first",
        [
            modules.interaction_classes["RealAgnosticResidualInteractionBlock"],
            modules.interaction_classes["RealAgnosticInteractionBlock"],
            modules.interaction_classes["RealAgnosticDensityInteractionBlock"],
        ],
    )
    @pytest.mark.parametrize(
        "hidden_irreps",
        [
            # o3.Irreps("32x0e + 32x1o"),
            # o3.Irreps("32x0e + 32x1o + 32x2e"),
            o3.Irreps("32x0e"),
        ],
    )
    def test_bidirectional_conversion(
        self,
        model_config: Dict[str, Any],
        batch: Dict[str, torch.Tensor],
        device: str,
    ):
        torch.manual_seed(42)

        # Create original E3nn model
        model_e3nn = modules.ScaleShiftMACE(**model_config)
        model_e3nn = model_e3nn.to(device)

        # Convert E3nn to CuEq
        model_cueq = run_e3nn_to_cueq(model_e3nn)
        model_cueq = model_cueq.to(device)

        # Convert CuEq back to E3nn
        model_e3nn_back = run_cueq_to_e3nn(model_cueq)
        model_e3nn_back = model_e3nn_back.to(device)

        # Test forward pass equivalence
        out_e3nn = model_e3nn(batch, training=True)
        out_cueq = model_cueq(batch, training=True)
        out_e3nn_back = model_e3nn_back(batch, training=True)

        # Check outputs match for both conversions
        torch.testing.assert_close(out_e3nn["energy"], out_cueq["energy"])
        torch.testing.assert_close(out_cueq["energy"], out_e3nn_back["energy"])
        torch.testing.assert_close(out_e3nn["forces"], out_cueq["forces"])
        torch.testing.assert_close(out_cueq["forces"], out_e3nn_back["forces"])

        # Test backward pass equivalence
        loss_e3nn = out_e3nn["energy"].sum()
        loss_cueq = out_cueq["energy"].sum()
        loss_e3nn_back = out_e3nn_back["energy"].sum()

        loss_e3nn.backward()
        loss_cueq.backward()
        loss_e3nn_back.backward()

        # Compare gradients for all conversions
        def print_gradient_diff(name1, p1, name2, p2, conv_type):
            if p1.grad is not None and p1.grad.shape == p2.grad.shape:
                if name1.split(".", 2)[:2] == name2.split(".", 2)[:2]:
                    error = torch.abs(p1.grad - p2.grad)
                    print(
                        f"{conv_type} - Parameter {name1}/{name2}, Max error: {error.max()}"
                    )
                    torch.testing.assert_close(p1.grad, p2.grad, atol=1e-5, rtol=1e-10)

        # E3nn to CuEq gradients
        for (name_e3nn, p_e3nn), (name_cueq, p_cueq) in zip(
            model_e3nn.named_parameters(), model_cueq.named_parameters()
        ):
            print_gradient_diff(name_e3nn, p_e3nn, name_cueq, p_cueq, "E3nn->CuEq")

        # CuEq to E3nn gradients
        for (name_cueq, p_cueq), (name_e3nn_back, p_e3nn_back) in zip(
            model_cueq.named_parameters(), model_e3nn_back.named_parameters()
        ):
            print_gradient_diff(
                name_cueq, p_cueq, name_e3nn_back, p_e3nn_back, "CuEq->E3nn"
            )

        # Full circle comparison (E3nn -> E3nn)
        for (name_e3nn, p_e3nn), (name_e3nn_back, p_e3nn_back) in zip(
            model_e3nn.named_parameters(), model_e3nn_back.named_parameters()
        ):
            print_gradient_diff(
                name_e3nn, p_e3nn, name_e3nn_back, p_e3nn_back, "Full circle"
            )
    
    # def test_jit_compile(
    #     self,
    #     model_config: Dict[str, Any],
    #     batch: Dict[str, torch.Tensor],
    #     device: str,
    # ):
    #     torch.manual_seed(42)

    #     # Create original E3nn model
    #     model_e3nn = modules.ScaleShiftMACE(**model_config)
    #     model_e3nn = model_e3nn.to(device)

    #     # Convert E3nn to CuEq
    #     model_cueq = run_e3nn_to_cueq(model_e3nn)
    #     model_cueq = model_cueq.to(device)

    #     # Convert CuEq back to E3nn
    #     model_e3nn_back = run_cueq_to_e3nn(model_cueq)
    #     model_e3nn_back = model_e3nn_back.to(device)

    #     # # Compile all models
    #     model_e3nn_compiled = jit.compile(model_e3nn)
    #     model_cueq_compiled = jit.compile(model_cueq)
    #     model_e3nn_back_compiled = jit.compile(model_e3nn_back)

    #     # Test forward pass equivalence
    #     out_e3nn = model_e3nn(batch, training=True)
    #     out_cueq = model_cueq(batch, training=True)
    #     out_e3nn_back = model_e3nn_back(batch, training=True)

    #     out_e3nn_compiled = model_e3nn_compiled(batch, training=True)
    #     out_cueq_compiled = model_cueq_compiled(batch, training=True)
    #     out_e3nn_back_compiled = model_e3nn_back_compiled(batch, training=True)

    #     # Check outputs match for both conversions
    #     torch.testing.assert_close(out_e3nn["energy"], out_cueq["energy"])
    #     torch.testing.assert_close(out_cueq["energy"], out_e3nn_back["energy"])
    #     torch.testing.assert_close(out_e3nn["forces"], out_cueq["forces"])
    #     torch.testing.assert_close(out_cueq["forces"], out_e3nn_back["forces"])

    #     torch.testing.assert_close(out_e3nn["energy"], out_e3nn_compiled["energy"])
    #     torch.testing.assert_close(out_cueq["energy"], out_cueq_compiled["energy"])
    #     torch.testing.assert_close(out_e3nn_back["energy"], out_e3nn_back_compiled["energy"])
    #     torch.testing.assert_close(out_e3nn["forces"], out_e3nn_compiled["forces"])
    #     torch.testing.assert_close(out_cueq["forces"], out_cueq_compiled["forces"])