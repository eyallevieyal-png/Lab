from src.memory import MemoryHierarchy


def test_qpu_transfer_is_supported():
    hierarchy = MemoryHierarchy()
    transfer_time = hierarchy.transfer_time(64 * 1024 * 1024, "DRAM", "QPU")
    assert transfer_time > 0
    assert "QPU" in hierarchy.tiers


def test_memory_hierarchy_has_expected_tiers():
    hierarchy = MemoryHierarchy()
    assert set(hierarchy.tiers) >= {"HBM", "DRAM", "CXL", "NVMe", "QPU"}
