import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tenantkit.generate import generate_all, load_tenants  # noqa: E402

PLAN = ["easy", "medium", "hard"] * 4


@pytest.fixture(scope="session")
def small_data(tmp_path_factory):
    d = tmp_path_factory.mktemp("tenants")
    generate_all(d, n=12, plan=PLAN)
    return d


@pytest.fixture(scope="session")
def tenants(small_data):
    return load_tenants(small_data)


@pytest.fixture(scope="session")
def easy_tenant(tenants):
    return next(t for t in tenants if t.difficulty == "easy")


@pytest.fixture(scope="session")
def hard_tenant(tenants):
    return next(t for t in tenants if t.difficulty == "hard")
