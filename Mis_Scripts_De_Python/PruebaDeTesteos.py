import pytest
import torch

torch.cuda.is_available()


class TestClass:
    age = 23

    def test_one(self):
        x = "this"
        assert "h" in x

    def test_two(self):
        x = "hello"
        assert hasattr(TestClass, "age")
