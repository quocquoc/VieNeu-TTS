import pytest
from unittest.mock import patch, MagicMock
from vieneu.factory import Vieneu

@patch("vieneu.turbo.TurboVieNeuTTS")
def test_factory_turbo(mock_turbo):
    Vieneu(mode="turbo")
    mock_turbo.assert_called_once()

@patch("vieneu.turbo.TurboGPUVieNeuTTS")
def test_factory_turbo_gpu(mock_turbo_gpu):
    Vieneu(mode="turbo_gpu")
    mock_turbo_gpu.assert_called_once()

@patch("vieneu.fast.FastVieNeuTTS")
def test_factory_fast(mock_fast):
    Vieneu(mode="fast")
    mock_fast.assert_called_once()

@patch("vieneu.standard.VieNeuTTS")
def test_factory_standard(mock_standard):
    Vieneu(mode="standard")
    mock_standard.assert_called_once()

@patch("vieneu.remote.RemoteVieNeuTTS")
def test_factory_remote(mock_remote):
    Vieneu(mode="remote")
    mock_remote.assert_called_once()

@patch("vieneu.core_xpu.XPUVieNeuTTS")
def test_factory_xpu(mock_xpu):
    Vieneu(mode="xpu")
    mock_xpu.assert_called_once()

def test_factory_invalid_mode():
    # Factory with unknown mode should return None (match-case without default)
    # The current implementation of factory.py ends with the last case
    assert Vieneu(mode="unknown") is None
