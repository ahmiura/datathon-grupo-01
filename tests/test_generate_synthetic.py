from unittest.mock import patch
from src.data.generate_synthetic import generate_synthetic_financial_data

@patch("src.data.generate_synthetic.pd.DataFrame.to_csv")
@patch("src.data.generate_synthetic.os.makedirs")
def test_generate_synthetic(mock_makedirs, mock_to_csv):
    """Garante que a geração de dados sintéticos crie o CSV mockado"""
    generate_synthetic_financial_data("dummy_path/synthetic.csv", num_records=5)
    mock_to_csv.assert_called_once()