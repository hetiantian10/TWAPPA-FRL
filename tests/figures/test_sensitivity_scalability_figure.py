from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
GENERATOR = ROOT / "article" / "figures" / "generate_sensitivity_scalability_summary.py"


def load_generator():
    assert GENERATOR.exists(), "Figure 3 must have a reproducible generator"
    spec = importlib.util.spec_from_file_location("sensitivity_figure", GENERATOR)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_figure_uses_all_table_s3_configurations_and_uncertainty():
    module = load_generator()
    rows = module.sensitivity_rows()

    assert len(rows) == 8
    assert {(row.clients, row.malicious_ratio) for row in rows} == {
        (5, 20),
        (5, 40),
        (10, 10),
        (10, 20),
        (10, 30),
        (20, 10),
        (20, 20),
        (20, 30),
    }
    assert all(row.accuracy_sd >= 0 for row in rows)
    assert all(row.mean_weight_sd >= 0 for row in rows)
    assert all(row.suppression_sd >= 0 for row in rows)
    assert next(
        row for row in rows if (row.clients, row.malicious_ratio) == (20, 30)
    ).suppression_mean == 0.0


def test_render_writes_single_column_png_and_vector_pdf(tmp_path):
    module = load_generator()
    png_path = tmp_path / "figure.png"
    pdf_path = tmp_path / "figure.pdf"

    figure = module.render(png_path=png_path, pdf_path=pdf_path)

    assert len(figure.axes) == 3
    assert png_path.read_bytes().startswith(b"\x89PNG")
    assert pdf_path.read_bytes().startswith(b"%PDF")


def test_panel_style_keys_use_requested_left_corner_anchors(tmp_path):
    module = load_generator()
    figure = module.render(
        png_path=tmp_path / "figure.png",
        pdf_path=tmp_path / "figure.pdf",
    )
    utility_ax, weight_ax, _ = figure.axes

    utility_key = next(
        text for text in utility_ax.texts if text.get_text().startswith("solid: accuracy")
    )
    assert utility_key.get_position() == (0.02, 0.04)
    assert utility_key.get_horizontalalignment() == "left"
    assert utility_key.get_verticalalignment() == "bottom"

    weight_key = next(
        text for text in weight_ax.texts if text.get_text().startswith("solid/filled: mean")
    )
    assert weight_key.get_position() == (0.02, 0.96)
    assert weight_key.get_horizontalalignment() == "left"
    assert weight_key.get_verticalalignment() == "top"
