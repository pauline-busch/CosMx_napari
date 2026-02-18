import os
import numpy as np
import pandas as pd
import napari
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# ----------------------------
# USER SETTINGS
# ----------------------------

#POLYGONS_CSV = r"/Volumes/T7/CosMx/Datasets/6k/napari/napari_input/polygons/Remission_TL_TMA1_polygons.csv"
#TRANSCRIPTS_CSV = r"/Volumes/T7/CosMx/Datasets/6k/napari/napari_input/transcripts/Remission_TL_TMA1_tx.csv"

POLYGONS_CSV = r"F:/CosMx/Datasets/6k/napari/napari_input/polygons/Remission_TL_TMA1_polygons.csv"
TRANSCRIPTS_CSV = r"F:/CosMx/Datasets/6k/napari/napari_input/transcripts/Remission_TL_TMA1_tx.csv"

# Optional: load existing tissue annotation polygons if present (napari shapes CSV)
ANNOTATIONS_CSV = None
# ANNOTATIONS_CSV = r"/Volumes/T7/CosMx/Datasets/6k/napari_exports/annotations/anti-TNF-IR_TR_TMA1_annotations.csv"

# Optional: restrict to a gene subset for speed (set to None to load all)
# GENES_OF_INTEREST = None
GENES_OF_INTEREST = [
    "ADAMTS5",
    "CD55",
    "CLU",
    "CXCL1",
    "CXCL6",
    "DEFB1",
    "ERRFI1",
    "FGF10",
    "GPX3",
    "HBEGF",
    "IGFBP5",
    "INHBA",
    "ITGB8",
    "MMP1",
    "MMP3",
    "NTN4",
    "PCSK6",
    "PGF",
    "PRG4",
    "RCAN1",
    "SEMA3A",
    "SEMA3C",
    "SEMA5A",
    "SLC2A12",
    "SLC7A2",
    "SOX5",
    "TIMP3",
    "VEGFC"]

# GENES_OF_INTEREST = [
#     "AICDA", 
#     "BCL6", 
#     "CXCL13", 
#     "CXCR5", 
#     "MKI67", 
#     "TOP2A"
#     ]

# ----------------------------
# COLORS
# ----------------------------

celltype_colors = {
    "Fibroblasts": "#ff8000",
    "Macrophages": "#8208c4",
    "Endothelial.cells": "#990000",
    "Pericytes.Mural": "#c29a84",
    "B.cells": "#00d5ff",
    "T.NK.cells": "#1a34ff",
    "Plasmablasts": "#ff99cc",
    "Mast.cells": "#666633",
    "Neutrophils": "#aaff80",
    "Dendritic.cells": "#1b9e77",
    "Plasmacytoid.DCs": "#33a02c",
    "RBC": "#e60000",
    "Adipocytes": "#ffcc00",
    "Unassigned": "#cccccc",
}

annotation_colors = {
    "lining": "#df82e7",
    "vessel": "#00ff99",
    "TLO": "#2C89B4",
    "other": "#cccccc",
}

# ----------------------------
# HELPERS
# ----------------------------
def load_cell_polygons(polygons_csv: str):
    polys = pd.read_csv(polygons_csv)

    # Ensure numeric coords, coerce NAs like "NA" to real NaN
    for col in ["x_global_px", "y_global_px"]:
        polys[col] = pd.to_numeric(polys[col], errors="coerce")

    shapes = []
    cell_types = []
    cell_ids = []

    # groupby cell
    for cell, df in polys.groupby("cell", sort=False):
        coords = df[["y_global_px", "x_global_px"]].to_numpy(dtype=float)

        # Remove NaNs / Infs
        coords = coords[np.isfinite(coords).all(axis=1)]
        if coords.shape[0] < 3:
            continue

        # Remove duplicate points (keeping order)
        _, idx = np.unique(coords, axis=0, return_index=True)
        coords = coords[np.sort(idx)]
        if coords.shape[0] < 3:
            continue

        # Close polygon if needed
        if not np.allclose(coords[0], coords[-1]):
            coords = np.vstack([coords, coords[0]])

        shapes.append(coords)

        # cell type label (take first non-null)
        ct = df["InSituTypeID_refined"].dropna()
        ct = ct.iloc[0] if len(ct) else "Unassigned"

        cell_types.append(ct)
        cell_ids.append(cell)

    props = {"cell_type": np.array(cell_types, dtype=object),
             "cell_id": np.array(cell_ids, dtype=object)}

    return shapes, props


def load_transcripts(transcripts_csv: str, genes_of_interest=None):
    tx = pd.read_csv(transcripts_csv)

    # Ensure numeric coords, coerce "NA" -> NaN
    for col in ["x_global_px", "y_global_px"]:
        tx[col] = pd.to_numeric(tx[col], errors="coerce")

    tx = tx.dropna(subset=["x_global_px", "y_global_px", "target"])

    if genes_of_interest is not None:
        tx = tx[tx["target"].isin(genes_of_interest)].copy()

    points = tx[["y_global_px", "x_global_px"]].to_numpy(dtype=float)
    genes = tx["target"].astype(str).to_numpy(dtype=object)

    props = {"gene": genes}
    return points, props


def load_annotations_if_present(path: str):
    if path and os.path.exists(path):
        # napari shapes CSVs are readable via pandas, but easiest is:
        # let napari load it directly after creating viewer.
        return True
    return False


def ensure_parent_dir(path: str):
    if not path:
        return
    parent = os.path.dirname(path)
    if parent and not os.path.exists(parent):
        os.makedirs(parent, exist_ok=True)


# ----------------------------
# MAIN
# ----------------------------

def main():
    # ----------------------------
    # Load data
    # ----------------------------
    shapes, cell_props = load_cell_polygons(POLYGONS_CSV)
    tx_points, tx_props = load_transcripts(TRANSCRIPTS_CSV, GENES_OF_INTEREST)

    viewer = napari.Viewer()

    # ----------------------------
    # Cells layer (polygons)
    # ----------------------------
    cells_layer = viewer.add_shapes(
    shapes,
    shape_type="polygon",
    properties=cell_props,
    face_color="cell_type",
    edge_color="white",
    edge_width=0.3,
    name="Cells",
    opacity=0.5,
)

    celltype_colors_rgba = {
        k: tuple(float(x) for x in mcolors.to_rgba(v))
        for k, v in celltype_colors.items()
}

    face_colors = []
    for ct in cell_props["cell_type"]:
        if ct in celltype_colors_rgba:
            face_colors.append(celltype_colors_rgba[ct])
        else:
            # fallback color if something unexpected appears
            face_colors.append((0.7, 0.7, 0.7, 1.0))

    cells_layer.face_color = np.array(face_colors)

    # ----------------------------
    # Transcripts layer (points)
    # ----------------------------
    if len(tx_points) > 0:

        # Rebuild dataframe for easier grouping
        tx_df = pd.DataFrame({
            "y": tx_points[:, 0],
            "x": tx_points[:, 1],
            "gene": tx_props["gene"]
        })

        unique_genes = sorted(tx_df["gene"].unique())

        cmap = plt.get_cmap("tab20")

        for i, gene in enumerate(unique_genes):

            sub = tx_df[tx_df["gene"] == gene]

            coords = sub[["y", "x"]].to_numpy(dtype=float)

            if coords.shape[0] == 0:
                continue

            color = tuple(float(c) for c in cmap(i % 20))

            color_array = np.tile(np.array(color), (coords.shape[0], 1))

            viewer.add_points(
                coords,
                size=8,
                name=f"{gene}_tx",
                face_color=color_array,
                border_color=color_array,
                opacity=0.9,
            )

    # ----------------------------
    # Annotation layer (editable polygons)
    # ----------------------------
    ann_layer = viewer.add_shapes(
    name="Tissue_annotations",
    shape_type="polygon",
    properties={"annotation": []},
    face_color="annotation",
    edge_color="white",
    edge_width=2,
    opacity=0.35,
)

    ann_layer.face_color_mode = "cycle"
    ann_layer.face_color_cycle = annotation_colors

    # ----------------------------
    # Load existing annotations (if present)
    # ----------------------------
    if ANNOTATIONS_CSV and os.path.exists(ANNOTATIONS_CSV):
        loaded = viewer.open(ANNOTATIONS_CSV)
        loaded_layer = loaded[0]

        ann_layer.data = loaded_layer.data
        ann_layer.properties = loaded_layer.properties
        ann_layer.text = loaded_layer.text

        viewer.layers.remove(loaded_layer)

        annotation_colors_rgba = {
            k: tuple(float(x) for x in mcolors.to_rgba(v))
            for k, v in annotation_colors.items()
}

        ann_layer.face_color_cycle = annotation_colors_rgba
        ann_layer.face_color_mode = "cycle"

    # ----------------------------
    # HOTKEYS FOR LABELING ANNOTATIONS
    # ----------------------------

    def set_selected_annotation(label: str):
        layer = viewer.layers["Tissue_annotations"]
        selected = list(layer.selected_data)

        if not selected:
            print("No annotation shape selected.")
            return

        # ensure annotation property length matches number of shapes
        if "annotation" not in layer.properties or \
           len(layer.properties["annotation"]) != len(layer.data):
            layer.properties["annotation"] = np.array(
                [""] * len(layer.data), dtype=object
            )

        ann = layer.properties["annotation"].astype(object)

        for i in selected:
            ann[i] = label

        layer.properties["annotation"] = ann
        layer.refresh()

        print(f"Labeled {len(selected)} shape(s) as '{label}'.")

    @viewer.bind_key("1")
    def _(viewer):
        set_selected_annotation("lining")

    @viewer.bind_key("2")
    def _(viewer):
        set_selected_annotation("vessel")

    @viewer.bind_key("3")
    def _(viewer):
        set_selected_annotation("TLO")

    @viewer.bind_key("4")
    def _(viewer):
        set_selected_annotation("other")

    # ----------------------------
    # Save annotations (press "s")
    # ----------------------------
    @viewer.bind_key("s")
    def _(viewer):
        layer = viewer.layers["Tissue_annotations"]

        records = []

        for region_id, poly in enumerate(layer.data):
            region_type = layer.properties["annotation"][region_id]

            if region_type == "":
                continue

            for point in poly:
                records.append(
                    {
                        "region_id": region_id,
                        "region_type": region_type,
                        "y": float(point[0]),
                        "x": float(point[1]),
                    }
                )

        df = pd.DataFrame(records)

        out_path = POLYGONS_CSV.replace("_polygons.csv", "_annotation_polygons.csv")
        df.to_csv(out_path, index=False)

        print(f"Exported annotation polygons to {out_path}")

    # ----------------------------
    # Help (press "h")
    # ----------------------------
    @viewer.bind_key("h")
    def _(viewer):
        print(
            "\nHotkeys:\n"
            "  1 = lining\n"
            "  2 = vessel\n"
            "  3 = TLO\n"
            "  4 = other\n"
            "  s = save annotations\n"
            "  h = show help\n"
        )

    napari.run()

if __name__ == "__main__":
    main()