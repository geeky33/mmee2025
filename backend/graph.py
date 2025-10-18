from graphviz import Digraph

# Create directed graph
dot = Digraph(comment="Outlier Detection Pipeline", format="png")
dot.attr(rankdir="LR", size="8")

# Input
dot.node("E", "Embeddings (Image + Text)\n[512D vectors]", shape="box", style="filled", fillcolor="lightblue")

# Feature spaces
dot.node("S1", "Raw joint (512D)", shape="ellipse", style="filled", fillcolor="lightyellow")
dot.node("S2", "PCA/t-SNE/UMAP\n(2D projection)", shape="ellipse", style="filled", fillcolor="lightyellow")

# Outlier methods
dot.node("M1", "Isolation Forest", shape="box", style="filled", fillcolor="lightgreen")
dot.node("M2", "Local Outlier Factor", shape="box", style="filled", fillcolor="lightgreen")
dot.node("M3", "kNN Quantile", shape="box", style="filled", fillcolor="lightgreen")
dot.node("M4", "DBSCAN (noise)", shape="box", style="filled", fillcolor="lightgreen")

# Output
dot.node("L", "Labels (0/1)\n• 1 = outlier\n• 0 = clean", shape="box", style="filled", fillcolor="lightpink")
dot.node("S", "Scores (float)\nHigher = more anomalous", shape="box", style="filled", fillcolor="lightpink")

# Evaluation
dot.node("GT", "Ground Truth Bad Captions\n(bad_caption_gt.json)", shape="box", style="filled", fillcolor="orange")
dot.node("CM", "Confusion Matrix\nTP / FP / FN / TN", shape="box", style="filled", fillcolor="lightgray")

# Connections
dot.edges([("E", "S1"), ("E", "S2")])
dot.edge("S1", "M1")
dot.edge("S1", "M2")
dot.edge("S1", "M3")
dot.edge("S1", "M4")
dot.edge("S2", "M1")
dot.edge("S2", "M2")
dot.edge("S2", "M3")
dot.edge("S2", "M4")

dot.edge("M1", "L")
dot.edge("M1", "S")
dot.edge("M2", "L")
dot.edge("M2", "S")
dot.edge("M3", "L")
dot.edge("M3", "S")
dot.edge("M4", "L")
dot.edge("M4", "S")

dot.edge("L", "CM")
dot.edge("GT", "CM")
dot.edge("S", "CM")

# Save and render
output_path = "/outlier_pipeline"
dot.render(output_path, cleanup=True)

output_path + ".png"
