from graphviz import Digraph

# Create a flowchart diagram
dot = Digraph(comment="NN Outlier Detection Flow")

dot.attr(rankdir="LR", size="8,5")

# Nodes
dot.node("E", "Embeddings\n(CLIP vectors)", shape="box", style="rounded,filled", color="lightblue")
dot.node("AE", "Autoencoder\n(Encoder + Decoder)", shape="box", style="rounded,filled", color="lightgreen")
dot.node("R", "Reconstruction\nError", shape="box", style="rounded,filled", color="orange")
dot.node("O", "Outlier?", shape="diamond", style="filled", color="lightcoral")

# Edges
dot.edge("E", "AE")
dot.edge("AE", "R")
dot.edge("R", "O")

# Save and render
output_path = "/mnt/data/nn_outlier_detection_flowchart"
dot.render(output_path, format="png", cleanup=True)

output_path + ".png"
