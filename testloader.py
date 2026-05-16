from dataloader import create_dataloader

loader = create_dataloader(
    "processed",
    batch_size=2
)

for batch in loader:

    print(batch)

    print("\nNode Features:")
    print(batch.x.shape)

    print("\nEdges:")
    print(batch.edge_index.shape)

    print("\nTargets:")
    print(batch.y.shape)

    break